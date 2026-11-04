"""把 Agent 从一条 CLI 命令变成一个有界面的服务。

## 为什么需要这一层

CLI 只能证明「能力存在」，不能证明「产品存在」。`cvagent ask` 在你的终端里跑得
通，不构成任何人可以打开的东西，也不构成一个能被追问「这答案谁给的」的界面。

## 三个必须在这一层解决的问题

**1. 图片路径不能由前端决定。**
`pipeline.extract()` 从问题文本里抽路径（绝对路径照收；相对路径则要真的存在
于磁盘才算），抽到就交给 `detect_cabbage` 读盘。如果前端能把任意路径写进问题，
这个功能就等于一次任意文件读取——而且错误信息里还会带出文件是否存在。所以上传
之后只下发 `image_id`，真实路径留在服务端的映射表里，问题文本里的路径由服务端
自己拼，用户写的那部分先经 `strip_paths` 抹掉。

**2. 分层必须在界面上可见。**
这个项目的主张是「规则能兜住的不要调 LLM」。如果页面只显示一个答案，用户
无从判断它花了 0.13 秒还是 11 秒、有没有编。所以响应里带 `route`、
`escalate_reason` 和每一步工具调用，界面把「规则 / Agent」并排摆出来——
分层是运行时状态，不是 README 里的一句话。

**3. 推理要串行，但等 LLM 不能占锁。**
只有一张 GPU，ultralytics 同一个 model 并发 predict 会互相踩显存。但 Agent 等
LLM 的那十几秒不该堵住下一个人。所以锁只包 `dispatch`（工具执行），不包整个
请求；请求本身丢进线程池，事件循环保持可用。

跑法：

    cvagent serve --port 8000
    # 或
    uvicorn cvagent.serve:app --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import ROOT
from .pipeline import strip_paths

# ── 可调项 ──────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("CVAGENT_UPLOADS", ROOT / "runs" / "uploads"))
MAX_UPLOAD_MB = int(os.getenv("CVAGENT_MAX_UPLOAD_MB", "16"))
MAX_QUESTION_CHARS = int(os.getenv("CVAGENT_MAX_QUESTION", "2000"))
MAX_UPLOADS = 32  # 内存映射表上限，超出按 LRU 淘汰
WEB_DIR = Path(__file__).parent / "web"

# 魔术字节 → 扩展名。不信任 Content-Type 和文件名：浏览器给的 MIME 可以伪造，
# 而后面 engine.validate() 是按扩展名白名单放行的，两者必须对得上。
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)


def _sniff_ext(data: bytes) -> str | None:
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _scrub_paths(text: str) -> str:
    """把用户输入里的路径抹掉，只留服务端注入的那一条。

    注意这里抹的是**用户写的**文本，抹完才拼接服务端路径，顺序不能反。

    真正干活的是 `pipeline.strip_paths()`：路径"长什么样"只能有一份定义。
    这一层自己写正则的后果实测过——pipeline 加了相对路径之后，这里的正则还是
    只认绝对路径，于是一句 `runs/uploads/x.jpg 里有多少株` 就能让服务端去读
    用户指定的文件。判据合并之后这类偏差不会再出现。
    """
    return strip_paths(text)


# ── 上传表 ──────────────────────────────────────────────────────────────
_uploads: dict[str, Path] = {}
_upload_order: list[str] = []
_uploads_lock = threading.Lock()

# 一个浏览器会话最近用什么图。只存 image_id（12 位指纹），不存路径。
_sessions: dict[str, str] = {}


def _remember_upload(iid: str, path: Path) -> None:
    with _uploads_lock:
        _uploads[iid] = path
        if iid in _upload_order:
            _upload_order.remove(iid)
        _upload_order.append(iid)
        while len(_upload_order) > MAX_UPLOADS:
            old = _upload_order.pop(0)
            _uploads.pop(old, None)


def _lookup(iid: str) -> Path | None:
    with _uploads_lock:
        return _uploads.get(iid)


# ── 推理资源 ────────────────────────────────────────────────────────────
# 单例：加载一次模型，全进程共用。`_state` 记的是加载结果而不是异常，
# 因为服务不能因为权重缺失就起不来——页面得能打开，让人自己看到为什么不能用。
_state: dict = {"ready": False, "error": "", "warmup_ms": 0.0, "device": "", "weights": ""}
# CLI 的 --weights / --device 在 uvicorn 启动前塞进来。用参数而不是改环境变量，
# 是因为 config 是 import 时求值的，那时再 setenv 已经晚了。
_engine_overrides: dict = {}
# 模型侧同理。写死成 config 默认值是不够的：默认模型可能没有额度，
# 而有额度的那个往往不叫默认名——这件事必须在启动参数里能改。
_llm_overrides: dict = {}
_engine = None
_tools = None
_infer_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job")


class _LockedTools:
    """给 registry 套一把锁，只锁工具执行，不锁 LLM 等待。

    Pipeline 与 Agent 都只用到 dispatch/schemas，代理这两个就够了——
    多包一层也不会让它们的行为有任何差别。
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def dispatch(self, name, raw_args):
        with _infer_lock:
            return self._inner.dispatch(name, raw_args)

    def schemas(self):
        return self._inner.schemas()

    def describe(self):
        return self._inner.describe()

    def names(self):
        return self._inner.names()


def _init_engine(weights: str | None = None, device: str | None = None) -> None:
    """加载权重并预热。失败只记录，不抛。"""
    global _engine, _tools
    from .tools.registry import default_registry
    from .vision.engine import YoloEngine

    try:
        engine = YoloEngine(weights=weights or None, device=device or None)
        t0 = time.perf_counter()
        # 首次调用含 cuDNN autotune，实测 20~30s。不挪到启动阶段，第一个
        # 真实请求就要背这段时间，前端只会看到「一直在转圈」。
        ms = engine.warmup()
        _state.update(
            ready=True, error="", warmup_ms=round(ms, 1),
            device=engine.device, weights=engine.weights.name,
        )
        _engine = engine
        _tools = _LockedTools(default_registry(engine))
        print(f"[serve] 模型就绪 {engine.weights.name} device={engine.device} "
              f"预热 {ms / 1000:.1f}s（总 {time.perf_counter() - t0:.1f}s）", flush=True)
    except Exception as e:  # noqa: BLE001 - 启动失败要把原因带到界面上
        _state.update(ready=False, error=f"{type(e).__name__}: {e}")
        print(f"[serve] 模型不可用：{e}", flush=True)


# ── 请求/响应模型 ────────────────────────────────────────────────────────
class UploadResult(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    bytes: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    image_id: str | None = None
    session: str | None = Field(None, max_length=64)
    mode: str = Field("auto", pattern="^(auto|pipeline|agent)$")


def _build_question(req: AskRequest) -> str:
    """把问题拼成 pipeline 与 Agent 都能吃的形态。

    先抹用户写的路径，再拼服务端路径——顺序反了就等于自己开后门。
    """
    q = _scrub_paths(req.question.strip())
    iid = req.image_id or (_sessions.get(req.session) if req.session else None)
    if not iid:
        return q
    path = _lookup(iid)
    if path is None:
        raise HTTPException(410, "这张图已从服务端清理，请重新上传")
    if req.session:
        _sessions[req.session] = iid
    return f"{q}\n图片路径：{path}"


def _slim_obs(obs: dict) -> dict:
    """观测瘦身。前端要的是「结论长什么样」，不是 300 个框坐标。"""
    out: dict = {}
    for k, v in obs.items():
        if isinstance(v, list) and len(v) > 8:
            out[k] = f"<{len(v)} 项省略>"
        elif isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + "…"
        else:
            out[k] = v
    return out


def _boxes_for(traj) -> list[list[float]]:
    """从 engine 缓存里取**整张图**的框，供前端画。

    为什么不在工具返回里取：`detect_cabbage` 默认 summary 只给前 3 个框，
    那是模型的上下文预算，不是界面的。而缓存里本来就存着全量结果，取一次
    不花推理时间，也保证「画出来的框数」和「答案里的株数」出自同一份数据。
    """
    if _engine is None:
        return []
    iid = next(
        (s.observation["image_id"] for s in reversed(traj.steps)
         if s.ok and isinstance(s.observation.get("image_id"), str)),
        None,
    )
    if not iid:
        return []
    hit = _engine.get_cached(iid)
    if hit is None:
        return []
    return [[round(b.x1), round(b.y1), round(b.x2), round(b.y2), round(b.conf, 3)]
            for b in hit.boxes[:400]]


def _pack(traj, route: str, reason: str | None, t0: float) -> dict:
    tot = traj.totals() if traj.turns else {}
    tool_ms = tot.get("tool_ms") or sum(s.elapsed_ms for s in traj.steps)
    return {
        "answer": traj.answer,
        "route": route,
        "escalate_reason": reason,
        "stopped_reason": traj.stopped_reason,
        "model": traj.model or "—",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "boxes": _boxes_for(traj),
        "steps": [
            {
                "index": s.index,
                "turn": s.turn,
                "tool": s.tool,
                "args": {k: v for k, v in s.args.items() if k != "image_path"},
                "ok": s.ok,
                "elapsed_ms": round(s.elapsed_ms, 1),
                "brief": s.brief(),
                "observation": _slim_obs(s.observation),
            }
            for s in traj.steps
        ],
        "stats": {
            "n_tool_calls": traj.n_tool_calls,
            "n_turns": traj.n_turns,
            "n_failed_calls": traj.n_failed_calls,
            "llm_ms": round(float(tot.get("llm_ms", 0.0)), 1),
            "tool_ms": round(float(tool_ms), 1),
            "prompt_tokens": int(tot.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(tot.get("completion_tokens", 0) or 0),
        },
    }


def _as_http(e: Exception) -> HTTPException:
    """把内部异常翻译成状态码。

    供应商失败用 502 而不是 500：这不是本服务的 bug，换个 provider 或等额度
    恢复就能好，调用方不该为此来翻我们的日志。规则层完全不受影响——LLM 挂了，
    命中的规则照样出答案，这是分层最实用的一处副产品。
    """
    from .llm.base import ProviderError

    if isinstance(e, ProviderError):
        return HTTPException(502, f"模型侧失败：{e}")
    return HTTPException(500, f"{type(e).__name__}: {e}")


def _run(q: str, mode: str, emit) -> dict:
    """一次完整请求：规则层 → 判断是否需要 Agent → 交付。全部同步，跑在线程里。"""
    if _tools is None or not _state["ready"]:
        raise HTTPException(503, f"检测模型不可用：{_state['error'] or '尚未加载'}")

    from .agent.loop import Agent
    from .pipeline import Pipeline, escalate_reason, handoff

    t0 = time.perf_counter()

    if mode in ("auto", "pipeline"):
        pre = Pipeline(_tools).run(q)
        reason = escalate_reason(pre)
        if reason is None or mode == "pipeline":
            return _pack(pre, "pipeline", None, t0)
        emit({"type": "route", "route": "agent", "reason": reason})
        extra = handoff(pre)
    else:
        reason = None
        extra = ""

    from .llm.registry import build_provider

    provider = build_provider(_llm_overrides.get("provider"), _llm_overrides.get("model"))
    agent = Agent(
        provider,
        _tools,
        verbose=False,
        on_step=lambda s: emit({
            "type": "step", "index": s.index, "tool": s.tool,
            "ok": s.ok, "elapsed_ms": round(s.elapsed_ms, 1), "brief": s.brief(),
        }),
    )
    traj = agent.run(
        q, extra_context=extra,
        on_delta=lambda piece: emit({"type": "delta", "text": piece}),
    )
    return _pack(traj, "agent", reason, t0)


# ── 应用 ────────────────────────────────────────────────────────────────
async def _lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_init_engine, **_engine_overrides)
    yield
    _pool.shutdown(wait=False)


app = FastAPI(
    title="甘蓝视觉 Agent",
    description="分层问答：规则优先，判断不了才升级到 LLM Agent",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.post("/api/upload", response_model=UploadResult)
async def upload(file: UploadFile = File(...)) -> UploadResult:
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"超过 {MAX_UPLOAD_MB}MB 限制")

    ext = _sniff_ext(data)
    if ext is None:
        raise HTTPException(415, "不是可识别的图片格式（支持 jpg/png/bmp/tif/webp）")

    # 内容指纹当 id：同一张图重复上传不会产生新文件，也让 refine 的缓存
    # 能跨请求命中（engine 的缓存键就是文件指纹）。
    iid = hashlib.sha1(data).hexdigest()[:12]
    path = UPLOAD_DIR / f"{iid}{ext}"
    if not path.exists():
        await asyncio.to_thread(path.write_bytes, data)

    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, f"图片无法解码：{e}") from e

    _remember_upload(iid, path)
    return UploadResult(image_id=iid, filename=file.filename or path.name,
                        width=w, height=h, bytes=len(data))


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    q = _build_question(req)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_pool, _run, q, req.mode, lambda _e: None)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise _as_http(e) from e


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE。Agent 路径要十几秒，没有中间态用户会以为卡死。

    事件类型：route（决定升级）/ step（一次工具调用结束）/ delta（模型吐字）
    / done（完整结果）/ error。
    """
    q = _build_question(req)
    events: queue.Queue = queue.Queue()

    def emit(ev: dict) -> None:
        events.put(ev)

    def work() -> None:
        try:
            events.put({"type": "done", **_run(q, req.mode, emit)})
        except HTTPException as e:
            events.put({"type": "error", "message": e.detail, "status": e.status_code})
        except Exception as e:  # noqa: BLE001
            h = _as_http(e)
            events.put({"type": "error", "message": h.detail, "status": h.status_code})
        finally:
            events.put(None)  # 哨兵：无论成功失败都让生成器退出

    asyncio.get_running_loop().run_in_executor(_pool, work)

    async def gen():
        yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"
        while True:
            try:
                ev = await asyncio.to_thread(events.get, True, 10.0)
            except queue.Empty:
                yield ": ping\n\n"  # 长时间无事件时保活，防中间层掐断
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sample")
async def sample():
    """示例图。让人不用先去找一张航拍图就能把流程走完。"""
    from fastapi.responses import FileResponse

    p = Path(os.getenv("CVAGENT_SAMPLE", ROOT / "examples" / "sample.jpg"))
    if not p.exists():
        raise HTTPException(404, "示例图不存在，请设置 CVAGENT_SAMPLE 指向一张航拍图")
    return FileResponse(p, media_type="image/jpeg", filename="sample.jpg")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok" if _state["ready"] else "degraded",
        "ready": _state["ready"],
        "error": _state["error"],
        "weights": _state["weights"],
        "device": _state["device"],
        "warmup_ms": _state["warmup_ms"],
        "uploads": len(_upload_order),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cvagent.serve:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
