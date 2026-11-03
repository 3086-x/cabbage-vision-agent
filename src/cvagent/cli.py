"""命令行入口。

`ask` 是 Agent 路径，`detect` 是不经 LLM 的直接推理路径。后者存在的意义：
出问题时能把"模型不会用工具"和"检测本身有问题"立刻分开，不用猜。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def cmd_ask(args) -> int:
    """分层入口：规则先跑，判断不了才升级到 Agent。

    两条路径共用同一个 registry、同一套工具契约，所以"谁答的"只在最后的
    统计行里体现，答案格式对调用方是一致的。
    """
    from .agent.loop import Agent
    from .errors import ToolError
    from .pipeline import Pipeline, escalate_reason
    from .session import Session
    from .tools.registry import default_registry

    engine = _engine(args)
    registry = default_registry(engine)

    question = args.question
    if args.image:
        question = f"{question}\n图片路径：{args.image}"

    session = None
    extra_context = ""
    if args.session:
        try:
            session = Session.load(args.session)
        except ToolError as e:
            print(f"会话名无效：{e.message}\n  → {e.hint}", file=sys.stderr)
            return 2
        if args.new_session:
            session = Session(id=args.session)
        extra_context = session.context_for(question)

    print(f"问题：{args.question}")
    if args.image:
        print(f"图片：{args.image}")
    if extra_context:
        print(f"会话：{session.id}（已加载 {len(session.turns)} 轮历史）")
    print("正在加载检测模型…", flush=True)
    load_ms = engine.warmup()
    print(f"就绪（加载 {load_ms / 1000:.1f}s，后续单图推理约 0.1s）")
    print("─" * 60)

    # ── 第一层：确定性规则。命中就直接交付，不构建 provider、不发请求 ──
    if args.mode in ("auto", "pipeline"):
        pre = Pipeline(registry).run(question)
        reason = escalate_reason(pre)
        if reason is None:
            print("[规则] 命中，未调用 LLM")
            return _deliver(pre, args, session)
        if args.mode == "pipeline":
            print(f"[规则] 未走通：{reason}（--mode pipeline 不升级）")
            return _deliver(pre, args, session)
        print(f"[规则 → Agent] {reason}")
        # 把已经做过的动作交代给 Agent，免得它把同一次检测再跑一遍。
        # 重复调用并不贵：engine 有结果缓存，命中只是一次 dispatch，不是一次推理。
        extra_context = _handoff(pre, extra_context)

    # ── 第二层：Agent 循环。只有上一层判断不了时才进得来 ──
    from .llm.registry import build_provider

    provider = build_provider(args.provider, args.model)
    print(f"模型：{provider.name}/{provider.model}")

    def on_delta(piece: str) -> None:
        print(piece, end="", flush=True)

    agent = Agent(provider, registry, max_steps=args.max_steps, verbose=True)
    try:
        traj = agent.run(question, extra_context=extra_context,
                         on_delta=on_delta if args.stream else None)
    except Exception as e:  # noqa: BLE001
        print(f"\n运行失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.stream:
        print()
    return _deliver(traj, args, session)


def _handoff(pre, extra_context: str) -> str:
    """把规则层已经做过的动作，写成一段上下文交给 Agent。"""
    from .pipeline import handoff

    return "\n".join(p for p in (handoff(pre), extra_context) if p)


def _deliver(traj, args, session) -> int:
    """两条路径共用的收尾：打印答案、落会话、存轨迹。"""
    from .report import render_markdown, save
    from .session import Turn

    print("─" * 60)
    print(traj.answer)
    print("─" * 60)

    if traj.stopped_reason.startswith("pipeline"):
        # pipeline 没有决策轮，turns 是空的，用 totals() 会显示 0ms。
        # 这里的耗时直接取各个工具的实测值。
        tool_ms = sum(s.elapsed_ms for s in traj.steps)
        print(
            f"工具 {traj.n_tool_calls} 次；工具 {tool_ms:.0f}ms；"
            f"总 {traj.elapsed_s * 1000:.0f}ms；路径：规则（0 次 LLM）；"
            f"终止：{traj.stopped_reason}"
        )
    else:
        tot = traj.totals()
        print(
            f"工具 {traj.n_tool_calls} 次（失败 {traj.n_failed_calls}），{traj.n_turns} 轮；"
            f"LLM {tot['llm_ms']:.0f}ms / 工具 {tot['tool_ms']:.0f}ms；"
            f"token {tot['prompt_tokens']} in / {tot['completion_tokens']} out；"
            f"重试 {tot['retries']}；总 {traj.elapsed_s:.1f}s；终止：{traj.stopped_reason}"
        )

    if session is not None:
        count = next((s.observation.get("count") for s in reversed(traj.steps)
                      if isinstance(s.observation.get("count"), int)), None)
        session.append(Turn(q=args.question, a=traj.answer,
                            image=args.image or "", count=count, model=traj.model))
        print(f"会话已保存：{session.save()}（第 {len(session.turns)} 轮）")

    if not args.no_save:
        out = save(traj)
        print(f"已保存：{out}")
        if args.markdown:
            print()
            print(render_markdown(traj))
    else:
        print()
        print(render_markdown(traj))

    # 4 = 规则认不出、但被 --mode pipeline 拦住没升级。脚本可以据此改走 Agent。
    if traj.stopped_reason == "pipeline_unrouted":
        return 4
    return 0 if traj.answer and traj.stopped_reason != "max_steps" else 3


def cmd_detect(args) -> int:
    """不经 LLM，直接调工具。用于冒烟与调试。"""
    from .tools.registry import default_registry

    registry = default_registry(_engine(args))
    payload = registry.dispatch(
        "detect_cabbage",
        {"image_path": args.image, "conf": args.conf, "detail": "full" if args.full else "summary"},
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") else 1


def cmd_tools(args) -> int:
    from .tools.registry import default_registry

    registry = default_registry(_engine(args))
    if args.json:
        print(json.dumps(registry.schemas(), ensure_ascii=False, indent=2))
        return 0
    for t in registry.describe():
        print(f"● {t['name']}")
        print(f"  {t['description']}")
        print(f"  必需参数：{', '.join(t['required']) or '无'}")
        if t["tags"]:
            print(f"  标签：{', '.join(t['tags'])}")
        print()
    print(f"共 {len(registry.describe())} 个工具")
    return 0


def cmd_doctor(args) -> int:
    from . import config
    from .llm.registry import available_providers, build_provider, probe
    from .tools.registry import default_registry

    ok = True

    # 1 权重
    w = Path(args.weights) if args.weights else config.WEIGHTS
    if w.exists():
        print(f"[✓] 权重存在：{w}  ({w.stat().st_size / 1e6:.1f} MB)")
    else:
        ok = False
        print(f"[✗] 权重不存在：{w}")
        print("    运行 python -m cvagent setup-weights --from <主仓库 best.pt> 或设置 CVAGENT_WEIGHTS")

    # 2 推理环境
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"[✓] torch {torch.__version__}  CUDA={'可用' if cuda else '不可用（将用 CPU）'}")
        if cuda:
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        ok = False
        print(f"[✗] torch 不可用：{e}")

    # 3 工具
    try:
        reg = default_registry(_engine(args))
        print(f"[✓] 工具 {len(reg.names())} 个：{', '.join(reg.names())}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[✗] 工具初始化失败：{e}")

    # 4 LLM
    found = available_providers()
    print(f"[{'✓' if found else '!'}] 检测到 key 的 provider：{', '.join(found) or '无'}")
    if args.probe and found:
        try:
            p = build_provider(args.provider or "auto")
            good, msg = probe(p)
            print(f"[{'✓' if good else '✗'}] 探测 {p.name}：{msg}")
            ok = ok and good
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[✗] 探测失败：{e}")
    elif not found:
        print("    可用 CVAGENT_LLM=mock 离线跑通决策链，无需 key")

    print()
    print("结论：" + ("环境可运行" if ok else "存在阻塞项，见上方 [✗]"))
    return 0 if ok else 1


def cmd_models(args) -> int:
    """探测哪些模型名在当前 key 下真的可用，并验证是否支持 function calling。

    存在的理由很具体：同一个 key 下，不同模型的配额、开通状态、是否支持 tools
    完全不同，而报错码还各不相同（403 FreeTierOnly / 429 limit_requests /
    404 model_not_found / 403 access_denied）。靠逐个手写 curl 试太慢，
    而"key 是不是没给我"这类误判会让排查走错方向。
    """
    from .llm.registry import build_provider

    names = [m.strip() for m in (args.candidates or "").split(",") if m.strip()]
    if not names:
        names = list(DEFAULT_CANDIDATES)

    tool = {
        "type": "function",
        "function": {
            "name": "count_cabbage",
            "description": "统计图片中的甘蓝株数",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string", "description": "图片路径"}},
                "required": ["image_path"],
            },
        },
    }
    from .llm.base import Message

    print(f"{'模型':<26}{'对话':<6}{'工具调用':<8}备注")
    print("─" * 78)
    usable: list[str] = []
    for name in names:
        try:
            p = build_provider(args.provider, name)
            c = p.complete([Message("user", "数一下这张图有几株甘蓝：/tmp/a.jpg")], [tool])
            if c.tool_calls and c.tool_calls[0].name == "count_cabbage":
                print(f"{name:<26}{'✓':<6}{'✓':<8}{c.tool_calls[0].arguments[:40]}")
                usable.append(name)
            else:
                txt = (c.content or "").strip()
                print(f"{name:<26}{'✓':<6}{'✗':<8}未调用工具：{txt[:40]}")
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None) or type(e).__name__
            hint = (getattr(e, "hint", None) or "").split("。")[0][:44]
            print(f"{name:<26}{'✗':<6}{'-':<8}{code} {hint}")

    print("─" * 78)
    if usable:
        print(f"可用于 Agent 的模型（{len(usable)} 个）：{', '.join(usable)}")
        print(f"用法：cvagent ask \"...\" --provider {args.provider or 'dashscope'} --model {usable[0]}")
        print(f"或写入环境变量：CVAGENT_MODEL={usable[0]}")
    else:
        print("没有模型既能对话又支持 function calling，Agent 路径无法运行。")
    return 0 if usable else 1


# 只用来看图说话的名字容易被漏掉，这里给一份常见候选；可用 --candidates 覆盖。
DEFAULT_CANDIDATES = (
    "qwen3.8-flash", "qwen3.8-27b", "qwen3.8-max-0902",
    "deepseek-v4.1-flash", "kimi-k3", "glm-5.3",
    "qwen-plus", "qwen-turbo", "qwen-flash",
)


def cmd_setup_weights(args) -> int:
    from . import config

    src = Path(args.from_path).expanduser()
    if not src.exists():
        print(f"源文件不存在：{src}", file=sys.stderr)
        return 1
    dst = Path(args.to) if args.to else config.WEIGHTS
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"已复制到 {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_replay(args) -> int:
    """从 trajectory.json 重新渲染报告，用于把历史运行变成文档素材。"""
    from .agent.loop import Step, Trajectory, TurnStat
    from .report import render_markdown

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    traj = Trajectory(
        question=data["question"],
        answer=data.get("answer", ""),
        model=data.get("model", ""),
        stopped_reason=data.get("stopped_reason", "stop"),
        elapsed_s=data.get("elapsed_s", 0.0),
        # 老轨迹没有 turns 字段：按 0 补齐，不报错
        turns=[
            TurnStat(
                turn=int(t.get("turn", i)),
                llm_ms=float(t.get("llm_ms", 0) or 0),
                prompt_tokens=int(t.get("prompt_tokens", 0) or 0),
                completion_tokens=int(t.get("completion_tokens", 0) or 0),
                retries=int(t.get("retries", 0) or 0),
                tool_ms=float(t.get("tool_ms", 0) or 0),
                n_calls=int(t.get("n_calls", 0) or 0),
            )
            for i, t in enumerate(data.get("turns", []) or [])
        ],
        steps=[
            Step(index=s["index"], tool=s["tool"], args=s.get("args", {}),
                 observation=s.get("observation", {}), ok=s.get("ok", True),
                 elapsed_ms=s.get("elapsed_ms", 0.0), turn=s.get("turn", 0),
                 retries=int(s.get("retries", 0) or 0))
            for s in data.get("steps", [])
        ],
    )
    print(render_markdown(traj))
    return 0


def _engine(args):
    from .vision.engine import YoloEngine

    return YoloEngine(
        weights=args.weights or None,
        device=getattr(args, "device", None) or None,
    )


def cmd_serve(args) -> int:
    """起 Web 服务：拖图 + 对话，答案旁标注是规则给的还是模型给的。

    依赖是可选的（fastapi/uvicorn 不在核心依赖里，因为它们只有这一条路径要用），
    所以这里显式判一次，缺了就给出确切的安装命令，而不是抛一个 ImportError 栈。
    """
    try:
        import uvicorn
    except ImportError:
        print("缺少 web 依赖。安装：pip install -e '.[web]'", file=sys.stderr)
        return 2

    from . import serve

    serve._engine_overrides.update(
        weights=args.weights or None,
        device=getattr(args, "device", None) or None,
    )
    serve._llm_overrides.update(provider=args.provider, model=args.model)
    print(f"甘蓝视觉 Agent → http://{args.host}:{args.port}", flush=True)
    print(f"模型：{args.provider or 'auto'} / {args.model or '默认'}", flush=True)
    print("首次启动要加载并预热权重（CUDA 约 20~30s），就绪后终端会打印一行 [serve]", flush=True)
    uvicorn.run(serve.app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cvagent", description="甘蓝航拍图像分析 Agent")
    p.add_argument("--weights", help="权重路径，默认 weights/best.pt")
    p.add_argument("--device", help="cpu / cuda:0 / mps，默认读 CVAGENT_DEVICE")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="自然语言提问：默认先走规则，判断不了再升级到 Agent")
    a.add_argument("question", help="问题文本")
    a.add_argument("--image", help="图片路径，会被拼进问题")
    a.add_argument("--mode", default="auto", choices=["auto", "pipeline", "agent"],
                   help="auto=规则优先未命中升级（默认）；pipeline=只用规则；agent=直接进循环")
    a.add_argument("--provider", help="openai / dashscope / deepseek / mock / auto")
    a.add_argument("--model", help="模型名，覆盖默认值")
    a.add_argument("--max-steps", type=int, default=None)
    a.add_argument("--no-save", action="store_true", help="不落盘")
    a.add_argument("--markdown", action="store_true", help="额外打印 Markdown 决策链")
    a.add_argument("--stream", action="store_true", help="流式打印模型输出（先看到字，再看到工具行）")
    a.add_argument("--session", metavar="NAME", help="会话名：加载历史结论并在成功后写回")
    a.add_argument("--new-session", action="store_true", help="配合 --session，忽略已有历史重新开始")
    a.set_defaults(func=cmd_ask)

    d = sub.add_parser("detect", help="直接推理，不经 LLM")
    d.add_argument("image")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--full", action="store_true", help="返回全部框坐标")
    d.set_defaults(func=cmd_detect)

    t = sub.add_parser("tools", help="列出工具与 schema")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_tools)

    doc = sub.add_parser("doctor", help="自检：权重、torch、工具、LLM")
    doc.add_argument("--probe", action="store_true", help="实际发一次请求验证 LLM 可用")
    doc.add_argument("--provider", help="指定要探测的 provider")
    doc.set_defaults(func=cmd_doctor)

    m = sub.add_parser("models", help="探测哪些模型可用且支持工具调用")
    m.add_argument("--provider", help="openai / dashscope / deepseek / 任意兼容网关")
    m.add_argument("--candidates", help="逗号分隔的模型名，覆盖内置候选")
    m.set_defaults(func=cmd_models)

    s = sub.add_parser("setup-weights", help="从主仓库复制权重")
    s.add_argument("--from", dest="from_path", required=True)
    s.add_argument("--to")
    s.set_defaults(func=cmd_setup_weights)

    r = sub.add_parser("replay", help="重渲染 trajectory.json")
    r.add_argument("file")
    r.set_defaults(func=cmd_replay)

    w = sub.add_parser("serve", help="起 Web 界面：拖图 + 对话，标注每步是规则还是模型")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--provider", help="openai / dashscope / deepseek / mock / auto")
    w.add_argument("--model", help="模型名。默认模型没额度时用它换一个")
    w.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
