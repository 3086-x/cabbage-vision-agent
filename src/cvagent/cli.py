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
    from .agent.loop import Agent
    from .llm.registry import build_provider
    from .report import render_markdown, save
    from .tools.registry import default_registry

    engine = _engine(args)
    registry = default_registry(engine)
    provider = build_provider(args.provider, args.model)

    question = args.question
    if args.image:
        question = f"{question}\n图片路径：{args.image}"

    print(f"模型：{provider.name}/{provider.model}")
    print(f"问题：{args.question}")
    if args.image:
        print(f"图片：{args.image}")
    print("正在加载检测模型…", flush=True)
    load_ms = engine.warmup()
    print(f"就绪（加载 {load_ms / 1000:.1f}s，后续单图推理约 0.1s）")
    print("─" * 60)

    agent = Agent(provider, registry, max_steps=args.max_steps, verbose=True)
    try:
        traj = agent.run(question)
    except Exception as e:  # noqa: BLE001
        print(f"\n运行失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print("─" * 60)
    print(traj.answer)
    print("─" * 60)
    print(f"工具调用 {traj.n_tool_calls} 次（失败 {traj.n_failed_calls}），{traj.elapsed_s:.1f}s，终止：{traj.stopped_reason}")

    if not args.no_save:
        out = save(traj)
        print(f"已保存：{out}")
        if args.markdown:
            print()
            print(render_markdown(traj))
    else:
        print()
        print(render_markdown(traj))

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
    from .agent.loop import Step, Trajectory
    from .report import render_markdown

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    traj = Trajectory(
        question=data["question"],
        answer=data.get("answer", ""),
        model=data.get("model", ""),
        stopped_reason=data.get("stopped_reason", "stop"),
        elapsed_s=data.get("elapsed_s", 0.0),
        steps=[
            Step(index=s["index"], tool=s["tool"], args=s.get("args", {}),
                 observation=s.get("observation", {}), ok=s.get("ok", True),
                 elapsed_ms=s.get("elapsed_ms", 0.0))
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cvagent", description="甘蓝航拍图像分析 Agent")
    p.add_argument("--weights", help="权重路径，默认 weights/best.pt")
    p.add_argument("--device", help="cpu / cuda:0 / mps，默认读 CVAGENT_DEVICE")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="自然语言提问，走完整 Agent 循环")
    a.add_argument("question", help="问题文本")
    a.add_argument("--image", help="图片路径，会被拼进问题")
    a.add_argument("--provider", help="openai / dashscope / deepseek / mock / auto")
    a.add_argument("--model", help="模型名，覆盖默认值")
    a.add_argument("--max-steps", type=int, default=None)
    a.add_argument("--no-save", action="store_true", help="不落盘")
    a.add_argument("--markdown", action="store_true", help="额外打印 Markdown 决策链")
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

    s = sub.add_parser("setup-weights", help="从主仓库复制权重")
    s.add_argument("--from", dest="from_path", required=True)
    s.add_argument("--to")
    s.set_defaults(func=cmd_setup_weights)

    r = sub.add_parser("replay", help="重渲染 trajectory.json")
    r.add_argument("file")
    r.set_defaults(func=cmd_replay)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
