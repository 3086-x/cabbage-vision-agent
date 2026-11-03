"""一键离线 demo：真实检测 + 脚本化决策，不需要 API Key。

它验证的是「工具契约 + 循环工程」这条链路真的能跑通，而不是模型能力：
    warmup → detect_cabbage →（低置信框 > 0 时）refine_detection → 结论

用法：
    python scripts/demo.py
    python scripts/demo.py --image path/to/img.jpg --device cpu

图片默认依次查找：--image / CVAGENT_DEMO_IMAGE / examples 下第一张图 /
同级 cabbage_project/images 下第一张图。找不到会给出明确提示。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image(arg: str | None) -> Path | None:
    import os

    if arg:
        p = Path(arg)
        return p if p.exists() else None
    env = os.getenv("CVAGENT_DEMO_IMAGE", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    for d in (ROOT / "examples", ROOT.parent / "cabbage_project" / "images"):
        if d.is_dir():
            hits = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            if hits:
                return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="甘蓝 Agent 离线 demo")
    ap.add_argument("--image", help="示例图片；默认自动查找")
    ap.add_argument("--weights", help="权重路径，默认 weights/best.pt")
    ap.add_argument("--device", help="cpu / cuda:0 / mps，默认读 CVAGENT_DEVICE")
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    image = find_image(args.image)
    if image is None:
        print("找不到示例图片。用 --image 指定一张，或设置 CVAGENT_DEMO_IMAGE。", file=sys.stderr)
        return 2

    from cvagent.agent.loop import Agent
    from cvagent.llm.mock import MockProvider
    from cvagent.report import render_markdown, save
    from cvagent.tools.registry import default_registry
    from cvagent.vision.engine import YoloEngine

    engine = YoloEngine(weights=args.weights or None, device=args.device or None)
    registry = default_registry(engine)

    print(f"图片：{image}")
    print(f"权重：{engine.weights}")
    print("正在加载检测模型…", flush=True)
    load_ms = engine.warmup()
    print(f"就绪（加载 {load_ms / 1000:.1f}s）")
    print("─" * 60)

    agent = Agent(MockProvider(), registry, max_steps=args.max_steps, verbose=True)
    traj = agent.run(f"这张图有多少株甘蓝？长势密不密？\n图片路径：{image}")

    print("─" * 60)
    print(traj.answer)
    print(render_markdown(traj, title="离线 demo 决策链"))

    out = save(traj, slug="demo")
    print(f"已保存：{out}")

    # 取最后一次带 count 的观测（refine 后的值才是最终结论）
    count = 0
    for s in reversed(traj.steps):
        got = s.observation.get("count")
        if isinstance(got, int) and got > 0:
            count = got
            break
    if not traj.answer or count <= 0:
        print("demo 失败：没有拿到有效株数，检查权重与图片。", file=sys.stderr)
        return 2
    print(f"demo 通过：count={count}，工具调用 {traj.n_tool_calls} 次，{traj.elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
