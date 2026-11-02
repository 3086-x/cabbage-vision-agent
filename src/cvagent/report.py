"""把一次运行固化成可复查的产物。

Agent 的可信度问题在于：只给结论，别人无法判断它是查出来的还是编出来的。
所以每次运行都落盘 trajectory.json（可机读、可回归测试）与 report.md
（可放进 README 当决策链证据）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .agent.loop import Trajectory


def render_markdown(traj: Trajectory, *, title: str = "运行记录") -> str:
    lines = [
        f"# {title}",
        "",
        f"- 问题：{traj.question}",
        f"- 模型：`{traj.model}`",
        f"- 工具调用：{traj.n_tool_calls} 次（失败 {traj.n_failed_calls} 次）",
        f"- 耗时：{traj.elapsed_s:.1f}s",
        f"- 终止原因：`{traj.stopped_reason}`",
        "",
        "## 决策链",
        "",
    ]
    for s in traj.steps:
        mark = "✓" if s.ok else "✗"
        lines.append(f"**{s.index + 1}. `{s.tool}` {mark}**  ({s.elapsed_ms:.0f}ms)")
        if s.args:
            arg_txt = ", ".join(f"`{k}={v}`" for k, v in s.args.items())
            lines.append(f"   参数：{arg_txt}")
        obs = s.observation
        if s.ok:
            keep = {k: obs[k] for k in ("count", "density", "coverage", "conf_threshold", "total_count",
                                        "mean_count", "comparison", "stability", "image_id") if k in obs}
            lines.append(f"   观测：`{json.dumps(keep, ensure_ascii=False)}`")
        else:
            err = obs.get("error", {})
            lines.append(f"   错误：`{err.get('code')}` {err.get('message', '')}")
            if err.get("hint"):
                lines.append(f"   提示：{err['hint']}")
        lines.append("")

    lines += ["## 结论", "", traj.answer or "（无）", ""]
    return "\n".join(lines)


def save(traj: Trajectory, out_dir: Path | None = None, *, slug: str | None = None) -> Path:
    from .config import RUNS_DIR

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = (out_dir or RUNS_DIR) / (slug or stamp)
    target.mkdir(parents=True, exist_ok=True)

    (target / "trajectory.json").write_text(
        json.dumps(traj.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(traj), encoding="utf-8")
    return target
