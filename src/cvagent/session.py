"""跨进程的轻量会话记忆。

只做一件事：把上一轮的问题与结论留存下来，供下一次提问引用。
刻意不存完整 messages：那会把 token 预算迅速吃光，而且工具调用 id 在新请求里
没有意义。存「问题 + 结论 + 关键数字」三样，就够回答「刚才那张图多少株」。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SESSIONS_DIR
from .errors import InvalidArgument

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_TURNS = 20
CONTEXT_TURNS = 3
CONTEXT_CHARS = 1200
FIELD_CHARS = 200


@dataclass
class Turn:
    q: str
    a: str
    image: str = ""
    count: int | None = None
    model: str = ""
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return {"q": self.q, "a": self.a, "image": self.image,
                "count": self.count, "model": self.model, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            q=str(d.get("q", "")),
            a=str(d.get("a", "")),
            image=str(d.get("image", "")),
            count=d.get("count") if isinstance(d.get("count"), int) else None,
            model=str(d.get("model", "")),
            ts=str(d.get("ts", "")),
        )


@dataclass
class Session:
    id: str
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    turns: list[Turn] = field(default_factory=list)
    root: Path = field(default_factory=lambda: SESSIONS_DIR)

    # ---- 持久化 ----
    @property
    def path(self) -> Path:
        return self.root / f"{self.id}.json"

    @classmethod
    def load(cls, session_id: str, *, root: Path | None = None) -> "Session":
        _validate(session_id)
        s = cls(id=session_id, root=Path(root or SESSIONS_DIR))
        if not s.path.exists():
            return s
        try:
            data = json.loads(s.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 损坏的文件不该让一次提问失败：备份后当作新会话开始
            try:
                s.path.replace(s.path.with_suffix(f".bak-{int(time.time())}"))
            except OSError:
                pass
            return s
        s.created = str(data.get("created", s.created))
        s.updated = str(data.get("updated", s.updated))
        s.turns = [Turn.from_dict(t) for t in data.get("turns", []) if isinstance(t, dict)]
        return s

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.updated = time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.write_text(
            json.dumps(
                {"id": self.id, "created": self.created, "updated": self.updated,
                 "turns": [t.to_dict() for t in self.turns]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return self.path

    # ---- 记忆 ----
    def append(self, turn: Turn) -> None:
        self.turns.append(turn)
        del self.turns[:-MAX_TURNS]

    def context_for(self, question: str = "") -> str:
        """最近 3 轮的摘要，硬上限 1200 字。

        重复提问同一件事时直接跳过历史，避免把上一次的结论当成前提。
        """
        if not self.turns:
            return ""
        if question and self.turns[-1].q.strip() == question.strip():
            return ""
        rows = []
        used = len("[会话历史]\n")
        for i, t in enumerate(reversed(self.turns[-CONTEXT_TURNS:]), 1):
            q = _clip(t.q, FIELD_CHARS)
            a = _clip(t.a, FIELD_CHARS)
            bits = [f"{i}. 问：{q}"]
            meta = "，".join(x for x in (
                f"图 {Path(t.image).name}" if t.image else "",
                f"count={t.count}" if t.count is not None else "",
            ) if x)
            if meta:
                bits.append(f"   （{meta}）")
            bits.append(f"\n   答：{a}")
            block = "".join(bits) + "\n"
            if used + len(block) > CONTEXT_CHARS:
                break
            rows.append(block)
            used += len(block)
        if not rows:
            return ""
        return "[会话历史]\n" + "".join(rows) + "（以上是历史；若与本次问题冲突，以本次工具结果为准）"


def _validate(session_id: str) -> None:
    if not _ID_RE.match(session_id or ""):
        raise InvalidArgument(
            "session", session_id,
            "只允许字母、数字、下划线、连字符，长度 1~32",
            hint="换一个简单名字，如 --session demo。",
        )


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"
