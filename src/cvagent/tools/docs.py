"""`search_docs` 的轻量检索实现。

为什么不引入向量库 / jieba：
- 语料只有几十个 chunk，向量检索的收益接近 0，却多一个模型和一份索引文件；
- 中文分词器会把「置信度阈值」「株/亩」切成通用词，反而丢失专有搭配；
- 字符二元组 + idf 在这类短文档上足够，而且没有依赖、结果可解释。

打分：score = Σ idf(t) * (1 + log(tf))，标题命中 ×2。
所有命中都能在 `matched_terms` 里看到原因，便于调试召回偏差。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..config import DOCS_DIR, ROOT
from ..errors import InvalidArgument

_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

SNIPPET_RADIUS = 60
SNIPPET_MAX = 320
MAX_TOP_K = 5
MIN_TOP_K = 1


@dataclass
class Chunk:
    path: Path
    heading: str
    body: str
    terms: Counter

    @property
    def relative(self) -> str:
        try:
            return self.path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(self.path)

    @property
    def citation(self) -> str:
        return f"{self.relative}#{self.heading}"


def tokenize(text: str) -> list[str]:
    """ASCII 词 + 中文二元组（单字 run 保留单字）。

    只取二元组而不是「单字 + 二元组」：单字会把「量子纠缠」的「子」匹配到
    「电子」这类无关块，在几十块的语料上噪声大于召回收益。
    """
    tokens = [m.lower() for m in _ASCII_RE.findall(text)]
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            # 只取二元组：单字会把「量子纠缠」里的「子」匹配到「电子」之类，
            # 在这份几十块的语料上全是噪声。
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _split_chunks(path: Path, text: str) -> list[tuple[str, str]]:
    """按二级标题切块；一级标题只作文档名，不单独成块。"""
    lines = text.splitlines()
    title = path.stem
    chunks: list[tuple[str, str]] = []
    heading: str | None = None
    buf: list[str] = []
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip() or title
            continue
        if line.startswith("## "):
            if heading is not None:
                chunks.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
            continue
        if heading is not None:
            buf.append(line)
    if heading is not None:
        chunks.append((heading, "\n".join(buf).strip()))
    if not chunks:
        chunks.append((title, text.strip()))
    return chunks


class DocsIndex:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or DOCS_DIR).resolve()
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for heading, body in _split_chunks(path, text):
                terms = Counter(tokenize(f"{heading}\n{body}"))
                if terms:
                    self.chunks.append(Chunk(path, heading, body, terms))

    def _idf(self) -> dict[str, float]:
        n = len(self.chunks)
        df: Counter = Counter()
        for c in self.chunks:
            df.update(set(c.terms))
        return {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in df}

    def search(self, query: str, top_k: int = 3) -> dict:
        query = (query or "").strip()
        if not query:
            raise InvalidArgument("query", "", "检索词不能为空", hint="换一个关键词，如『密度分级』或『conf 阈值』。")
        top_k = max(MIN_TOP_K, min(int(top_k), MAX_TOP_K))

        q_terms = Counter(tokenize(query))
        if not q_terms:
            raise InvalidArgument("query", query, "没有可检索的字符")
        idf = self._idf()

        scored: list[tuple[float, Chunk, list[str]]] = []
        for c in self.chunks:
            score = 0.0
            matched: list[str] = []
            for t in q_terms:
                tf = c.terms.get(t, 0)
                if not tf:
                    continue
                weight = idf.get(t, 1.0) * (1.0 + math.log(tf))
                if t in tokenize(c.heading):
                    weight *= 2.0
                score += weight * q_terms[t]
                matched.append(t)
            if score > 0:
                # 同一批词的 idf 偏高会淹没有效信息，这里只保留贡献最大的 6 个
                matched.sort(key=lambda t: idf.get(t, 1.0), reverse=True)
                scored.append((score, c, matched[:6]))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "path": c.relative,
                "heading": c.heading,
                "score": round(score, 3),
                "snippet": _snippet(c.body, matched),
                "matched_terms": matched,
            }
            for score, c, matched in scored[:top_k]
        ]
        return {
            "ok": True,
            "query": query,
            "found": bool(results),
            "n_chunks_indexed": len(self.chunks),
            "results": results,
            "citations": [f"{r['path']}#{r['heading']}" for r in results],
            "hint": (
                "改用更具体的词，如『密度分级』『复核』『幻觉率』『株/亩』。"
                if not results else "引用时保留 path#heading 出处。"
            ),
        }


def _snippet(body: str, terms: list[str]) -> str:
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return ""
    pos = min((body.find(t) for t in terms if t in body), default=-1)
    if pos < 0:
        return body[:SNIPPET_MAX] + ("…" if len(body) > SNIPPET_MAX else "")
    start = max(0, pos - SNIPPET_RADIUS)
    end = min(len(body), pos + SNIPPET_RADIUS)
    out = body[start:end]
    if start > 0:
        out = "…" + out
    if end < len(body):
        out = out + "…"
    return out[:SNIPPET_MAX]
