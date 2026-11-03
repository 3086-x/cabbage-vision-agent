"""search_docs 的契约测试：检索命中、参数错误、无结果、出处与预算。"""

from __future__ import annotations

import pytest

from cvagent.config import DOCS_DIR
from cvagent.errors import InvalidArgument
from cvagent.tools.docs import DocsIndex
from cvagent.tools.registry import ToolRegistry
from cvagent.tools.spec import build_tools


@pytest.fixture(scope="module")
def index() -> DocsIndex:
    return DocsIndex(DOCS_DIR)


def test_chinese_query_hits_density(index):
    out = index.search("密度怎么分级")
    assert out["ok"] and out["found"]
    first = out["results"][0]
    assert first["path"].endswith("density.md")
    assert "密度" in first["heading"]


def test_empty_query_is_invalid_arg(index):
    with pytest.raises(InvalidArgument) as e:
        index.search("   ")
    assert e.value.code == "INVALID_ARG"


def test_no_match_returns_ok_found_false(index):
    out = index.search("量子纠缠退相干")
    assert out["ok"] is True
    assert out["found"] is False
    assert out["results"] == []
    assert "hint" in out


def test_citations_point_to_existing_files(index):
    out = index.search("复核稳定性", top_k=5)
    assert out["citations"]
    for cite in out["citations"]:
        path = cite.split("#", 1)[0]
        assert (DOCS_DIR.parent.parent / path).exists(), cite


def test_snippet_is_bounded(index):
    out = index.search("阈值 复核 密度 评测 局限", top_k=5)
    for r in out["results"]:
        assert len(r["snippet"]) <= 320


def test_top_k_is_capped_by_schema():
    class FakeEngine:
        pass

    registry = ToolRegistry(build_tools(FakeEngine()))
    bad = registry.dispatch("search_docs", {"query": "密度", "top_k": 99})
    assert bad["ok"] is False
    assert bad["error"]["code"] == "INVALID_ARG"

    good = registry.dispatch("search_docs", {"query": "密度", "top_k": 2})
    assert good["ok"] is True
    assert len(good["results"]) <= 2


def test_missing_query_is_invalid_arg():
    class FakeEngine:
        pass

    registry = ToolRegistry(build_tools(FakeEngine()))
    out = registry.dispatch("search_docs", "{}")
    assert out["ok"] is False
    assert out["error"]["code"] == "INVALID_ARG"
    assert "search_docs" in out["error"]["message"]
