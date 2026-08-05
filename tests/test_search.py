"""tests/test_search.py — 搜索层单元测试。"""

import pytest

from unified_memory.search.types import ScoredResult, FusionResult
from unified_memory.search.fusion import RRFusion


class TestScoredResult:
    """ScoredResult 类型测试。"""

    def test_create(self):
        sr = ScoredResult(id="test1", text="hello world", score=0.9, source="semantic", rank=1)
        assert sr.id == "test1"
        assert sr.score == 0.9
        assert sr.source == "semantic"

    def test_defaults(self):
        sr = ScoredResult(id="test1", text="hello")
        assert sr.score == 0.0
        assert sr.source == "unknown"
        assert sr.rank == 0
        assert sr.metadata == {}


class TestFusionResult:
    """FusionResult 类型测试。"""

    def test_create(self):
        fr = FusionResult(id="test1", text="hello", score=0.5, sources=["semantic", "bm25"])
        assert fr.id == "test1"
        assert "semantic" in fr.sources
        assert "bm25" in fr.sources


class TestRRFusion:
    """RRFusion 融合器测试。"""

    def test_fuse_empty(self):
        rrf = RRFusion()
        result = rrf.fuse({})
        assert result == []

    def test_fuse_single_source(self):
        rrf = RRFusion()
        results = {
            "semantic": [
                ScoredResult(id="a", text="doc a", score=0.9, rank=0),
                ScoredResult(id="b", text="doc b", score=0.8, rank=1),
            ]
        }
        fused = rrf.fuse(results)
        assert len(fused) == 2
        assert fused[0].id == "a"
        assert fused[0].score > fused[1].score

    def test_fuse_multi_source(self):
        rrf = RRFusion()
        results = {
            "semantic": [
                ScoredResult(id="a", text="doc a", score=0.9, rank=0),
                ScoredResult(id="b", text="doc b", score=0.8, rank=1),
            ],
            "bm25": [
                ScoredResult(id="b", text="doc b", score=0.9, rank=0),
                ScoredResult(id="c", text="doc c", score=0.7, rank=1),
            ],
        }
        fused = rrf.fuse(results)
        # b 出现在两个策略中，应该得分最高
        assert fused[0].id == "b"
        assert len(fused) == 3

    def test_interleave(self):
        rrf = RRFusion()
        results = {
            "semantic": [
                ScoredResult(id="a", text="doc a", rank=0),
                ScoredResult(id="b", text="doc b", rank=1),
            ],
            "bm25": [
                ScoredResult(id="c", text="doc c", rank=0),
                ScoredResult(id="d", text="doc d", rank=1),
            ],
        }
        fused = rrf.interleave(results, top_k=4)
        assert len(fused) == 4
        # interleave 顺序: a, c, b, d
        assert fused[0].id == "a"
        assert fused[1].id == "c"
        assert fused[2].id == "b"
        assert fused[3].id == "d"