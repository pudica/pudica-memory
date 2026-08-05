"""tests/test_pipeline.py — 管线层单元测试。"""

import pytest

from unified_memory.pipeline.l0_dedup import L0Dedup
from unified_memory.pipeline.l1_extractor import L1Extractor
from unified_memory.pipeline.l3_search import L3Search
from unified_memory.search.types import FusionResult


class TestL0Dedup:
    """L0 去重缓存测试。"""

    def test_is_duplicate(self):
        dedup = L0Dedup(maxsize=10)
        assert not dedup.is_duplicate("hello world")
        assert dedup.is_duplicate("hello world")

    def test_add(self):
        dedup = L0Dedup()
        dedup.add("new content", "id_123")
        assert dedup.is_duplicate("new content")
        assert "id_123" in dedup.processed_ids

    def test_clear(self):
        dedup = L0Dedup()
        dedup.is_duplicate("test")
        dedup.clear()
        assert not dedup.is_duplicate("test")  # 清空后重新添加

    def test_lru_eviction(self):
        dedup = L0Dedup(maxsize=2)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        dedup.is_duplicate("c")  # 触发淘汰 a
        # a 被淘汰，所以不是重复
        result = dedup.cache
        # 验证缓存大小为 2
        assert len(result) == 2
        # b 和 c 应该还在
        assert dedup.is_duplicate("b")
        assert dedup.is_duplicate("c")

    def test_get_fingerprint(self):
        dedup = L0Dedup()
        fp1 = dedup.get_fingerprint("hello")
        fp2 = dedup.get_fingerprint("hello")
        assert fp1 == fp2
        fp3 = dedup.get_fingerprint("world")
        assert fp1 != fp3


class TestL3Search:
    """L3 搜索 token budget 裁剪测试。"""

    def test_trim_to_budget(self):
        # 创建 L3Search 实例，使用 mock TEMPREngine
        class MockTempEngine:
            pass

        l3 = L3Search(MockTempEngine(), default_budget=50)

        results = [
            FusionResult(id="a", text="x" * 30, score=0.9),
            FusionResult(id="b", text="y" * 30, score=0.8),
            FusionResult(id="c", text="z" * 30, score=0.7),
        ]

        # budget 50，只能容纳 2 条
        trimmed = l3._trim_to_budget(results, 50)
        assert len(trimmed) == 1  # 30 > 50 只能留 1 条

        # budget 足够
        trimmed = l3._trim_to_budget(results, 200)
        assert len(trimmed) == 3

    def test_trim_empty(self):
        l3 = L3Search(None)
        trimmed = l3._trim_to_budget([], 100)
        assert trimmed == []

    def test_trim_single_overflow(self):
        l3 = L3Search(None)
        results = [FusionResult(id="a", text="x" * 200, score=0.9)]
        trimmed = l3._trim_to_budget(results, 50)
        assert len(trimmed) == 1
        assert len(trimmed[0].text) <= 53  # 50 + "..."
        assert trimmed[0].text.endswith("...")