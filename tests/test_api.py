"""tests/test_api.py — API 层单元测试。"""

import pytest

from unified_memory.api.tools import ToolRegistry


class TestToolRegistry:
    """ToolRegistry 单元测试。"""

    @pytest.mark.asyncio
    async def test_register_and_list(self):
        # 创建 mock 对象
        class MockEngine:
            async def ingest(self, content, metadata=None):
                return "mock_id"
            def get_status(self):
                return {"running": True, "buffer_size": 0}

        class MockTempEngine:
            async def search(self, query, top_k=20, time_range=None):
                from unified_memory.search.types import FusionResult
                return [FusionResult(id="mock", text="result", score=0.9)]

        class MockKG:
            async def get_entity_context(self, entity):
                from unified_memory.store.kg import Entity
                return {"entity": Entity(id="1", name="test", entity_type="person"), "relations": [], "neighbors": []}
            async def get_all_entities(self):
                return ["test", "test2"]
            def get_stats(self):
                return {"entities": 2, "relations": 0, "entity_types": {}}

        class MockScheduler:
            async def trigger_reflect(self):
                return {"insights": [], "gaps": []}
            async def trigger_consolidate(self):
                return {"dedup": {}, "merge": {}, "link": {}, "upgrade": {}}
            def get_status(self):
                return {"running": True, "reflect_count": 0}

        class MockChroma:
            def search(self, query, n_results=10, wing=None, room=None):
                return []

        class MockPool:
            async def acquire(self):
                class MockConn:
                    async def execute(self, sql, *args):
                        class MockCursor:
                            async def fetchall(self):
                                return []
                            async def fetchone(self):
                                return None
                        return MockCursor()
                    async def commit(self):
                        pass
                return MockConn()
            async def release(self, conn):
                pass

        registry = ToolRegistry(MockEngine(), MockTempEngine(), MockKG(), MockScheduler(), MockChroma(), MockPool())

        tools = registry.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "mempalace_search" in tool_names
        assert "mempalace_add_drawer" in tool_names
        assert "mempalace_list_wings" in tool_names
        assert "memory_search" in tool_names
        assert "memory_ingest" in tool_names
        assert "kg_query" in tool_names
        assert "pipeline_run" in tool_names
        assert "reflect_trigger" in tool_names
        assert "consolidate_trigger" in tool_names
        assert "system_health" in tool_names
        assert "system_stats" in tool_names

    def test_get_tool(self):
        registry = ToolRegistry(None, None, None, None, None, None)
        tool = registry.get_tool("nonexistent")
        assert tool is None

    @pytest.mark.asyncio
    async def test_mempalace_search(self):
        class MockChroma:
            def search(self, query, n_results=10, wing=None, room=None):
                return [{"id": "1", "content": "test", "score": 0.9, "metadata": {}}]

        registry = ToolRegistry(None, None, None, None, MockChroma(), None)
        result = await registry._mempalace_search("test", 5)
        assert result["total"] == 1
        assert result["results"][0]["id"] == "1"