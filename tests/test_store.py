"""tests/test_store.py — 存储层单元测试。"""

import asyncio
import os
import tempfile
import pytest

from unified_memory.store.sqlite_store import SQLitePool, WriteBuffer, WriteOp
from unified_memory.store.chroma_store import ChromaStore
from unified_memory.store.kg import KnowledgeGraph, Entity, Relation


class TestSQLitePool:
    """SQLitePool 单元测试。"""

    @pytest.mark.asyncio
    async def test_init_and_close(self):
        pool = SQLitePool(":memory:")
        await pool.init()
        assert pool._pool is not None
        await pool.close()

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        pool = SQLitePool(":memory:")
        await pool.init()
        conn = await pool.acquire()
        assert conn is not None
        await pool.release(conn)
        await pool.close()

    @pytest.mark.asyncio
    async def test_execute(self):
        pool = SQLitePool(":memory:")
        await pool.init()
        await pool.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        await pool.execute("INSERT INTO test (name) VALUES (?)", ("hello",))
        cursor = await pool.execute("SELECT * FROM test")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "hello"
        await pool.close()


class TestWriteBuffer:
    """WriteBuffer 单元测试。"""

    @pytest.mark.asyncio
    async def test_write_and_flush(self):
        ops = []
        flush_called = False

        async def flush_fn(batch):
            nonlocal flush_called
            flush_called = True
            ops.extend(batch)

        buffer = WriteBuffer(flush_fn, max_size=3, flush_interval=0.5)
        await buffer.write(WriteOp("INSERT", {"id": "1"}))
        await buffer.write(WriteOp("INSERT", {"id": "2"}))
        assert len(buffer._buffer) == 2
        await buffer.flush()
        assert flush_called
        assert len(ops) == 2
        await buffer.close()

    @pytest.mark.asyncio
    async def test_auto_flush_on_max(self):
        ops = []
        async def flush_fn(batch):
            ops.extend(batch)

        buffer = WriteBuffer(flush_fn, max_size=2, flush_interval=10)
        await buffer.write(WriteOp("INSERT", {"id": "1"}))
        await buffer.write(WriteOp("INSERT", {"id": "2"}))
        await asyncio.sleep(0.1)
        assert len(ops) == 2
        await buffer.close()


class TestChromaStore:
    """ChromaStore 单元测试（需要 ChromaDB 安装）。"""

    @pytest.mark.asyncio
    async def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaStore(persist_directory=tmpdir)
            store.init()
            assert store._collection is not None
            store.close()


class TestKnowledgeGraph:
    """KnowledgeGraph 单元测试。"""

    @pytest.mark.asyncio
    async def test_add_entity(self):
        kg = KnowledgeGraph()
        entity = await kg.add_entity("test_entity", "person", {"source": "test"})
        assert entity.name == "test_entity"
        assert entity.entity_type == "person"

    @pytest.mark.asyncio
    async def test_get_entity(self):
        kg = KnowledgeGraph()
        await kg.add_entity("alice", "person")
        entity = await kg.get_entity("alice")
        assert entity is not None
        assert entity.name == "alice"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self):
        kg = KnowledgeGraph()
        entity = await kg.get_entity("nonexistent")
        assert entity is None

    @pytest.mark.asyncio
    async def test_add_relation(self):
        kg = KnowledgeGraph()
        await kg.add_entity("alice", "person")
        await kg.add_entity("bob", "person")
        rel = await kg.add_relation("alice", "knows", "bob", 0.9, "test")
        assert rel.subject == "alice"
        assert rel.predicate == "knows"
        assert rel.object == "bob"

    @pytest.mark.asyncio
    async def test_get_neighbors(self):
        kg = KnowledgeGraph()
        await kg.add_entity("alice", "person")
        await kg.add_entity("bob", "person")
        await kg.add_entity("charlie", "person")
        await kg.add_relation("alice", "knows", "bob")
        await kg.add_relation("alice", "knows", "charlie")
        neighbors = await kg.get_neighbors("alice")
        assert len(neighbors) == 2
        assert ("bob", "knows") in neighbors
        assert ("charlie", "knows") in neighbors

    @pytest.mark.asyncio
    async def test_get_similar_entities(self):
        kg = KnowledgeGraph()
        await kg.add_entity("alice", "person")
        await kg.add_entity("bob", "person")
        await kg.add_entity("charlie", "person")
        await kg.add_entity("delta", "person")
        # alice 和 bob 共享 common_entity
        await kg.add_entity("common_entity", "thing")
        await kg.add_relation("alice", "related_to", "common_entity")
        await kg.add_relation("bob", "related_to", "common_entity")
        # charlie 没有公共实体
        await kg.add_relation("charlie", "related_to", "other_entity")
        similar = await kg.get_similar_entities("alice")
        similar_names = [s[0] for s in similar]
        assert "bob" in similar_names
        assert "charlie" not in similar_names

    @pytest.mark.asyncio
    async def test_get_stats(self):
        kg = KnowledgeGraph()
        await kg.add_entity("alice", "person")
        await kg.add_entity("bob", "person")
        await kg.add_entity("company_x", "org")
        await kg.add_relation("alice", "works_at", "company_x")
        stats = kg.get_stats()
        assert stats["entities"] == 3
        assert stats["relations"] == 1
        assert stats["entity_types"]["person"] == 2
        assert stats["entity_types"]["org"] == 1