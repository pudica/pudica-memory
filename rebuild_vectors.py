"""重建 ChromaDB 向量索引（修复语义搜索返回空的问题）。

背景：之前嵌入器无法加载（sentence_transformers 未装 + ONNX 模型下载慢），
记忆写入 SQLite 成功但 chroma 向量为空。现在已接入本地 bge-small-zh ONNX 中文嵌入器，
本脚本把 SQLite 中已有记忆重新写入 chroma 生成向量。

用法：
    PYTHONUTF8=1 UNIFIED_MEMORY_STORAGE_DIR=C:\\...\\data python rebuild_vectors.py
"""
import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("UNIFIED_MEMORY_STORAGE_DIR", os.path.join(PROJECT_ROOT, "data"))

from unified_memory.config import Config
from unified_memory.store.chroma_store import ChromaStore
from unified_memory.store.sqlite_store import SQLitePool


async def main():
    config = Config()
    print(f"data_dir: {config.data_dir}")
    print(f"sqlite  : {config.sqlite.db_path}")
    print(f"chroma  : {config.chroma.persist_dir}")

    pool = SQLitePool(db_path=config.sqlite.db_path, maxsize=1, timeout=10.0)
    await pool.initialize()

    conn = await pool.acquire()
    try:
        cursor = await conn.execute("SELECT id, content, wing, room, created_at, metadata FROM memories ORDER BY created_at")
        rows = await cursor.fetchall()
    finally:
        await pool.release(conn)
    await pool.close()

    print(f"SQLite 中记忆数: {len(rows)}")

    items = []
    for r in rows:
        meta = r["metadata"]
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta) or {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
        meta = dict(meta or {})
        meta.setdefault("created_at", r["created_at"])
        items.append(
            {
                "id": r["id"],
                "content": r["content"],
                "wing": r["wing"] or "default",
                "room": r["room"] or "general",
                "metadata": meta,
            }
        )

    store = ChromaStore(
        persist_dir=config.chroma.persist_dir,
        collection_name=config.chroma.collection_name,
    )
    store._ensure_collection()
    ids = store.add_batch(items)
    print(f"已写入 chroma 向量: {len(ids)} 条")
    count = store._collection.count()
    print(f"chroma 集合当前数量: {count}")
    return count


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) > 0 else 1)
