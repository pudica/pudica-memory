"""清理验证测试数据：测试记忆 + 场景 + 实体 + 任务日志。

删除规则：
- memories: content 含 "云南天海" 或 "深度检查" 的测试记忆（FTS 触发器自动同步删除）
- scenes / entities / relations: 全部是验证产生的测试数据
- reflect_logs / consolidation_logs: 验证触发产生的日志
- chroma: 同步删除对应测试记忆的向量
"""
import os
import sys

PROJECT = r"C:\Users\pudica\pudica-memory-v2.4"
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.environ["PYTHONUTF8"] = "1"
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(PROJECT, "data")

import sqlite3

db_path = os.path.join(PROJECT, "data", "unified_memory.db")
chroma_dir = os.path.join(PROJECT, "data", "chroma")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 1. 找出测试记忆 id
rows = cur.execute(
    "SELECT id, content FROM memories WHERE content LIKE '%云南天海%' OR content LIKE '%深度检查%'"
).fetchall()
test_ids = [r[0] for r in rows]
print(f"测试记忆 {len(test_ids)} 条:")
for r in rows:
    print(f"  {r[0][:8]} {r[1][:30]}")

# 2. 删除测试记忆（FTS 触发器 memories_ad 自动同步删除 FTS 行）
for tid in test_ids:
    cur.execute("DELETE FROM memories WHERE id = ?", (tid,))

# 3. 删除场景/实体/关系（全部为验证测试数据）
scenes = cur.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
entities = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
relations = cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
cur.execute("DELETE FROM scenes")
cur.execute("DELETE FROM entities")
cur.execute("DELETE FROM relations")
cur.execute("DELETE FROM reflect_logs")
cur.execute("DELETE FROM consolidation_logs")
conn.commit()

# 4. 同步删除 chroma 中的测试记忆向量
try:
    sys.path.insert(0, PROJECT)
    from unified_memory.store.chroma_store import ChromaStore
    from unified_memory.store.onnx_zh_embedder import get_bge_onnx_embedding_function

    ef = get_bge_onnx_embedding_function()
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    coll = client.get_collection("memories", embedding_function=ef)
    if test_ids:
        existing = [i for i in test_ids if i in set(coll.get(ids=test_ids)["ids"])]
        if existing:
            coll.delete(ids=existing)
            print(f"chroma 删除 {len(existing)} 条测试向量")
    print(f"chroma 剩余: {coll.count()}")
    close = getattr(client, "close", None)
    if callable(close):
        close()
except Exception as e:
    print(f"chroma 清理跳过: {e}")

conn.close()

# 5. 核对
conn = sqlite3.connect(db_path)
cur = conn.cursor()
print("\n清理后:")
print(f"  memories: {cur.execute('SELECT COUNT(*) FROM memories').fetchone()[0]}")
print(f"  memories_fts: {cur.execute('SELECT COUNT(*) FROM memories_fts').fetchone()[0]}")
print(f"  scenes: {cur.execute('SELECT COUNT(*) FROM scenes').fetchone()[0]} (应为 0)")
print(f"  entities: {cur.execute('SELECT COUNT(*) FROM entities').fetchone()[0]} (应为 0)")
conn.close()
