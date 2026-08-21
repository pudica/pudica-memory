#!/usr/bin/env python3
"""清理 ChromaDB 中的空向量孤儿记录。"""
import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 1. 查 SQLite 中有多少条记录
db_path = os.path.join(os.path.dirname(__file__), "data", "unified_memory.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.execute("SELECT COUNT(*) AS count FROM memories")
row = cursor.fetchone()
print(f"SQLite memories 表: {row['count']} 条")

# 2. 查有多少条记录 content 为空
cursor = conn.execute("SELECT COUNT(*) AS count FROM memories WHERE content IS NULL OR content = ''")
row = cursor.fetchone()
print(f"其中 content 为空: {row['count']} 条")

# 3. 列出 content 为空的记录（用于调试）
cursor = conn.execute("SELECT id, source, created_at FROM memories WHERE content IS NULL OR content = ''")
empty = cursor.fetchall()
if empty:
    print(f"\n空记录列表:")
    for r in empty:
        print(f"  id={r['id']}, source={r['source']}, created_at={r['created_at']}")

# 4. 尝试通过 ChromaDB 查询空向量
try:
    import chromadb
    chroma_path = os.path.join(os.path.dirname(__file__), "data", "chroma")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("unified_memory")
    count = collection.count()
    print(f"\nChromaDB 集合 unified_memory: {count} 条向量")
    
    # 查出所有 metadatas 中 content 为空的
    all_data = collection.get(include=["metadatas", "documents"])
    empty_text = []
    for i, doc in enumerate(all_data["documents"]):
        if not doc or not doc.strip():
            empty_text.append(all_data["ids"][i])
    print(f"其中空文本向量: {len(empty_text)} 条")
    if empty_text:
        print(f"向量 IDs: {empty_text[:20]}")
        # 删除空向量
        collection.delete(ids=empty_text)
        print(f"已删除 {len(empty_text)} 条空向量")
        new_count = collection.count()
        print(f"ChromaDB 剩余: {new_count} 条向量")
except Exception as e:
    print(f"ChromaDB 操作失败: {e}")

conn.close()
print("\n完成")