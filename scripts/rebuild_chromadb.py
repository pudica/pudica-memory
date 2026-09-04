"""重建 ChromaDB 向量库（从 SQLite 回填所有记忆向量）"""
import sys, os, json, sqlite3, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(os.path.dirname(__file__), "..", "data")

from unified_memory.store.chroma_store import ChromaStore

async def rebuild():
    storage_dir = os.environ.get("UNIFIED_MEMORY_STORAGE_DIR", "data")
    db_path = os.path.join(storage_dir, "unified_memory.db")
    chroma_path = os.path.join(storage_dir, "chroma")

    import shutil
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"已删除旧 chromadb: {chroma_path}")

    store = ChromaStore(persist_dir=chroma_path, collection_name="memories")
    store._ensure_collection()
    print(f"初始化 ChromaStore 完成")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, content, source, created_at, fact_type, metadata FROM memories ORDER BY created_at")
    rows = cur.fetchall()
    conn.close()
    print(f"SQLite 共 {len(rows)} 条记忆")

    batch_size = 100
    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        ids, texts, metadatas = [], [], []
        for r in batch:
            ids.append(r["id"])
            texts.append(r["content"] or "")
            md = {"source": r["source"] or "", "created_at": str(r["created_at"] or 0)}
            if r["fact_type"]:
                md["fact_type"] = r["fact_type"]
            if r["metadata"]:
                try:
                    extra = json.loads(r["metadata"])
                    if isinstance(extra, dict):
                        for k, v in extra.items():
                            if isinstance(v, (str, int, float, bool)):
                                md[k] = v
                except Exception:
                    pass
            metadatas.append(md)
        store._add_safe(documents=texts, ids=ids, metadatas=metadatas)
        written += len(batch)
        print(f"  已写入 {written}/{len(rows)}")

    count = store._collection.count()
    print(f"\n✅ 重建完成: ChromaDB {count} 条")
    return count

if __name__ == "__main__":
    count = asyncio.run(rebuild())
    print(f"完成，共 {count} 条向量")