"""pudica-memory 健康检查：验证 SQLite + ChromaDB + KG 完整性"""
import sys, os, sqlite3, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(os.path.dirname(__file__), "..", "data")

storage_dir = os.environ["UNIFIED_MEMORY_STORAGE_DIR"]
db_path = os.path.join(storage_dir, "unified_memory.db")
chroma_path = os.path.join(storage_dir, "chroma")

# 1. SQLite
conn = sqlite3.connect(db_path)
total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
empty_summary = conn.execute("SELECT COUNT(*) FROM memories WHERE summary IS NULL OR summary=''").fetchone()[0]
null_authority = conn.execute("SELECT COUNT(*) FROM memories WHERE authority IS NULL").fetchone()[0]
entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
mental_models = conn.execute("SELECT COUNT(*) FROM mental_models").fetchone()[0]
verbatim = conn.execute("SELECT COUNT(*) FROM verbatim").fetchone()[0]
conn.close()

print(f"SQLite     : {total} memories, {entities} entities, {relations} relations, {scenes} scenes, {mental_models} mental_models, {verbatim} verbatim")
if empty_summary > 0: print(f"  ⚠ {empty_summary} memories missing summary")
if null_authority > 0: print(f"  ⚠ {null_authority} memories missing authority")
else: print("  ✅ All memories have authority")

# 2. ChromaDB
from unified_memory.store.chroma_store import ChromaStore
store = ChromaStore(persist_dir=chroma_path, collection_name="memories")
store._ensure_collection()
chroma_count = store._collection.count()
print(f"ChromaDB   : {chroma_count} vectors")
if chroma_count == total:
    print("  ✅ Vector count matches SQLite")
elif chroma_count < total:
    print(f"  ⚠ ChromaDB missing {total - chroma_count} vectors — run rebuild_chromadb.py")
else:
    print(f"  ⚠ ChromaDB has {chroma_count - total} orphan vectors")

# 3. Search test
results = store.search("王哥 气阴两虚", n_results=3)
print(f"Search     : {len(results)} results for '王哥 气阴两虚'")
for r in results:
    print(f"  {r['score']:.3f} | {r['content'][:60]}")

print("\n✅ 健康检查完成")