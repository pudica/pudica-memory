"""Cron: Dump ALL memories for wiki candidate analysis."""
import sys, os, sqlite3

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
os.environ['UNIFIED_MEMORY_STORAGE_DIR'] = os.path.join(PROJECT_ROOT, 'data')
os.environ['PYTHONUTF8'] = '1'

db_dir = os.path.join(PROJECT_ROOT, 'data')
db_path = os.path.join(db_dir, 'unified_memory.db')

if not os.path.exists(db_path):
    print(f'DB_NOT_FOUND:{db_path}')
    sys.exit(0)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Dump all memories with full detail
rows = conn.execute('''
    SELECT id, content, authority, trust_score, fact_type, source, created_at, summary
    FROM memories
    ORDER BY
      CASE authority
        WHEN 'critical' THEN 0
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        ELSE 3
      END,
      trust_score DESC
''').fetchall()

print(f'TOTAL:{len(rows)}')
for r in rows:
    d = dict(r)
    content = (d.get('content') or '')[:500]
    summary = (d.get('summary') or '')[:300]
    print(f'---')
    print(f'ID:{d["id"][:16]}')
    print(f'authority:{d["authority"]}')
    print(f'trust_score:{d["trust_score"]}')
    print(f'fact_type:{d["fact_type"]}')
    print(f'source:{d["source"]}')
    print(f'created_at:{d["created_at"]}')
    print(f'content:{content}')
    if summary:
        print(f'summary:{summary}')

# Check ChromaDB collection
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    import chromadb
    from unified_memory.store.chroma_store import ChromaStore
    from unified_memory.config import Config
    config = Config.load()
    client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, 'data', 'chromadb'))
    cols = client.list_collections()
    for col in cols:
        c = client.get_collection(col.name)
        print(f'CHROMA:{col.name} count={c.count()}')
except Exception as e:
    print(f'CHROMA_ERR:{e}')

conn.close()