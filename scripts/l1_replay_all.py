#!/usr/bin/env python
"""
pudica-memory L1 批量回放脚本
对 SQLite 中已有的记忆执行 L1 提取（summary、authority、fact_type、KG 实体）
不走 LLM，用本地规则，适合批量回放旧数据。

用法: python l1_replay_all.py
"""
import sys, json, asyncio, os, sqlite3

sys.path.insert(0, 'src')
from unified_memory.config import Config
from unified_memory.pipeline.l1_extractor import L1Extractor
from unified_memory.store.kg import KnowledgeGraph


async def main():
    config = Config.load('config.json')
    data_dir = config.data_dir
    db_path = f'{data_dir}/unified_memory.db'

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT id, content, source, created_at FROM memories ORDER BY created_at'
    ).fetchall()
    conn.close()
    print(f'共 {len(rows)} 条记忆')

    l1 = L1Extractor(llm=None)
    kg = KnowledgeGraph({'db_path': os.path.join(data_dir, 'kg.db')})

    updated = 0
    for i, r in enumerate(rows):
        rid = r['id']
        content = r['content']
        if not content:
            continue
        result = await l1.extract([content])
        entities = result.get('entities', [])
        summary = (result.get('summary') or '')[:200]
        fact_type = result.get('fact_type', 'observation')
        authority = result.get('authority', 'medium')
        trust_score = result.get('trust_score', 0.5)

        conn2 = sqlite3.connect(db_path)
        conn2.execute(
            'UPDATE memories SET summary=?, fact_type=?, authority=?, trust_score=? WHERE id=?',
            (summary, fact_type, authority, trust_score, rid),
        )
        conn2.commit()
        conn2.close()

        for e in entities:
            kg.upsert_entity(e['name'], e.get('type', 'knowledge'))
        updated += 1

        if (i + 1) % 20 == 0:
            print(f'  处理 {i+1}/{len(rows)}')

    print(f'完成！共更新 {updated} 条记忆')


if __name__ == '__main__':
    asyncio.run(main())