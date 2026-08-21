"""L1 LLM 批量重提取 — 直连 SQLite+KG 模式。

用法:
    python scripts/l1_llm_replay.py              # 重跑全部
    python scripts/l1_llm_replay.py --limit 10   # 只跑前 10 条测试

前提：MCP 子进程已停，DB 无锁。
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import subprocess
import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("l1_llm_replay")


async def call_with_retry(llm, prompt, response_format, retries=6):
    """带指数退避的 LLM 调用，覆盖限流/超时/连接等瞬时错误。

    429/5xx/超时/连接错误均属 httpx.HTTPError，可安全重试。
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return await llm.call(prompt, response_format=response_format)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = min(3.0 * (2 ** attempt), 60.0)
                logger.warning(
                    "LLM 瞬时错误(%s)，%.1fs 后重试 %d/%d",
                    type(exc).__name__, wait, attempt + 1, retries - 1,
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from unified_memory.config import Config
from unified_memory.main import LLMClient
from unified_memory.pipeline.l1_extractor import (
    _build_llm_prompt, FACT_TYPE_KEYWORDS, L1Extractor, GARBAGE_ENTITIES,
)
from unified_memory.store.sqlite_store import SQLitePool
from unified_memory.store.kg import KnowledgeGraph
from unified_memory.store.chroma_store import ChromaStore


def stop_mcp_processes():
    """停掉所有 python 进程（MCP 子进程），清 WAL 锁。"""
    logger.info("停掉所有 Python 进程（MCP 子进程）...")
    subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
    import time
    time.sleep(2)
    # 清 WAL 锁
    data_dir = os.path.join(PROJECT_ROOT, "data")
    for f in ["unified_memory.db-shm", "unified_memory.db-wal"]:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            os.remove(p)
            logger.info("  已清理: %s", f)
    logger.info("MCP 进程已停，DB 锁已清")


async def main():
    parser = argparse.ArgumentParser(description="L1 LLM 批量重提取（直连 DB 模式）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不改数据")
    parser.add_argument("--batch", type=int, default=1, help="每批 LLM 调用数量")
    parser.add_argument("--no-kill", action="store_true", help="不自动停 MCP 进程（手工停好了再跑）")
    args = parser.parse_args()

    # 1. 停 MCP 子进程（除非 --no-kill）
    if not args.no_kill:
        stop_mcp_processes()

    # 2. 初始化
    config = Config.load()
    config.llm.timeout = 120  # 容忍 ark 慢响应，避免长内容/限流排队频繁超时重试
    llm = LLMClient(config.llm)
    logger.info("LLM: %s (%s)", config.llm.model, config.llm.api_base)

    pool = SQLitePool(config.sqlite.db_path)
    await pool.initialize()

    # 3. 获取所有记忆
    conn = await pool.acquire()
    try:
        cur = await conn.execute(
            "SELECT id, content, source, created_at, summary, "
            "fact_type, authority FROM memories ORDER BY created_at ASC"
        )
        rows = await cur.fetchall()
    finally:
        await pool.release(conn)

    if args.limit > 0:
        rows = rows[:args.limit]
    total = len(rows)
    logger.info("读取到 %d 条记忆", total)

    # 4. 初始化 KG（加载已有实体/关系到内存索引）
    kg = KnowledgeGraph(pool=pool)
    await kg.load_from_db()

    success = 0
    fail = 0
    skipped = 0

    # 5. 分批 LLM 提取
    for i in range(0, total, args.batch):
        batch = rows[i:i + args.batch]
        tasks = []
        for row in batch:
            try:
                content = json.loads(row["content"])
                if isinstance(content, list):
                    messages = [m["content"] if isinstance(m, dict) else str(m) for m in content]
                else:
                    messages = [str(content)]
            except (json.JSONDecodeError, TypeError, KeyError):
                messages = [str(row["content"])]

            prompt = _build_llm_prompt(messages)
            tasks.append(call_with_retry(llm, prompt, {"type": "json_object"}))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for j, (row, resp) in enumerate(zip(batch, responses)):
            memory_id = row["id"]
            idx = i + j + 1

            if isinstance(resp, Exception):
                logger.warning("[%d/%d] %s LLM 失败: %s", idx, total, memory_id[:8], resp)
                fail += 1
                continue

            try:
                cleaned = resp.strip()
                if cleaned.startswith("```"):
                    first_nl = cleaned.find("\n")
                    if first_nl != -1:
                        cleaned = cleaned[first_nl + 1:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                result = json.loads(cleaned)

                entities = result.get("entities", [])
                relations = result.get("relations", [])
                summary = result.get("summary", row["summary"] or "")

                fact_type = result.get("fact_type", result.get("type", ""))
                if not fact_type or fact_type not in FACT_TYPE_KEYWORDS:
                    extractor = L1Extractor()
                    fact_type = extractor._classify_fact_type(
                        (summary or "") + " " + " ".join(messages)
                    )

                if args.dry_run:
                    entity_names = [e.get("name", e.get("entity", "")) for e in entities if e.get("name", e.get("entity", ""))]
                    real_entities = [n for n in entity_names if n not in GARBAGE_ENTITIES]
                    logger.info(
                        "[%d/%d] %s → 实体: %d(%d有效), 关系: %d, 类型: %s",
                        idx, total, memory_id[:8],
                        len(entity_names), len(real_entities),
                        len(relations), fact_type,
                    )
                    if real_entities:
                        logger.info("        实体: %s", ", ".join(real_entities[:5]))
                    continue

                # 6. 更新记忆（只更新 summary 和 fact_type，不写 entities/relations 到 memories 表）
                entity_names = [e.get("name", e.get("entity", "")) for e in entities]
                relation_list = [{
                    "source": r.get("source", r.get("subject", "")),
                    "target": r.get("target", r.get("object", "")),
                    "relation": r.get("relation", r.get("predicate", "related_to")),
                } for r in relations]

                c = await pool.acquire()
                try:
                    await c.execute(
                        "UPDATE memories SET "
                        "summary=?, fact_type=?, "
                        "updated_at=strftime('%%s','now') "
                        "WHERE id=?",
                        (
                            summary[:500],
                            fact_type,
                            memory_id,
                        ),
                    )
                finally:
                    await pool.release(c)

                # 7. 更新 KG 实体（先删旧的，再添新的）
                for e in entities:
                    name = e.get("name", e.get("entity", ""))
                    etype = e.get("type", "unknown")
                    if name and name not in GARBAGE_ENTITIES:
                        try:
                            await kg.delete_entity(name)
                        except Exception:
                            pass
                        await kg.add_entity(name, etype, {
                            "extracted_by": "llm",
                            "source_memory": memory_id[:8],
                        })

                # 8. 更新 KG 关系（先删旧关系，再添新关系）
                for r in relation_list:
                    src = r["source"]
                    tgt = r["target"]
                    rel = r["relation"]
                    if src and tgt and src not in GARBAGE_ENTITIES and tgt not in GARBAGE_ENTITIES:
                        try:
                            await kg.delete_relation(src, tgt)
                        except Exception:
                            pass
                        await kg.add_relation(src, rel, tgt, source="llm_replay")

                success += 1
                if idx % 10 == 0:
                    logger.info("[%d/%d] ✅ %s (%d实体, %d关系)", idx, total, memory_id[:8], len(entity_names), len(relation_list))

            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.warning("[%d/%d] %s 解析失败: %s", idx, total, memory_id[:8], e)
                fail += 1

    # 9. 清理 KG 垃圾实体
    if not args.dry_run:
        logger.info("清理 KG 垃圾实体...")
        for name in GARBAGE_ENTITIES:
            try:
                await kg.delete_entity(name)
                logger.info("  已清理: %s", name)
            except Exception:
                pass

    # KnowledgeGraph 无需显式 shutdown（内存索引 + 每次写入已落库）
    await pool.close()
    logger.info("=" * 40)
    logger.info("完成！成功: %d, 失败: %d, 跳过: %d", success, fail, skipped)
    logger.info("=" * 40)

    if not args.dry_run and not args.no_kill:
        logger.info("MCP 进程已停，重启 Hermes Gateway 才能恢复记忆服务。")
        logger.info("请执行: hermes gateway restart")


if __name__ == "__main__":
    asyncio.run(main())