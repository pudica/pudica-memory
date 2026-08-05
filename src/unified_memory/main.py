"""unified_memory/main.py — 统一记忆系统入口点。

提供：
- CLI 模式：`python -m unified_memory.main --mcp` 启动 MCP 服务器
- 测试模式：`python -m unified_memory.main --test` 运行集成测试
- 配置加载：支持环境变量 UNIFIED_MEMORY_CONFIG 指定 JSON 配置路径
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

from unified_memory.config import Config, LLMConfig
from unified_memory.store.chroma_store import ChromaStore
from unified_memory.store.sqlite_store import SQLitePool
from unified_memory.store.kg import KnowledgeGraph
from unified_memory.store.mental_models import MentalModelStore
from unified_memory.pipeline.engine import PipelineEngine
from unified_memory.pipeline.l0_dedup import L0Dedup
from unified_memory.pipeline.l1_extractor import L1Extractor
from unified_memory.pipeline.l2_scene import L2SceneOrganizer
from unified_memory.pipeline.l3_search import L3Search
from unified_memory.search.dense import SemanticRetriever
from unified_memory.search.sparse import BM25Retriever
from unified_memory.search.graph import GraphRetriever
from unified_memory.search.temporal import TemporalRetriever
from unified_memory.search.fusion import RRFusion, TEMPREngine
from unified_memory.search.reranker import Reranker
from unified_memory.tasks.reflect import Reflector
from unified_memory.tasks.consolidation import Consolidator
from unified_memory.tasks.compression import Compressor
from unified_memory.tasks.scheduler import TaskScheduler
from unified_memory.middleware.auto_memory import AutoMemoryMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 真实 LLM 客户端（OpenAI 兼容 API）
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI 兼容的 LLM HTTP 客户端。

    支持 chat completions 和 JSON mode（response_format）。
    通过 config.llm 初始化：api_base, api_key, model, max_tokens, temperature, timeout。
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._endpoint = urljoin(config.api_base.rstrip("/") + "/", "chat/completions")

    async def call(self, prompt: str, **kwargs) -> str:
        """调用 LLM 获取文本响应。

        Args:
            prompt: 提示文本
            **kwargs: 可选参数，如 response_format={"type": "json_object"}

        Returns:
            LLM 返回的文本内容
        """
        messages = [{"role": "user", "content": prompt}]
        body = {
            "model": kwargs.get("model", self._config.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
            "temperature": kwargs.get("temperature", self._config.temperature),
            "stream": False,
        }
        # 如果传了 response_format，加入请求
        if "response_format" in kwargs:
            body["response_format"] = kwargs["response_format"]

        try:
            resp = await self._client.post(self._endpoint, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            raise

    async def close(self):
        await self._client.aclose()


# ---------------------------------------------------------------------------
# 应用组装
# ---------------------------------------------------------------------------

class UnifiedMemoryApp:
    """统一的记忆系统应用容器。

    将所有组件组装在一起，提供统一的初始化、启动、关闭接口。
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm: Optional[LLMClient] = None
        self.pool: Optional[SQLitePool] = None
        self.chroma: Optional[ChromaStore] = None
        self.kg: Optional[KnowledgeGraph] = None
        self.mental_models: Optional[MentalModelStore] = None
        self.pipeline: Optional[PipelineEngine] = None
        self.scheduler: Optional[TaskScheduler] = None
        self.compressor: Optional[Compressor] = None
        self.reranker: Optional[Reranker] = None
        self.middleware: Optional[AutoMemoryMiddleware] = None
        self.registry: Any = None  # ToolRegistry，由 initialize() 创建
        self._mcp_server: Any = None
        self._initialized = False

    async def initialize(self):
        """初始化所有组件，按依赖顺序。"""
        t0 = time.time()
        logger.info("初始化 pudica-Memory...")

        # 1. LLM 客户端
        self.llm = LLMClient(self.config.llm)
        logger.info("  LLM 客户端: %s (%s)", self.config.llm.model, self.config.llm.api_base)

        # 2. SQLite 存储
        self.pool = SQLitePool(
            db_path=self.config.sqlite.db_path,
            maxsize=self.config.sqlite.pool_size,
            timeout=self.config.sqlite.timeout,
        )
        await self.pool.initialize()
        logger.info("  SQLite 池: %s (pool=%d)", self.config.sqlite.db_path, self.config.sqlite.pool_size)

        # 3. ChromaDB 向量存储
        self.chroma = ChromaStore(
            persist_dir=self.config.chroma.persist_dir,
            collection_name=self.config.chroma.collection_name,
        )
        self.chroma._ensure_collection()
        logger.info("  ChromaDB: %s (collection=%s)", self.config.chroma.persist_dir, self.config.chroma.collection_name)

        # 4. 知识图谱
        self.kg = KnowledgeGraph(pool=self.pool)
        await self.kg.load_from_db()
        logger.info("  知识图谱: %d 实体, %d 关系 (LRU cache=%d)", len(self.kg._all_entity_names), len(self.kg._relations), len(self.kg._entities))

        # 4b. 心智模型（Hindsight: Mental Models）
        if self.config.mental_model.enabled:
            self.mental_models = MentalModelStore(
                pool=self.pool,
                settings={
                    "belief_update_threshold": self.config.mental_model.belief_update_threshold,
                    "max_beliefs": self.config.mental_model.max_beliefs,
                    "belief_ttl": self.config.mental_model.belief_ttl,
                },
            )
            await self.mental_models.load_from_db()
            logger.info("  心智模型: %d 条信念", self.mental_models.get_stats()["total"])

        # 4c. 重排器（Hindsight: Cross-Encoder）
        if self.config.reranker.enabled:
            self.reranker = Reranker(
                strategy=self.config.reranker.strategy,
                top_n=self.config.reranker.top_n,
                final_k=self.config.reranker.final_k,
                llm=self.llm if self.config.reranker.strategy == "llm" else None,
                max_concurrent=self.config.reranker.max_concurrent,
            )
            logger.info("  重排器: strategy=%s, top_n=%d, final_k=%d",
                         self.config.reranker.strategy, self.config.reranker.top_n,
                         self.config.reranker.final_k)

        # 5. Pipeline 管线
        dedup = L0Dedup(maxsize=self.config.pipeline.l0_maxsize)
        # 从 SQLite 恢复去重缓存，防止重启后重复写入
        await dedup.load_from_db(self.pool)
        # L1 默认走本地规则提取，LLM 作为增强（LLM 可用时自动启用增强模式）
        extractor = L1Extractor(llm=self.llm)
        scene_organizer = L2SceneOrganizer(self.kg, self.pool)

        # 搜索组件
        semantic = SemanticRetriever(self.chroma)
        bm25 = BM25Retriever(self.pool)
        graph = GraphRetriever(self.kg)
        temporal = TemporalRetriever(self.pool)
        fusion = RRFusion()
        temp_engine = TEMPREngine(semantic, bm25, graph, temporal, fusion)
        l3 = L3Search(temp_engine, default_budget=self.config.pipeline.l3_default_budget)

        self.pipeline = PipelineEngine(
            pool=self.pool,
            chroma=self.chroma,
            dedup=dedup,
            extractor=extractor,
            scene_organizer=scene_organizer,
            search=l3,
            llm=self.llm,
            batch_size=self.config.pipeline.l1_batch_size,
            idle_timeout=self.config.pipeline.l1_idle_timeout,
            verbatim_enabled=self.config.pipeline.verbatim_enabled,
            reranker=self.reranker,
            mental_models=self.mental_models,
        )
        await self.pipeline.start()
        logger.info("  Pipeline: l0_maxsize=%d, batch_size=%d, idle_timeout=%.1fs",
                     self.config.pipeline.l0_maxsize, self.config.pipeline.l1_batch_size,
                     self.config.pipeline.l1_idle_timeout)

        # 6. 后台任务
        reflector = Reflector(
            self.llm, self.kg, self.chroma, self.pool,
            mental_models=self.mental_models,
        )
        consolidator = Consolidator(self.kg, self.chroma, self.pool)
        self.compressor = Compressor(
            self.pool, self.chroma, self.llm,
            settings={
                "min_age_seconds": self.config.compression.min_age_seconds,
                "batch_size": self.config.compression.batch_size,
                "keep_ratio": self.config.compression.keep_ratio,
            },
        ) if self.config.compression.enabled else None
        self.scheduler = TaskScheduler(
            reflector=reflector,
            consolidator=consolidator,
            reflect_interval_hours=24,
            consolidate_interval_hours=4,
            compressor=self.compressor,
            compress_interval_hours=self.config.compression.interval // 3600 if self.config.compression.enabled else 24,
        )
        await self.scheduler.start()
        logger.info("  调度器: reflect=24h, consolidate=4h" +
                     (f", compression={self.config.compression.interval}s" if self.compressor else ""))

        elapsed = time.time() - t0
        self._initialized = True
        logger.info("pudica-Memory 初始化完成 (%.2fs)", elapsed)

        # 创建 ToolRegistry（供 MCP 使用）
        from unified_memory.api.tools import ToolRegistry
        self.registry = ToolRegistry(
            engine=self.pipeline,
            temp_engine=temp_engine,
            kg=self.kg,
            scheduler=self.scheduler,
            chroma=self.chroma,
            pool=self.pool,
            mental_models=self.mental_models,
            compressor=self.compressor,
        )

        # 自动存取中间件（v3.2）
        if self.config.middleware.enabled:
            self.middleware = AutoMemoryMiddleware(
                app=self,
                auto_store=self.config.middleware.auto_store,
                auto_search=self.config.middleware.auto_search,
                auto_inject=self.config.middleware.auto_inject,
                min_message_length=self.config.middleware.min_message_length,
                search_top_k=self.config.middleware.search_top_k,
                max_context_items=self.config.middleware.max_context_items,
                max_context_chars=self.config.middleware.max_context_chars,
                store_llm_response=self.config.middleware.store_llm_response,
            )
            logger.info("  自动存取中间件: 已启用 (store=%s, search=%s, inject=%s)",
                         self.config.middleware.auto_store,
                         self.config.middleware.auto_search,
                         self.config.middleware.auto_inject)

    async def get_mcp_server(self):
        """获取 MCP 服务器实例（延迟初始化，返回 FastMCP 应用）。"""
        if not self._initialized:
            await self.initialize()
        if self._mcp_server is None:
            from unified_memory.api.mcp_server import create_mcp_server
            self._mcp_server = create_mcp_server(self)
        return self._mcp_server

    async def shutdown(self):
        """优雅关闭所有组件。"""
        logger.info("关闭 pudica-Memory...")
        if self.scheduler:
            await self.scheduler.stop()
        if self.pipeline:
            await self.pipeline.stop()  # stop() 内部已刷盘
        if self.chroma:
            self.chroma._ensure_client()
            # 释放 ChromaDB 客户端资源（SQLite 句柄 + 线程池）
            # 注意：不要调用 client.reset() —— 它清空整个向量库（且 1.5.9 默认禁用）
            try:
                close = getattr(self.chroma._client, "close", None)
                if callable(close):
                    close()
            except Exception as e:
                logger.warning("ChromaDB close 失败（可能已关闭）: %s", e)
            finally:
                self.chroma._client = None
                self.chroma._collection = None
        if self.pool:
            await self.pool.close()
        if self.llm:
            await self.llm.close()
        logger.info("pudica-Memory 已关闭")


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------

async def run_test(config: Config):
    """运行集成测试，验证所有核心功能。"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = UnifiedMemoryApp(config)
    await app.initialize()

    passed = 0
    failed = 0

    async def check(name: str, coro, expected_ok: bool = True):
        nonlocal passed, failed
        try:
            if coro is not None:
                result = await coro
                ok = result if result is not None else expected_ok
            else:
                ok = expected_ok
            if ok:
                passed += 1
                logger.info("  ✅ %s", name)
            else:
                failed += 1
                logger.warning("  ⚠️ %s: 返回空结果", name)
        except Exception as e:
            failed += 1
            logger.error("  ❌ %s: %s", name, e)

    async def _query(sql: str) -> list[dict]:
        conn = await app.pool.acquire()
        try:
            cursor = await conn.execute(sql)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await app.pool.release(conn)

    logger.info("====== pudica-Memory 集成测试 ======")

    await check("memories 表状态", _query("SELECT COUNT(*) as cnt FROM memories"))
    await check("scenes 表状态", _query("SELECT COUNT(*) as cnt FROM scenes"))

    # 写入测试数据
    test_content = "王哥的气阴两虚体质调理方案：生脉饮+玉屏风+参苓白术散+知柏地黄丸（2026年7月）"
    await app.pipeline.ingest(test_content, source="test")
    await app.pipeline.flush()  # 立即刷盘
    await asyncio.sleep(0.3)  # 等待异步写入完成

    try:
        stats = app.chroma.get_status()
        await check("ChromaDB 状态", asyncio.sleep(0), stats.get("collection") == config.chroma.collection_name)
    except Exception as e:
        logger.warning("  ⚠️ ChromaDB stats: %s", e)

    # 搜索测试
    await check("mempalace_search (语义)", app.pipeline.search("气阴两虚体质调理", top_k=3))
    await check("mempalace_get_taxonomy", _query("SELECT id, name FROM scenes LIMIT 5"))

    # 知识图谱测试
    await check("kg_query (实体)", app.kg.get_entity("气阴两虚"))
    await check("memory_stats", asyncio.sleep(0), isinstance(app.kg.get_stats(), dict))

    # 系统健康检查
    health = {
        "sqlite": True,
        "chroma": True,
        "pipeline": app.pipeline.is_running(),
        "scheduler": app.scheduler is not None,
    }
    await check("system_health", asyncio.sleep(0), health.get("pipeline", False))

    logger.info("====== 测试结果: %d ✅ / %d ❌ ======", passed, failed)
    await app.shutdown()
    return passed, failed


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="pudica-Memory — 统一记忆系统")
    parser.add_argument("--mcp", action="store_true", help="启动 MCP 服务器 (stdio 模式)")
    parser.add_argument("--http", action="store_true", help="启动 HTTP 服务器 (REST API)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP 监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 监听端口 (默认 8000)")
    parser.add_argument("--test", action="store_true", help="运行集成测试")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--auto", action="store_true", help="启用自动存取中间件（消息自动存储+记忆自动检索+上下文自动注入）")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # 日志配置
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 加载配置
    config_path = args.config or os.environ.get("UNIFIED_MEMORY_CONFIG")
    config = Config.load(config_path)
    config.log_level = args.log_level
    if args.auto:
        config.middleware.enabled = True

    if args.test:
        passed, failed = asyncio.run(run_test(config))
        sys.exit(0 if failed == 0 else 1)
    elif args.mcp:
        asyncio.run(run_mcp(config))
    elif args.http:
        asyncio.run(run_http(config, host=args.host, port=args.port))
    else:
        parser.print_help()


async def run_mcp(config: Config):
    """启动 MCP 服务器（stdio 模式）。"""
    from unified_memory.api.mcp_server import create_mcp_server

    app = UnifiedMemoryApp(config)
    await app.initialize()
    mcp = create_mcp_server(app)

    logger.info("pudica-Memory MCP 服务器启动 (stdio 模式)")
    try:
        await mcp.run_stdio_async()
    except KeyboardInterrupt:
        pass
    finally:
        await app.shutdown()


async def run_http(config: Config, host: str = "127.0.0.1", port: int = 8000):
    """启动 HTTP 服务器（REST API 模式）。

    Args:
        host: 监听地址
        port: 监听端口
    """
    from unified_memory.api.http_server import HTTPServer

    app = UnifiedMemoryApp(config)
    await app.initialize()
    http_server = HTTPServer(app.registry, middleware=app.middleware)

    logger.info("pudica-Memory HTTP 服务器启动: http://%s:%d", host, port)
    if app.middleware:
        logger.info("  自动存取中间件已启用: /api/v1/auto/*")
    try:
        await http_server.run_sse(host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        await app.shutdown()


if __name__ == "__main__":
    main()