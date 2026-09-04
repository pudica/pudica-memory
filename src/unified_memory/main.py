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
import shutil
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
from unified_memory.tasks.persona import PersonaDistiller
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
            if not isinstance(data, dict):
                raise ValueError(f"LLM 返回非 JSON 对象: {type(data).__name__}")
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"LLM 返回空的 choices: {data}")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not content:
                logger.warning("LLM 返回空内容: %s", data)
            return content
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

        try:
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
        except Exception:
            # Bug fix: 初始化失败时关闭 LLM 的 httpx AsyncClient，防止 HTTP 连接泄漏。
            await self.llm.close()
            raise

        # 4c. 重排器（Hindsight: Cross-Encoder）
        if self.config.reranker.enabled:
            cross_encoder_model = getattr(self.config.reranker, "cross_encoder_model", None)
            self.reranker = Reranker(
                strategy=self.config.reranker.strategy,
                top_n=self.config.reranker.top_n,
                final_k=self.config.reranker.final_k,
                llm=self.llm if self.config.reranker.strategy == "llm" else None,
                max_concurrent=self.config.reranker.max_concurrent,
                cross_encoder_model=cross_encoder_model,
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
        # 初始化实体注册表（白名单过滤）
        from unified_memory.pipeline.entity_registry import EntityRegistry
        entity_registry = EntityRegistry(self.pool)
        await entity_registry.initialize()
        extractor.set_registry(entity_registry)
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

        # Persona 蒸馏器（v3.4.0）
        persona_distiller = PersonaDistiller(
            pool=self.pool,
            chroma=self.chroma,
            llm=self.llm,
            settings={
                "min_authority": "high",
                "min_trust_score": 0.7,
            },
        )

        self.scheduler = TaskScheduler(
                    reflector=reflector,
                    consolidator=consolidator,
                    reflect_interval_hours=24,
                    consolidate_interval_hours=4,
                    compressor=self.compressor,
                    compress_interval_hours=max(1, self.config.compression.interval // 3600) if self.config.compression.enabled else 24,
                    persona_distiller=persona_distiller,
                    persona_interval_hours=24,
                    clean_expiry_interval_hours=6,
                    sqlite_store=self.pool,
                )
        await self.scheduler.start()
        logger.info("  调度器: reflect=24h, consolidate=4h" +
                     (f", compression={self.config.compression.interval}s" if self.compressor else "") +
                     ", persona=24h, clean_expiry=6h")

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
                    # PipelineEngine 没有 stop() 方法，用 _flush_buffer 刷盘
                    await self.pipeline._flush_buffer()
        if self.chroma:
            # 释放 ChromaDB 客户端资源（SQLite 句柄 + 线程池）
            # 注意：不要调用 client.reset() —— 它清空整个向量库（且 1.5.9 默认禁用）
            try:
                # Bug fix: _ensure_client() 现在包裹在 try 内，防止 ChromaDB
                # 处于异常状态时 shutdown 流程被意外中断。
                self.chroma._ensure_client()
                close = getattr(self.chroma._client, "close", None)
                if callable(close):
                    close()
            except Exception as e:
                logger.warning("ChromaDB close 失败（可能已关闭）: %s", e)
            finally:
                self.chroma._client = None
                self.chroma._collection = None
            # Bug fix: 关闭 ChromaDB 专用线程池（防止 ResourceWarning）
            from unified_memory.store.chroma_store import shutdown_chroma_executor
            shutdown_chroma_executor()
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

def create_embedded_app(config_path: Optional[str] = None, agent_id: str = "default") -> "UnifiedMemoryApp":
    """嵌入式运行模式：创建并启动 UnifiedMemoryApp 实例，免 HTTP 服务。

    v3.4.0 新增：免启动 HTTP/MCP 服务器，直接 import 后调用此函数即可使用。

    Args:
        config_path: 配置文件路径（默认为 None，使用自动发现）
        agent_id: 多 agent 隔离标识（默认为 "default"）

    Returns:
        已初始化并启动的 UnifiedMemoryApp 实例

    Usage:
        from unified_memory import create_embedded_app

        app = create_embedded_app()
        # 使用 app.ingest(...) 直接写入
        # 使用 app.search(...) 直接检索
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = Config.load(config_path)
    config.agent_id = agent_id

    app = UnifiedMemoryApp(config)
    loop.run_until_complete(app.initialize())
    # 启动后台调度器
    start_scheduler = hasattr(app, 'scheduler') and app.scheduler is not None
    if start_scheduler:
        loop.create_task(app.scheduler.run())
    logger.info("嵌入式模式已启动: agent_id=%s, data_dir=%s", agent_id, config.data_dir)
    return app


def _clean_pycache():
    """启动时自动清理 __pycache__ 目录，防止旧 .pyc 缓存导致改代码不生效。"""
    src_dir = os.path.join(os.path.dirname(__file__))
    removed = 0
    for root, dirs, files in os.walk(src_dir):
        if '__pycache__' in dirs:
            p = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(p)
                removed += 1
            except Exception as e:
                logger.warning("清理 __pycache__ 失败: %s (%s)", p, e)
    if removed > 0:
        logger.info("已清理 %d 个 __pycache__ 目录", removed)


def main():
    # 自动清理 __pycache__，防止旧 .pyc 缓存导致改代码不生效
    _clean_pycache()

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
    """启动 MCP 服务器（stdio 模式），自动重启。"""
    from unified_memory.api.mcp_server import create_mcp_server

    restart_count = 0
    while True:
        app = UnifiedMemoryApp(config)
        try:
            await app.initialize()
            mcp = create_mcp_server(app)

            logger.info("pudica-Memory MCP 服务器启动 (stdio 模式)")
            try:
                await mcp.run_stdio_async()
            except Exception as e:
                logger.error("MCP 服务器崩溃: %s", e)
        except Exception as e:
            logger.error("初始化失败: %s", e)
        finally:
            await app.shutdown()

        restart_count += 1
        wait = min(restart_count * 5, 60)
        logger.info("将在 %d 秒后重启 (累计重启次数: %d)", wait, restart_count)
        await asyncio.sleep(wait)


async def run_http(config: Config, host: str = "127.0.0.1", port: int = 8000):
    """启动 HTTP 服务器（REST API）。"""
    from unified_memory.api.http_server import HTTPServer
    import uvicorn

    app = UnifiedMemoryApp(config)
    await app.initialize()
    http_server = HTTPServer(registry=app.registry, middleware=app.middleware)
    fastapi_app = http_server.create_app()

    logger.info("HTTP 服务器启动: http://%s:%d", host, port)
    config_obj = uvicorn.Config(
        fastapi_app,
        host=host,
        port=port,
        log_level=config.log_level.lower(),
    )
    server = uvicorn.Server(config_obj)
    await server.serve()