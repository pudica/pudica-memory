"""api/http_server.py — HTTP 接口（FastAPI 实现）。

提供 /api/v1/search, /api/v1/ingest, /api/v1/health 等端点。
参考文档 8-10 节。
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HTTPServer:
    """HTTP 服务器封装（FastAPI）。

    提供 RESTful API 接口：
    - GET /api/v1/health → 健康检查
    - POST /api/v1/search → 多策略检索
    - POST /api/v1/ingest → 管线摄取
    - GET /api/v1/kg/query → 知识图谱查询
    - POST /api/v1/reflect → 手动触发 reflect
    - POST /api/v1/consolidate → 手动触发 consolidation
    - GET /api/v1/stats → 系统统计
    - GET /api/v1/mempalace/wings → 列出 wing
    - GET /api/v1/mempalace/rooms → 列出 room
    - GET /api/v1/mempalace/drawers → 列出 drawer
    - GET /api/v1/mempalace/status → 状态
    - POST /api/v1/auto/pre_process → 自动存取预处理（存储+检索+注入）
    - POST /api/v1/auto/post_process → 自动存储 LLM 回复
    - GET /api/v1/auto/stats → 中间件统计
    """

    def __init__(self, registry: Any, middleware: Any = None):
        """
        Args:
            registry: ToolRegistry 实例
            middleware: AutoMemoryMiddleware 实例（可选）
        """
        self._registry = registry
        self._middleware = middleware

    def _get_handler(self, tool_name: str):
        """安全获取工具 handler，不存在时返回 None。"""
        tool = self._registry.get_tool(tool_name)
        if tool is None:
            logger.warning("工具未注册: %s", tool_name)
            return None
        return tool.get("handler")

    def create_app(self) -> Any:
        """创建 FastAPI 应用。

        Returns:
            FastAPI 应用实例
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            logger.error("fastapi 未安装，请运行: pip install fastapi uvicorn")
            raise

        from unified_memory import __version__
        app = FastAPI(
            title="Unified Memory API",
            version=__version__,
            description="Unified Memory 多策略检索 + 知识图谱 API",
        )

        # ---- 请求模型 ----

        class SearchRequest(BaseModel):
            query: str
            top_k: int = 20

        class IngestRequest(BaseModel):
            content: str
            source: str = ""

        class KGQueryRequest(BaseModel):
            entity: str

        class AutoPreProcessRequest(BaseModel):
            message: str
            system_prompt: Optional[str] = None
            source: str = "auto"

        class AutoPostProcessRequest(BaseModel):
            llm_response: str
            user_message: Optional[str] = None
            source: str = "llm"

        # ---- 路由 ----

        @app.get("/api/v1/health")
        async def health():
            """健康检查。"""
            handler = self._get_handler("system_health")
            if handler is None:
                return {"status": "error", "message": "tool not registered"}
            return await handler()

        @app.post("/api/v1/search")
        async def search(req: SearchRequest):
            """多策略检索。"""
            handler = self._get_handler("memory_search")
            if handler is None:
                raise HTTPException(status_code=500, detail="memory_search not registered")
            return await handler(query=req.query, top_k=req.top_k)

        @app.post("/api/v1/ingest")
        async def ingest(req: IngestRequest):
            """管线摄取。"""
            handler = self._get_handler("memory_ingest")
            if handler is None:
                raise HTTPException(status_code=500, detail="memory_ingest not registered")
            return await handler(content=req.content, source=req.source)

        @app.get("/api/v1/kg/query")
        async def kg_query(entity: str):
            """知识图谱查询。"""
            if not entity or not entity.strip():
                raise HTTPException(status_code=400, detail="entity parameter is required")
            handler = self._get_handler("kg_query")
            if handler is None:
                raise HTTPException(status_code=500, detail="kg_query not registered")
            result = await handler(entity=entity)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result

        @app.post("/api/v1/reflect")
        async def trigger_reflect():
            """手动触发 reflect。"""
            handler = self._get_handler("reflect_trigger")
            if handler is None:
                raise HTTPException(status_code=500, detail="reflect_trigger not registered")
            return await handler()

        @app.post("/api/v1/consolidate")
        async def trigger_consolidate():
            """手动触发 consolidation。"""
            handler = self._get_handler("consolidate_trigger")
            if handler is None:
                raise HTTPException(status_code=500, detail="consolidate_trigger not registered")
            return await handler()

        @app.get("/api/v1/stats")
        async def stats():
            """系统统计。"""
            handler = self._get_handler("system_stats")
            if handler is None:
                raise HTTPException(status_code=500, detail="system_stats not registered")
            return await handler()

        # ---- mempalace 兼容路由 ----

        @app.get("/api/v1/mempalace/wings")
        async def mempalace_wings():
            """列出所有 wing。"""
            handler = self._get_handler("mempalace_list_wings")
            if handler is None:
                raise HTTPException(status_code=500, detail="mempalace_list_wings not registered")
            return await handler()

        @app.get("/api/v1/mempalace/rooms")
        async def mempalace_rooms(wing: str = "default"):
            """列出 room。"""
            handler = self._get_handler("mempalace_list_rooms")
            if handler is None:
                raise HTTPException(status_code=500, detail="mempalace_list_rooms not registered")
            return await handler(wing=wing)

        @app.get("/api/v1/mempalace/drawers")
        async def mempalace_drawers(wing: str = "default", room: str = "general", limit: int = 50):
            """列出 drawer。"""
            handler = self._get_handler("mempalace_list_drawers")
            if handler is None:
                raise HTTPException(status_code=500, detail="mempalace_list_drawers not registered")
            return await handler(wing=wing, room=room, limit=limit)

        @app.get("/api/v1/mempalace/status")
        async def mempalace_status():
            """状态。"""
            handler = self._get_handler("mempalace_status")
            if handler is None:
                raise HTTPException(status_code=500, detail="mempalace_status not registered")
            return await handler()

        @app.get("/api/v1/mempalace/taxonomy")
        async def mempalace_taxonomy():
            """分类。"""
            handler = self._get_handler("mempalace_get_taxonomy")
            if handler is None:
                raise HTTPException(status_code=500, detail="mempalace_get_taxonomy not registered")
            return await handler()

        # ---- 自动存取中间件路由 ----

        @app.post("/api/v1/auto/pre_process")
        async def auto_pre_process(req: AutoPreProcessRequest):
            """自动存取预处理：自动存储消息 + 自动检索记忆 + 注入上下文。

            返回增强后的 prompt，可直接传给 LLM。
            """
            if self._middleware is None:
                raise HTTPException(status_code=503, detail="AutoMemoryMiddleware not enabled")
            enriched = await self._middleware.pre_process(
                req.message, system_prompt=req.system_prompt, source=req.source
            )
            return {"enriched_prompt": enriched, "stats": self._middleware.get_stats()}

        @app.post("/api/v1/auto/post_process")
        async def auto_post_process(req: AutoPostProcessRequest):
            """自动存储 LLM 回复。"""
            if self._middleware is None:
                raise HTTPException(status_code=503, detail="AutoMemoryMiddleware not enabled")
            await self._middleware.post_process(
                req.llm_response, user_message=req.user_message, source=req.source
            )
            return {"status": "ok", "stats": self._middleware.get_stats()}

        @app.get("/api/v1/auto/stats")
        async def auto_stats():
            """中间件统计。"""
            if self._middleware is None:
                raise HTTPException(status_code=503, detail="AutoMemoryMiddleware not enabled")
            return self._middleware.get_stats()

        self._app = app
        return app

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """以 SSE 模式运行 HTTP 服务器。

        Args:
            host: 监听地址（默认 127.0.0.1，仅本地访问）
            port: 监听端口
        """
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn 未安装，请运行: pip install uvicorn")
            raise

        app = self.create_app()
        logger.info("HTTP 服务器启动: http://%s:%d", host, port)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()