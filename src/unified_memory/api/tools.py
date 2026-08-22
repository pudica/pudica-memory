"""api/tools.py — 工具注册逻辑。

MCP 工具注册表，兼容 mempalace 工具名 + 新增 unified-memory 工具。
参考文档 8-10 节。

认证分层（RouteAuth）：
    - user: 用户级写入（ingest, add_drawer），需要有效的 API key
    - system: 系统级管理（run, trigger, health），需要管理员 key
    - readonly: 只读（search, list, query），无需认证
    - public: 公共（status, stats），完全开放

用法:
    registry = ToolRegistry(engine, ...)
    registry.register("memory_search", handler, auth="readonly")
    registry.register("memory_ingest", handler, auth="user")
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AuthLevel(str, Enum):
    """认证级别（DSH 式 routeAuth 分层）。"""
    PUBLIC = "public"       # 完全开放（status, stats）
    READONLY = "readonly"   # 只读（search, list, query）
    USER = "user"           # 用户级写入（ingest, add_drawer）
    SYSTEM = "system"       # 系统级管理（run, trigger, reflect/consolidate）


@dataclass
class AuthConfig:
    """认证配置。

    DSH 式 routeAuth 模式：
    - 每个级别对应一个环境变量，配置里只存环境变量名
    - 运行时从环境变量读取实际值
    - 未配置的级别自动降级为低一级认证
    """
    user_key_env: str = "UNIFIED_MEMORY_USER_KEY"
    system_key_env: str = "UNIFIED_MEMORY_SYSTEM_KEY"

    _user_key: Optional[str] = field(default=None, repr=False)
    _system_key: Optional[str] = field(default=None, repr=False)

    def get_user_key(self) -> Optional[str]:
        if self._user_key is None:
            self._user_key = os.environ.get(self.user_key_env) or None
        return self._user_key

    def get_system_key(self) -> Optional[str]:
        if self._system_key is None:
            self._system_key = os.environ.get(self.system_key_env) or None
        return self._system_key

    def has_auth(self, level: AuthLevel) -> bool:
        if level == AuthLevel.PUBLIC or level == AuthLevel.READONLY:
            return False
        if level == AuthLevel.USER:
            return self.get_user_key() is not None
        if level == AuthLevel.SYSTEM:
            return self.get_system_key() is not None
        return False

    def verify(self, level: AuthLevel, provided_key: Optional[str]) -> bool:
        if level == AuthLevel.PUBLIC or level == AuthLevel.READONLY:
            return True
        if not provided_key:
            return False
        if level == AuthLevel.USER:
            expected = self.get_user_key()
            return expected is not None and provided_key == expected
        if level == AuthLevel.SYSTEM:
            expected = self.get_system_key()
            if expected is not None and provided_key == expected:
                return True
            return self.verify(AuthLevel.USER, provided_key)
        return False


class ToolRegistry:
    """MCP 工具注册表（带认证分层）。

    兼容 mempalace 工具名 + 新增 unified-memory 工具。
    每个工具注册时指定 auth_level，调用时自动验证。
    """

    def __init__(self, engine: Any, temp_engine: Any, kg: Any, scheduler: Any,
                 chroma: Any, pool: Any, mental_models: Any = None,
                 compressor: Any = None, auth: Optional[AuthConfig] = None):
        self._engine = engine
        self._temp_engine = temp_engine
        self._kg = kg
        self._scheduler = scheduler
        self._chroma = chroma
        self._pool = pool
        self._mental_models = mental_models
        self._compressor = compressor
        self._auth = auth or AuthConfig()
        self._tools: dict[str, dict] = {}
        self._register_all()

        self._session_injected: dict[str, set[str]] = {}
        self._SESSION_TTL: float = 1800.0
        self._session_last_active: dict[str, float] = {}
        self._threshold_override: float = 0.3

    def _register_all(self) -> None:
        """注册所有工具，每个工具指定认证级别。"""
        # ---- readonly 级别（无需认证） ----
        self._register("mempalace_search", self._mempalace_search,
                       "语义搜索记忆", {"query": "str", "top_k": "int (optional)"},
                       auth_level="readonly")
        self._register("mempalace_list_wings", self._mempalace_list_wings,
                       "列出所有 wing", {}, auth_level="readonly")
        self._register("mempalace_list_rooms", self._mempalace_list_rooms,
                       "列出 wing 下的 room", {"wing": "str (optional)"},
                       auth_level="readonly")
        self._register("mempalace_list_drawers", self._mempalace_list_drawers,
                       "列出 drawer", {"wing": "str", "room": "str", "limit": "int (optional)"},
                       auth_level="readonly")
        self._register("mempalace_get_drawer", self._mempalace_get_drawer,
                       "获取单个 drawer", {"id": "str"}, auth_level="readonly")
        self._register("mempalace_get_taxonomy", self._mempalace_get_taxonomy,
                       "获取分类", {}, auth_level="readonly")
        self._register("memory_search", self._memory_search,
                       "多策略检索", {"query": "str", "top_k": "int (optional)"},
                       auth_level="readonly")
        self._register("kg_query", self._kg_query,
                       "知识图谱查询", {"entity": "str"}, auth_level="readonly")
        self._register("mental_models_query", self._mental_models_query,
                       "查询用户心智模型/信念",
                       {"category": "str (optional)", "key": "str (optional)"},
                       auth_level="readonly")
        self._register("verbatim_recall", self._verbatim_recall,
                       "逐字回溯原始记忆", {"query": "str", "limit": "int (optional)"},
                       auth_level="readonly")
        self._register("memory_context", self._memory_context,
                       "获取完整上下文（记忆+心智模型）",
                       {"query": "str", "top_k": "int (optional)"},
                       auth_level="readonly")

        # ---- public 级别（完全开放） ----
        self._register("mempalace_status", self._mempalace_status,
                       "获取系统状态", {}, auth_level="public")
        self._register("system_health", self._system_health,
                       "系统健康检查", {}, auth_level="public")
        self._register("system_stats", self._system_stats,
                       "系统统计", {}, auth_level="public")

        # ---- user 级别（需要用户级 API key） ----
        self._register("mempalace_add_drawer", self._mempalace_add_drawer,
                       "写入内容到记忆", {"wing": "str", "room": "str", "content": "str"},
                       auth_level="user")
        self._register("memory_ingest", self._memory_ingest,
                       "管线摄取", {"content": "str", "source": "str (optional)"},
                       auth_level="user")
        self._register("entity_registry_register", self._entity_registry_register,
                       "注册实体到注册表", {"name": "str", "entity_type": "str (optional)", "status": "str (optional)"},
                       auth_level="user")
        self._register("entity_registry_confirm", self._entity_registry_confirm,
                       "确认候选实体为可信", {"name": "str"},
                       auth_level="user")
        self._register("entity_registry_reject", self._entity_registry_reject,
                       "拒绝实体（不再检测）", {"name": "str"},
                       auth_level="user")
        self._register("entity_registry_list_candidates", self._entity_registry_list_candidates,
                       "列出待确认的候选实体", {},
                       auth_level="user")
        self._register("entity_registry_list_confirmed", self._entity_registry_list_confirmed,
                       "列出已确认的实体", {},
                       auth_level="user")

        # ---- system 级别（需要管理员 API key） ----
        self._register("pipeline_run", self._pipeline_run,
                       "手动触发管线", {"content": "str", "source": "str (optional)"},
                       auth_level="system")
        self._register("reflect_trigger", self._reflect_trigger,
                       "手动触发 reflect", {}, auth_level="system")
        self._register("consolidate_trigger", self._consolidate_trigger,
                       "手动触发 consolidation", {}, auth_level="system")
        self._register("memory_compress", self._memory_compress,
                       "手动触发记忆压缩", {}, auth_level="system")

        # ---- P1: 事实冲突检测与合并 ----
        self._register("kg_detect_conflicts", self._kg_detect_conflicts,
                       "检测 KG 实体 type 冲突", {}, auth_level="readonly")
        self._register("kg_merge_conflicts", self._kg_merge_conflicts,
                       "合并 KG 实体 type 冲突（保留高频 type）",
                       {"dry_run": "bool (optional, default=true)"},
                       auth_level="system")

        # ---- P2: 场景组织增强 ----
        self._register("scene_list", self._scene_list,
                       "列出所有场景",
                       {"limit": "int (optional)", "offset": "int (optional)"},
                       auth_level="readonly")
        self._register("scene_merge", self._scene_merge,
                       "合并相似场景",
                       {"scene_id_a": "str", "scene_id_b": "str"},
                       auth_level="system")

    def _register(self, name: str, handler: Any, description: str, params: dict,
                  auth_level: str = "readonly") -> None:
        """注册单个工具，带认证级别。"""
        self._tools[name] = {
            "name": name,
            "handler": handler,
            "description": description,
            "params": params,
            "auth_level": AuthLevel(auth_level),
        }

    def get_tool(self, name: str) -> Optional[dict]:
        """获取工具定义。"""
        return self._tools.get(name)

    def verify_tool_auth(self, name: str, api_key: Optional[str] = None) -> bool:
        """验证工具调用是否通过认证。"""
        tool = self._tools.get(name)
        if not tool:
            return False
        level = tool.get("auth_level", AuthLevel.READONLY)
        return self._auth.verify(level, api_key)

    def list_tools(self) -> list[dict]:
        """列出所有工具。"""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "params": t["params"],
                "auth_level": t.get("auth_level", AuthLevel.READONLY).value,
            }
            for t in self._tools.values()
        ]

    # ---- mempalace 兼容工具实现 ----

    async def _mempalace_search(self, query: str, top_k: int = 10) -> dict:
        """mempalace 兼容的语义搜索。"""
        loop = asyncio.get_running_loop()
        from unified_memory.store.chroma_store import get_chroma_executor
        results = await loop.run_in_executor(
            get_chroma_executor(), lambda: self._chroma.search(query, n_results=top_k),
        )
        return {
            "results": [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "score": r["score"],
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ],
            "total": len(results),
        }

    async def _mempalace_add_drawer(self, wing: str, room: str, content: str) -> dict:
        """mempalace 兼容的写入。"""
        msg_id = await self._engine.ingest(content, source="mcp",
                                           metadata={"wing": wing, "room": room})
        return {"id": msg_id, "status": "success"}

    async def _mempalace_list_wings(self) -> dict:
        """列出所有 wing。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT DISTINCT COALESCE(wing, 'default') AS wing FROM memories ORDER BY wing")
            rows = await cursor.fetchall()
            wings = [row["wing"] for row in rows if row["wing"] is not None]
            return {"wings": wings or ["default"], "total": len(wings) or 1}
        finally:
            await self._pool.release(conn)

    async def _mempalace_list_rooms(self, wing: str = "default") -> dict:
        """列出 wing 下的 room。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT DISTINCT room FROM memories WHERE wing = ? ORDER BY room", (wing,))
            rows = await cursor.fetchall()
            rooms = [row["room"] for row in rows]
            return {"wing": wing, "rooms": rooms or ["general"], "total": len(rooms) or 1}
        finally:
            await self._pool.release(conn)

    async def _mempalace_list_drawers(self, wing: str, room: str, limit: int = 50) -> dict:
        """列出 drawer。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, content, wing, room, created_at, metadata FROM memories WHERE wing=? AND room=? ORDER BY created_at DESC LIMIT ?",
                (wing, room, limit))
            rows = await cursor.fetchall()
            drawers = []
            for row in rows:
                md = json.loads(row["metadata"]) if row["metadata"] else {}
                drawers.append({
                    "id": row["id"],
                    "content": row["content"][:200],
                    "wing": row["wing"],
                    "room": row["room"],
                    "created_at": row["created_at"],
                    "fact_type": md.get("fact_type", "observation"),
                })
            return {"wing": wing, "room": room, "drawers": drawers, "total": len(drawers)}
        finally:
            await self._pool.release(conn)

    async def _mempalace_get_drawer(self, id: str) -> dict:
        """获取单个 drawer。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, content, wing, room, created_at, metadata FROM memories WHERE id = ?",
                (id,))
            row = await cursor.fetchone()
            if not row:
                return {"error": "not found", "id": id}
            md = json.loads(row["metadata"]) if row["metadata"] else {}
            return {
                "id": row["id"],
                "content": row["content"],
                "wing": row["wing"],
                "room": row["room"],
                "created_at": row["created_at"],
                "fact_type": md.get("fact_type", "observation"),
            }
        finally:
            await self._pool.release(conn)

    async def _mempalace_get_taxonomy(self) -> dict:
        """获取分类。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT DISTINCT wing, room FROM memories ORDER BY wing, room")
            rows = await cursor.fetchall()
            taxonomy = {}
            for row in rows:
                wing = row["wing"] or "default"
                room = row["room"] or "general"
                if wing not in taxonomy:
                    taxonomy[wing] = []
                if room not in taxonomy[wing]:
                    taxonomy[wing].append(room)
            return {"taxonomy": taxonomy}
        finally:
            await self._pool.release(conn)

    async def _mempalace_status(self) -> dict:
        """获取系统状态。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute("SELECT COUNT(*) AS count FROM memories")
            row = await cursor.fetchone()
            total = row["count"] if row else 0
            return {
                "status": "healthy",
                "total_memories": total,
                "engine_running": self._engine.is_running(),
                "buffer_size": len(self._engine._buffer) if hasattr(self._engine, '_buffer') else 0,
            }
        finally:
            await self._pool.release(conn)

    # ---- 新增 unified-memory 工具 ----

    async def _memory_search(self, query: str, top_k: int = 20) -> dict:
        """多策略检索（稀疏+稠密+时间融合+知识图谱）。"""
        results = await self._engine.search(query, top_k=top_k)
        return {"results": results, "total": len(results)}

    async def _memory_ingest(self, content: str, source: str = "") -> dict:
        """管线摄取（L0→L1→L2→store）。"""
        msg_id = await self._engine.ingest(content, source=source)
        return {"id": msg_id, "status": "success" if msg_id else "duplicate"}

    async def _kg_query(self, entity: str) -> dict:
        """知识图谱查询。"""
        results = await self._kg.query(entity)
        return {"entity": entity, "results": results}

    async def _pipeline_run(self, content: str, source: str = "") -> dict:
        """手动触发管线。"""
        msg_id = await self._engine.ingest(content, source=source)
        return {"id": msg_id, "status": "success"}

    async def _reflect_trigger(self) -> dict:
        """手动触发 reflect。"""
        await self._scheduler.trigger_reflect()
        return {"status": "reflection_triggered"}

    async def _consolidate_trigger(self) -> dict:
        """手动触发 consolidation。"""
        await self._scheduler.trigger_consolidation()
        return {"status": "consolidation_triggered"}

    async def _system_health(self) -> dict:
        """系统健康检查。"""
        # 从 DB 读真实计数（engine.ingest_count 在 MCP 子进程可能为 0）
        db_count = 0
        try:
            conn = await self._pool.acquire()
            try:
                cursor = await conn.execute("SELECT COUNT(*) AS count FROM memories")
                row = await cursor.fetchone()
                db_count = row["count"] if row else 0
            finally:
                await self._pool.release(conn)
        except Exception:
            pass
        return {
            "status": "healthy",
            "engine_running": self._engine.is_running(),
            "total_memories": db_count,
            "buffer_size": len(self._engine._buffer) if hasattr(self._engine, '_buffer') else 0,
        }

    async def _system_stats(self) -> dict:
        """系统统计。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute("SELECT COUNT(*) AS count FROM memories")
            row = await cursor.fetchone()
            total = row["count"] if row else 0
            return {
                "total_memories": total,
                "ingest_count": self._engine.ingest_count,
                "flush_count": self._engine.flush_count,
                "l1_count": self._engine.l1_count,
                "l2_count": self._engine.l2_count,
                "search_count": self._engine.search_count,
            }
        finally:
            await self._pool.release(conn)

    # ---- v3.0 新增工具（Hindsight + MemPalace） ----

    async def _mental_models_query(self, category: str = "", key: str = "") -> dict:
        """查询用户心智模型/信念。"""
        if self._mental_models is None:
            return {"error": "mental_models not available"}
        # 组合 category 和 key 为查询字符串
        query_str = f"{category} {key}".strip()
        results = await self._mental_models.query(query_str=query_str, top_k=20)
        return {"results": results}

    async def _mental_models_strong(self, min_confidence: float = 0.8) -> dict:
        """获取高置信度信念。"""
        if self._mental_models is None:
            return {"error": "mental_models not available"}
        results = await self._mental_models.get_strong(min_confidence=min_confidence)
        return {"results": results}

    async def _memory_compress(self) -> dict:
        """手动触发记忆压缩。"""
        if self._compressor is None:
            return {"error": "compressor not available"}
        result = await self._compressor.compress()
        return {"status": "compressed", "result": result}

    async def _verbatim_recall(self, query: str, limit: int = 10) -> dict:
        """逐字回溯原始记忆。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, memory_id, raw_content, source, created_at FROM verbatim WHERE raw_content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit))
            rows = await cursor.fetchall()
            results = [
                {
                    "id": row["id"],
                    "memory_id": row["memory_id"],
                    "raw_content": row["raw_content"][:500],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            return {"results": results, "total": len(results)}
        finally:
            await self._pool.release(conn)

    async def _memory_context(self, query: str, top_k: int = 10) -> dict:
        """获取完整上下文（记忆+心智模型）。"""
        memories = await self._engine.search(query, top_k=top_k)
        mental = []
        if self._mental_models is not None:
            mental = await self._mental_models.query(query_str="", top_k=top_k)
        return {
            "memories": memories,
            "mental_models": mental,
            "total": len(memories) + len(mental),
        }

    # ---- 实体注册表管理工具 ----

    def _get_registry(self):
        """获取 EntityRegistry 实例（从 engine 的 pipeline 中获取）。"""
        # self._engine 是 UnifiedMemoryApp，pipeline 是 PipelineEngine
        import unified_memory.pipeline.entity_registry as er_mod
        pipeline = getattr(self._engine, 'pipeline', None)
        if pipeline is not None:
            stages = getattr(pipeline, '_stages', [])
            for stage in stages:
                extractor = getattr(stage, '_extractor', None)
                if extractor is not None:
                    reg = getattr(extractor, '_registry', None)
                    if reg is not None:
                        return reg
        # 兜底：直接通过 pool 构造
        pool = self._pool
        if pool is not None:
            reg = er_mod.EntityRegistry(pool)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(reg.initialize())
            except Exception:
                pass
            return reg
        return None

    async def _entity_registry_register(self, name: str, entity_type: str = "org",
                                       status: str = "candidate") -> dict:
        """注册实体到注册表。"""
        reg = self._get_registry()
        if reg is None:
            return {"error": "EntityRegistry not initialized"}
        result = await reg.register(name, entity_type, status=status)
        return {"status": "success", "result": result, "name": name}

    async def _entity_registry_confirm(self, name: str) -> dict:
        """确认候选实体为可信。"""
        reg = self._get_registry()
        if reg is None:
            return {"error": "EntityRegistry not initialized"}
        ok = await reg.confirm(name)
        return {"status": "success" if ok else "not_found", "name": name}

    async def _entity_registry_reject(self, name: str) -> dict:
        """拒绝实体（不再检测）。"""
        reg = self._get_registry()
        if reg is None:
            return {"error": "EntityRegistry not initialized"}
        ok = await reg.reject(name)
        return {"status": "success" if ok else "not_found", "name": name}

    async def _entity_registry_list_candidates(self) -> dict:
        """列出待确认的候选实体。"""
        reg = self._get_registry()
        if reg is None:
            return {"error": "EntityRegistry not initialized"}
        candidates = await reg.list_candidates()
        return {"candidates": candidates, "total": len(candidates)}

    async def _entity_registry_list_confirmed(self) -> dict:
        """列出已确认的实体。"""
        reg = self._get_registry()
        if reg is None:
            return {"error": "EntityRegistry not initialized"}
        confirmed = await reg.list_confirmed()
        return {"confirmed": confirmed, "total": len(confirmed)}

    # ---- P1: 事实冲突检测与合并 ----

    async def _kg_detect_conflicts(self) -> dict:
        """检测 KG 实体 type 冲突。"""
        if not hasattr(self._kg, 'get_entity_type_conflicts'):
            return {"error": "KG does not support get_entity_type_conflicts"}
        conflicts = await self._kg.get_entity_type_conflicts()
        return {"conflicts": conflicts, "total": len(conflicts)}

    async def _kg_merge_conflicts(self, dry_run: bool = True) -> dict:
        """合并 KG 实体 type 冲突（保留高频 type）。"""
        if not hasattr(self._kg, 'merge_entity_conflicts'):
            return {"error": "KG does not support merge_entity_conflicts"}
        result = await self._kg.merge_entity_conflicts(dry_run=dry_run)
        return {"status": "dry_run" if dry_run else "merged", "result": result}

    # ---- P2: 场景组织增强 ----

    async def _scene_list(self, limit: int = 50, offset: int = 0) -> dict:
        """列出所有场景。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, summary, entity_count, created_at, updated_at FROM scenes ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset))
            rows = await cursor.fetchall()
            scenes = [
                {
                    "id": row["id"],
                    "summary": row["summary"][:200] if row["summary"] else "",
                    "entity_count": row["entity_count"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]
            cursor2 = await conn.execute("SELECT COUNT(*) AS count FROM scenes")
            total = (await cursor2.fetchone())["count"]
            return {"scenes": scenes, "total": total, "offset": offset, "limit": limit}
        finally:
            await self._pool.release(conn)

    async def _scene_merge(self, scene_id_a: str, scene_id_b: str) -> dict:
        """合并两个场景（保留场景 A，将 B 的记忆迁移到 A 后删除 B）。"""
        conn = await self._pool.acquire()
        try:
            # 1. 检查两个场景都存在
            cursor = await conn.execute(
                "SELECT id, summary FROM scenes WHERE id IN (?, ?)", (scene_id_a, scene_id_b))
            rows = await cursor.fetchall()
            ids_found = {row["id"] for row in rows}
            if scene_id_a not in ids_found:
                return {"error": f"scene {scene_id_a} not found"}
            if scene_id_b not in ids_found:
                return {"error": f"scene {scene_id_b} not found"}

            # 2. 迁移场景 B 的记忆到场景 A
            await conn.execute(
                "UPDATE memories SET scene_id = ? WHERE scene_id = ?",
                (scene_id_a, scene_id_b))

            # 3. 删除场景 B
            await conn.execute("DELETE FROM scenes WHERE id = ?", (scene_id_b,))

            await conn.commit()
            return {"status": "merged", "kept": scene_id_a, "removed": scene_id_b}
        finally:
            await self._pool.release(conn)