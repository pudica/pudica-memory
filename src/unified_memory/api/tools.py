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

        # ---- L3 Persona 工具（v3.5.0 新增：画像蒸馏 MCP 暴露） ----
        self._register("get_persona", self._get_persona,
                       "获取用户画像（按主题或全部）",
                       {"topic": "str (optional)", "min_confidence": "str (optional, low/medium/high)"},
                       auth_level="readonly")
        self._register("persona_trigger", self._persona_trigger,
                       "手动触发 Persona 蒸馏",
                       {"force": "bool (optional, default=false)"},
                       auth_level="system")

        # ---- 场景导航树（v3.5.0 新增：树状场景索引） ----
        self._register("scene_tree", self._scene_tree,
                       "场景导航树（按 wing → room → 时间分组）",
                       {"wing": "str (optional)", "limit": "int (optional)"},
                       auth_level="readonly")

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

        # ---- 3-tool 聚合接口（PQL 风格） ----
        self._register("palace_query", self._palace_query,
                       "PQL 查询: FIND <query> [SEARCH|TAXONOMY|KG|DIARY|STATUS|SCENE] [top_k=N]",
                       {"query": "str — PQL DSL 或 JSON"},
                       auth_level="readonly")
        self._register("palace_exec", self._palace_exec,
                       "PQL 写入: ADD <content> [TO wing/room] | [MINE|UPDATE|DELETE]",
                       {"command": "str — PQL 命令 DSL 或 JSON"},
                       auth_level="user")
        self._register("palace_coordinate", self._palace_coordinate,
                       "PQL 管理: REFLECT | CONSOLIDATE | COMPRESS | STATUS | CHECKPOINT",
                       {"command": "str — PQL 协调 DSL"},
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
        return await self._temp_engine.search(query, top_k)

    async def _mempalace_list_wings(self) -> dict:
        """列出所有 wing。"""
        return {"wings": self._temp_engine.list_wings()}

    async def _mempalace_list_rooms(self, wing: Optional[str] = None) -> dict:
        """列出 wing 下的 room。"""
        return {"rooms": self._temp_engine.list_rooms(wing)}

    async def _mempalace_list_drawers(self, wing: Optional[str] = None,
                                      room: Optional[str] = None,
                                      limit: int = 50) -> dict:
        """列出 drawer。"""
        return {"drawers": self._temp_engine.list_drawers(wing, room, limit)}

    async def _mempalace_get_drawer(self, id: str) -> dict:
        """获取单个 drawer。"""
        return {"drawer": self._temp_engine.get_drawer(id)}

    async def _mempalace_get_taxonomy(self) -> dict:
        """获取分类。"""
        return {"taxonomy": self._temp_engine.get_taxonomy()}

    async def _mempalace_status(self) -> dict:
        """获取系统状态。"""
        return {"status": self._temp_engine.status()}

    async def _mempalace_add_drawer(self, wing: str, room: str, content: str) -> dict:
        """写入内容到记忆。"""
        return await self._engine.ingest(content, source=f"mempalace:{wing}/{room}")

    # ---- unified-memory 工具实现 ----

    async def _memory_search(self, query: str, top_k: int = 10) -> dict:
        """多策略检索（语义 + BM25 + 时间 + 图链接）。

        P1 增强: 返回中附带 _search_details 字段，显示各来源结果数和贡献比例。
        """
        results = await self._temp_engine.search(query, top_k)

        # 统计各来源贡献（FusionResult.sources 是 list[str]）
        details = {"total": len(results), "sources": {}}
        for r in results:
            srcs = getattr(r, "sources", None) or getattr(r, "_source", None) or ["unknown"]
            if isinstance(srcs, str):
                srcs = [srcs]
            for s in srcs:
                details["sources"][s] = details["sources"].get(s, 0) + 1
        # 算比例
        if details["total"] > 0:
            details["source_pct"] = {
                k: round(v / details["total"] * 100, 1)
                for k, v in details["sources"].items()
            }

        from dataclasses import asdict
        serialized = []
        for r in results:
            try:
                entry = asdict(r)
                if isinstance(entry.get("metadata"), dict):
                    entry["metadata"] = {str(k): str(v) for k, v in entry["metadata"].items()}
                serialized.append(entry)
            except Exception:
                serialized.append({"id": str(getattr(r, "id", "")), "text": str(getattr(r, "text", ""))})
        return {"results": serialized, "_search_details": details}

    async def _memory_ingest(self, content: str, source: Optional[str] = None) -> dict:
        """管线摄取。"""
        result = await self._engine.ingest(content, source=source)
        return {"status": "ingested", "id": result}

    async def _memory_context(self, query: str, top_k: int = 5) -> dict:
            """获取完整上下文（记忆 + 心智模型 + 画像）。

            v3.5.0 增强：注入 persona 画像和场景导航，使 agent 在对话时拥有
            更完整的用户认知上下文（类似 TencentDB memory-prompt/composer.ts 的注入模式）。
            """
            memories = await self._temp_engine.search(query, top_k)
            models = []
            if self._mental_models:
                models = await self._mental_models.query_by_relevance(query, top_k)

            # Persona 注入（v3.5.0）：获取用户画像作为上下文
            persona_prompt = ""
            try:
                persona_text = await self._get_persona_inline(min_confidence="medium")
                if persona_text:
                    persona_prompt = f"\n\n[用户画像]\n{persona_text}"
            except Exception as e:
                logger.warning("Persona 注入失败: %s", e)

            # 场景导航注入（v3.5.0）：获取场景导航树作为上下文
            scene_tree_text = ""
            try:
                scene_tree_data = await self._build_scene_tree()
                if scene_tree_data:
                    scene_tree_text = f"\n\n[场景导航]\n{scene_tree_data}"
            except Exception as e:
                logger.warning("场景导航注入失败: %s", e)

            return {"memories": memories, "mental_models": models,
                    "persona": persona_prompt.strip(), "scene_tree": scene_tree_text.strip()}

    async def _memory_compress(self) -> dict:
        """手动触发记忆压缩。"""
        if self._compressor:
            result = await self._compressor.compress()
            return {"status": "compressed", "result": result}
        return {"status": "no compressor configured"}

    async def _system_health(self) -> dict:
        """系统健康检查。"""
        total = 0
        try:
            conn = await self._pool.acquire()
            try:
                cursor = await conn.execute("SELECT COUNT(*) AS cnt FROM memories")
                row = await cursor.fetchone()
                if row:
                    total = row["cnt"]
            finally:
                await self._pool.release(conn)
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
        return {
            "status": "healthy",
            "total_memories": total,
            "session_stats": {
                "ingest_count": self._engine.ingest_count if self._engine else 0,
                "flush_count": self._engine.flush_count if self._engine else 0,
                "search_count": self._engine.search_count if self._engine else 0,
                "l1_count": self._engine.l1_count if self._engine else 0,
                "l2_count": self._engine.l2_count if self._engine else 0,
            },
        }

    async def _system_stats(self) -> dict:
        """系统统计。"""
        try:
            conn = await self._pool.acquire()
            try:
                memories_cursor = await conn.execute("SELECT COUNT(*) AS cnt FROM memories")
                memories = await memories_cursor.fetchone()
                scenes_cursor = await conn.execute("SELECT COUNT(*) AS cnt FROM scenes")
                scenes = await scenes_cursor.fetchone()
                entities = 0
                if self._kg:
                    entities = len(await self._kg.get_all_entities())
                return {
                    "memories": memories["cnt"] if memories else 0,
                    "scenes": scenes["cnt"] if scenes else 0,
                    "entities": entities,
                }
            finally:
                await self._pool.release(conn)
        except Exception as e:
            return {"error": str(e)}

    async def _kg_query(self, entity: str) -> dict:
        """知识图谱查询。"""
        if not self._kg:
            return {"error": "KG not available"}
        context = await self._kg.get_entity_context(entity)
        return {"entity": entity, "context": context}

    async def _mental_models_query(self, category: Optional[str] = None,
                                    key: Optional[str] = None) -> dict:
        """查询用户心智模型/信念。"""
        if not self._mental_models:
            return {"error": "mental models not available"}
        if key:
            model = self._mental_models.get_by_key(key)
            return {"mental_model": model}
        models = self._mental_models.list_by_category(category) if category else self._mental_models.list_all()
        return {"mental_models": models}

    async def _verbatim_recall(self, query: str, limit: int = 5) -> dict:
        """逐字回溯原始记忆。"""
        try:
            conn = await self._pool.acquire()
            try:
                cursor = await conn.execute(
                    "SELECT content, source, created_at FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,))
                rows = await cursor.fetchall()
                return {"memories": [dict(row) for row in rows]}
            finally:
                await self._pool.release(conn)
        except Exception as e:
            return {"error": str(e)}

    async def _pipeline_run(self, content: str, source: Optional[str] = None) -> dict:
        """手动触发管线。"""
        result = await self._engine.ingest(content, source=source)
        return {"status": "pipeline_run", "id": result}

    async def _reflect_trigger(self) -> dict:
        """手动触发 reflect。"""
        if hasattr(self._engine, 'reflect'):
            await self._engine.reflect()
            return {"status": "reflect triggered"}
        return {"status": "reflect not available"}

    async def _consolidate_trigger(self) -> dict:
        """手动触发 consolidation。"""
        if hasattr(self._engine, 'consolidate'):
            await self._engine.consolidate()
            return {"status": "consolidation triggered"}
        return {"status": "consolidation not available"}

    async def _entity_registry_register(self, name: str, entity_type: str = "unknown",
                                         status: str = "candidate") -> dict:
        """注册实体到注册表。"""
        if not hasattr(self._engine, '_entity_registry') or not self._engine._entity_registry:
            return {"error": "entity registry not available"}
        self._engine._entity_registry.register(name, entity_type=entity_type, status=status)
        return {"status": "registered", "name": name}

    async def _entity_registry_confirm(self, name: str) -> dict:
        """确认候选实体为可信。"""
        if not hasattr(self._engine, '_entity_registry') or not self._engine._entity_registry:
            return {"error": "entity registry not available"}
        self._engine._entity_registry.confirm(name)
        return {"status": "confirmed", "name": name}

    async def _entity_registry_reject(self, name: str) -> dict:
        """拒绝实体（不再检测）。"""
        if not hasattr(self._engine, '_entity_registry') or not self._engine._entity_registry:
            return {"error": "entity registry not available"}
        self._engine._entity_registry.reject(name)
        return {"status": "rejected", "name": name}

    async def _entity_registry_list_candidates(self) -> dict:
        """列出待确认的候选实体。"""
        if not hasattr(self._engine, '_entity_registry') or not self._engine._entity_registry:
            return {"error": "entity registry not available"}
        state = self._engine._entity_registry.get_state()
        candidates = [e for e in state.get("entities", []) if e.get("state") == "candidate"]
        return {"candidates": candidates}

    async def _entity_registry_list_confirmed(self) -> dict:
        """列出已确认的实体。"""
        if not hasattr(self._engine, '_entity_registry') or not self._engine._entity_registry:
            return {"error": "entity registry not available"}
        state = self._engine._entity_registry.get_state()
        confirmed = [e for e in state.get("entities", []) if e.get("state") == "confirmed"]
        return {"confirmed": confirmed}

    async def _kg_detect_conflicts(self) -> dict:
        """检测 KG 实体 type 冲突。"""
        if not self._kg:
            return {"error": "KG not available"}
        conflicts = await self._kg.detect_type_conflicts()
        return {"conflicts": conflicts}

    async def _kg_merge_conflicts(self, dry_run: bool = True) -> dict:
        """合并 KG 实体 type 冲突。"""
        if not self._kg:
            return {"error": "KG not available"}
        result = await self._kg.merge_type_conflicts(dry_run=dry_run)
        return {"result": result}

    async def _scene_list(self, limit: int = 20, offset: int = 0) -> dict:
        """列出所有场景。"""
        try:
            conn = await self._pool.acquire()
            try:
                cursor = await conn.execute(
                    "SELECT id, name, summary, created_at, updated_at FROM scenes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset))
                rows = await cursor.fetchall()
                return {"scenes": [dict(row) for row in rows]}
            finally:
                await self._pool.release(conn)
        except Exception as e:
            return {"error": str(e)}

    async def _scene_merge(self, scene_id_a: str, scene_id_b: str) -> dict:
        """合并相似场景。"""
        try:
            conn = await self._pool.acquire()
            try:
                cursor = await conn.execute("SELECT id FROM scenes WHERE id IN (?, ?)", (scene_id_a, scene_id_b))
                rows = await cursor.fetchall()
                ids_found = {row["id"] for row in rows}
                if scene_id_a not in ids_found:
                    return {"error": f"scene {scene_id_a} not found"}
                if scene_id_b not in ids_found:
                    return {"error": f"scene {scene_id_b} not found"}

                # 迁移场景 B 的记忆到场景 A
                await conn.execute(
                    "UPDATE memories SET scene_id = ? WHERE scene_id = ?",
                    (scene_id_a, scene_id_b))

                # 删除场景 B
                await conn.execute("DELETE FROM scenes WHERE id = ?", (scene_id_b,))

                await conn.commit()
                return {"status": "merged", "kept": scene_id_a, "removed": scene_id_b}
            finally:
                await self._pool.release(conn)
        except Exception as e:
            return {"error": str(e)}

    # ---- 3-tool 聚合接口（PQL 风格） ----

    def _parse_pql(self, text: str) -> dict:
        """简易 PQL 解析器：将 DSL 文本转为标准 action 字典。

        支持格式:
            FIND <query> [SEARCH|TAXONOMY|KG|DIARY|STATUS|SCENE] [top_k=N]
            ADD <content> [TO wing/room]
            REFLECT | CONSOLIDATE | COMPRESS | STATUS
        """
        text = text.strip()
        if not text:
            return {"action": "help", "msg": "PQL 查询语言。可用: FIND, ADD, REFLECT, CONSOLIDATE, COMPRESS, STATUS"}

        # JSON 兜底
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"action": "error", "msg": "JSON 解析失败"}

        parts = text.split()
        cmd = parts[0].upper()

        if cmd == "FIND":
            action = "search"
            query_parts = []
            top_k = 10
            scope = None
            for p in parts[1:]:
                pu = p.upper()
                if pu in ("SEARCH", "KG", "TAXONOMY", "DIARY", "STATUS", "SCENE"):
                    scope = pu.lower()
                elif pu.startswith("TOP_K="):
                    try:
                        top_k = int(p.split("=")[1])
                    except ValueError:
                        pass
                else:
                    query_parts.append(p)
            return {"action": action, "query": " ".join(query_parts),
                    "scope": scope, "top_k": top_k}

        elif cmd == "ADD":
            content = " ".join(parts[1:])
            wing = "default"
            room = "general"
            if " TO " in text:
                loc = text.upper().split(" TO ", 1)[1].strip()
                if "/" in loc:
                    wing, room = loc.split("/", 1)
                else:
                    wing = loc
            return {"action": "ingest", "content": content, "wing": wing, "room": room}

        elif cmd in ("REFLECT", "CONSOLIDATE", "COMPRESS", "STATUS", "CHECKPOINT"):
            return {"action": cmd.lower()}

        return {"action": "help", "msg": f"未知命令: {cmd}. 可用: FIND, ADD, REFLECT, CONSOLIDATE, COMPRESS, STATUS"}

    async def _palace_query(self, query: str) -> dict:
        """PQL 查询接口。"""
        parsed = self._parse_pql(query)
        action = parsed.get("action")
        if action == "search":
            return await self._memory_search(
                query=parsed.get("query", ""),
                top_k=parsed.get("top_k", 10)
            )
        elif action == "help":
            return parsed
        return {"error": f"unknown action: {action}"}

    async def _palace_exec(self, command: str) -> dict:
        """PQL 写入接口。"""
        parsed = self._parse_pql(command)
        action = parsed.get("action")
        if action == "ingest":
            return await self._memory_ingest(
                content=parsed.get("content", ""),
                source=f"pql:{parsed.get('wing', 'default')}/{parsed.get('room', 'general')}"
            )
        return {"error": f"unknown action: {action}"}

    async def _palace_coordinate(self, command: str) -> dict:
        """PQL 管理接口。"""
        parsed = self._parse_pql(command)
        action = parsed.get("action")
        if action == "reflect":
            return await self._reflect_trigger()
        elif action == "consolidate":
            return await self._consolidate_trigger()
        elif action == "compress":
            return await self._memory_compress()
        elif action == "status":
            return await self._system_health()
        return {"error": f"unknown action: {action}"}

    # ------------------------------------------------------------------
    # L3 Persona 工具（v3.5.0 新增）
    # ------------------------------------------------------------------

    async def _get_persona(self, topic: Optional[str] = None,
                            min_confidence: str = "low") -> dict:
        """获取用户画像（通过 scheduler 中的 persona_distiller 查询）。"""
        if not self._scheduler or not hasattr(self._scheduler, "_persona_distiller"):
            return {"error": "persona distiller not available"}
        distiller = self._scheduler._persona_distiller
        if not distiller:
            return {"error": "persona distiller not configured"}
        try:
            personas = await distiller.get_personas(
                topic=topic, min_confidence=min_confidence
            )
            return {"personas": personas, "count": len(personas)}
        except Exception as e:
            return {"error": str(e)}

    async def _persona_trigger(self, force: bool = False) -> dict:
        """手动触发 Persona 蒸馏。"""
        if not self._scheduler or not hasattr(self._scheduler, "_persona_distiller"):
            return {"error": "persona distiller not available"}
        distiller = self._scheduler._persona_distiller
        if not distiller:
            return {"error": "persona distiller not configured"}
        try:
            result = await distiller.distill(force=force)
            return {"status": "distilled", "result": result}
        except Exception as e:
            return {"error": str(e)}

    async def _get_persona_inline(self, min_confidence: str = "medium") -> str:
        """获取格式化的 persona 文本（供 memory_context 注入使用）。"""
        if not self._scheduler or not hasattr(self._scheduler, "_persona_distiller"):
            return ""
        distiller = self._scheduler._persona_distiller
        if not distiller:
            return ""
        try:
            personas = await distiller.get_personas(min_confidence=min_confidence)
            if not personas:
                return ""
            out = []
            for p in personas[:10]:
                conf = p.get("confidence", "low")
                icon = {"high": "✅", "medium": "📌", "low": "ℹ️"}.get(conf, "ℹ️")
                summary = p.get("summary", "")
                key_facts = p.get("key_facts", [])
                line = f"{icon} [{p['topic']}] {summary}"
                if key_facts:
                    facts_text = " | ".join(kf[:40] for kf in key_facts[:3])
                    line += f"\n   -> 关键: {facts_text}"
                out.append(line)
            return "\n".join(out)
        except Exception as e:
            logger.warning("获取 inline persona 失败: %s", e)
            return ""

    # ------------------------------------------------------------------
    # 场景导航树（v3.5.0 新增）
    # ------------------------------------------------------------------

    async def _scene_tree(self, wing: Optional[str] = None, limit: int = 50) -> dict:
        """场景导航树（按 wing -> room -> 时间分组）。"""
        try:
            tree_text = await self._build_scene_tree(wing=wing, limit=limit)
            return {"scene_tree": tree_text, "tree_type": "text"}
        except Exception as e:
            return {"error": str(e)}

    async def _build_scene_tree(self, wing: Optional[str] = None,
                                 limit: int = 50) -> str:
        """构建场景导航树文本。"""
        try:
            conn = await self._pool.acquire()
            try:
                if wing:
                    cursor = await conn.execute(
                        """SELECT s.id, s.name, s.summary, s.wing, s.room,
                                  s.created_at, s.updated_at,
                                  COUNT(m.id) AS memory_count
                           FROM scenes s
                           LEFT JOIN memories m ON m.scene_id = s.id
                           WHERE s.wing = ?
                           GROUP BY s.id
                           ORDER BY s.wing, s.room, s.created_at DESC
                           LIMIT ?""",
                        (wing, limit),
                    )
                else:
                    cursor = await conn.execute(
                        """SELECT s.id, s.name, s.summary, s.wing, s.room,
                                  s.created_at, s.updated_at,
                                  COUNT(m.id) AS memory_count
                           FROM scenes s
                           LEFT JOIN memories m ON m.scene_id = s.id
                           GROUP BY s.id
                           ORDER BY s.wing, s.room, s.created_at DESC
                           LIMIT ?""",
                        (limit,),
                    )
                rows = await cursor.fetchall()
                if not rows:
                    return "（暂无场景数据）"

                tree: dict[str, dict[str, list[dict]]] = {}
                for row in rows:
                    w = row["wing"] or "default"
                    r = row["room"] or "general"
                    if w not in tree:
                        tree[w] = {}
                    if r not in tree[w]:
                        tree[w][r] = []
                    tree[w][r].append({
                        "id": row["id"],
                        "name": row["name"],
                        "summary": (row["summary"] or "")[:80],
                        "created_at": row["created_at"],
                        "memory_count": row["memory_count"],
                    })

                out = []
                for w in sorted(tree.keys()):
                    out.append(f"\n├─ 🏠 {w}")
                    for r in sorted(tree[w].keys()):
                        scenes = tree[w][r]
                        out.append(f"│  ├─ 📁 {r} ({len(scenes)} 个场景)")
                        for s in scenes[:10]:
                            mem_label = f'{s["memory_count"]}条记忆' if s["memory_count"] else "0条"
                            out.append(f"│  │  ├─ 📄 {s['name']} [{mem_label}]")
                            if s.get("summary"):
                                out.append(f"│  │  │   {s['summary'][:60]}")
                return "\n".join(out)

            finally:
                await self._pool.release(conn)
        except Exception as e:
            logger.warning("构建场景树失败: %s", e)
            return "（场景树构建失败）"
