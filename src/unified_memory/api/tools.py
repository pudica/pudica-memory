"""api/tools.py — 工具注册逻辑。

MCP 工具注册表，兼容 mempalace 工具名 + 新增 unified-memory 工具。
参考文档 8-10 节。
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """MCP 工具注册表。

    兼容 mempalace 工具名：
    - mempalace_search → 语义搜索
    - mempalace_add_drawer → 写入
    - mempalace_list_wings → 列出所有 wing
    - mempalace_list_rooms → 列出 room
    - mempalace_list_drawers → 列出 drawer
    - mempalace_get_drawer → 获取单个 drawer
    - mempalace_get_taxonomy → 获取分类
    - mempalace_status → 获取状态

    新增 unified-memory 工具：
    - memory_search → 多策略检索
    - memory_ingest → 管线摄取
    - kg_query → 知识图谱查询
    - pipeline_run → 手动触发管线
    - reflect_trigger → 手动触发 reflect
    - consolidate_trigger → 手动触发 consolidation
    - system_health → 系统健康检查
    - system_stats → 系统统计
    """

    def __init__(self, engine: Any, temp_engine: Any, kg: Any, scheduler: Any, chroma: Any, pool: Any,
                 mental_models: Any = None, compressor: Any = None):
        """
        Args:
            engine: PipelineEngine 实例
            temp_engine: TEMPREngine 实例
            kg: KnowledgeGraph 实例
            scheduler: TaskScheduler 实例
            chroma: ChromaStore 实例
            pool: SQLitePool 实例
            mental_models: MentalModelStore 实例（可选）
            compressor: Compressor 实例（可选）
        """
        self._engine = engine
        self._temp_engine = temp_engine
        self._kg = kg
        self._scheduler = scheduler
        self._chroma = chroma
        self._pool = pool
        self._mental_models = mental_models
        self._compressor = compressor
        self._tools: dict[str, dict] = {}
        self._register_all()

        # 注入层缓存
        self._session_injected: dict[str, set[str]] = {}  # session_id → set[memory_id]
        self._SESSION_TTL: float = 1800.0  # 30 分钟
        self._session_last_active: dict[str, float] = {}
        self._threshold_override: float = 0.3

    def _register_all(self) -> None:
        """注册所有工具。"""
        # ---- mempalace 兼容工具 ----
        self._register("mempalace_search", self._mempalace_search,
                       "语义搜索记忆", {"query": "str", "top_k": "int (optional)"})
        self._register("mempalace_add_drawer", self._mempalace_add_drawer,
                       "写入内容到记忆", {"wing": "str", "room": "str", "content": "str"})
        self._register("mempalace_list_wings", self._mempalace_list_wings,
                       "列出所有 wing", {})
        self._register("mempalace_list_rooms", self._mempalace_list_rooms,
                       "列出 wing 下的 room", {"wing": "str (optional)"})
        self._register("mempalace_list_drawers", self._mempalace_list_drawers,
                       "列出 drawer", {"wing": "str", "room": "str", "limit": "int (optional)"})
        self._register("mempalace_get_drawer", self._mempalace_get_drawer,
                       "获取单个 drawer", {"id": "str"})
        self._register("mempalace_get_taxonomy", self._mempalace_get_taxonomy,
                       "获取分类", {})
        self._register("mempalace_status", self._mempalace_status,
                       "获取系统状态", {})

        # ---- 新增 unified-memory 工具 ----
        self._register("memory_search", self._memory_search,
                       "多策略检索", {"query": "str", "top_k": "int (optional)"})
        self._register("memory_ingest", self._memory_ingest,
                       "管线摄取", {"content": "str", "source": "str (optional)"})
        self._register("kg_query", self._kg_query,
                       "知识图谱查询", {"entity": "str"})
        self._register("pipeline_run", self._pipeline_run,
                       "手动触发管线", {"content": "str", "source": "str (optional)"})
        self._register("reflect_trigger", self._reflect_trigger,
                       "手动触发 reflect", {})
        self._register("consolidate_trigger", self._consolidate_trigger,
                       "手动触发 consolidation", {})
        self._register("system_health", self._system_health,
                       "系统健康检查", {})
        self._register("system_stats", self._system_stats,
                       "系统统计", {})

        # ---- v3.0 新增工具（Hindsight + MemPalace） ----
        self._register("mental_models_query", self._mental_models_query,
                       "查询用户心智模型/信念", {"category": "str (optional)", "key": "str (optional)"})
        self._register("mental_models_strong", self._mental_models_strong,
                       "获取高置信度信念", {"min_confidence": "float (optional)"})
        self._register("memory_compress", self._memory_compress,
                       "手动触发记忆压缩", {})
        self._register("verbatim_recall", self._verbatim_recall,
                       "逐字回溯原始记忆", {"query": "str", "limit": "int (optional)"})
        self._register("memory_context", self._memory_context,
                       "获取完整上下文（记忆+心智模型）", {"query": "str", "top_k": "int (optional)"})

    def _register(self, name: str, handler: Any, description: str, params: dict) -> None:
        """注册单个工具。"""
        self._tools[name] = {
            "name": name,
            "handler": handler,
            "description": description,
            "params": params,
        }

    def get_tool(self, name: str) -> Optional[dict]:
        """获取工具定义。"""
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """列出所有工具。"""
        return [
            {"name": t["name"], "description": t["description"], "params": t["params"]}
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
        msg_id = await self._engine.ingest(content, source="mcp", metadata={"wing": wing, "room": room})
        return {"id": msg_id, "status": "success"}

    async def _mempalace_list_wings(self) -> dict:
        """列出所有 wing。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute("SELECT DISTINCT COALESCE(wing, 'default') AS wing FROM memories ORDER BY wing")
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
                "SELECT DISTINCT room FROM memories WHERE wing = ? ORDER BY room",
                (wing,),
            )
            rows = await cursor.fetchall()
            rooms = [row["room"] for row in rows]
            return {"wing": wing, "rooms": rooms or ["general"], "total": len(rooms) or 1}
        finally:
            await self._pool.release(conn)

    async def _mempalace_list_drawers(self, wing: str = "default", room: str = "general", limit: int = 50) -> dict:
        """列出 drawer。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, content, created_at FROM memories WHERE wing = ? AND room = ? ORDER BY created_at DESC LIMIT ?",
                (wing, room, limit),
            )
            rows = await cursor.fetchall()
            drawers = [{"id": row["id"], "preview": row["content"][:200], "created_at": row["created_at"]} for row in rows]
            return {"wing": wing, "room": room, "drawers": drawers, "total": len(drawers)}
        finally:
            await self._pool.release(conn)

    async def _mempalace_get_drawer(self, id: str) -> dict:
        """获取单个 drawer。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, content, metadata, created_at FROM memories WHERE id = ?",
                (id,),
            )
            row = await cursor.fetchone()
            if not row:
                return {"error": "not found"}
            meta_raw = row["metadata"]
            if isinstance(meta_raw, str):
                try:
                    meta_raw = json.loads(meta_raw)
                except (json.JSONDecodeError, TypeError):
                    meta_raw = {}
            return {"id": row["id"], "content": row["content"], "metadata": meta_raw, "created_at": row["created_at"]}
        finally:
            await self._pool.release(conn)

    async def _mempalace_get_taxonomy(self) -> dict:
        """获取分类。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT wing, room, COUNT(*) as count FROM memories GROUP BY wing, room ORDER BY wing, room"
            )
            rows = await cursor.fetchall()
            taxonomy: dict[str, dict] = {}
            for row in rows:
                w = row["wing"]
                if w not in taxonomy:
                    taxonomy[w] = {}
                taxonomy[w][row["room"]] = row["count"]
            return {"taxonomy": taxonomy}
        finally:
            await self._pool.release(conn)

    async def _mempalace_status(self) -> dict:
        """获取系统状态。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute("SELECT COUNT(*) as count FROM memories")
            row = await cursor.fetchone()
            memory_count = row["count"] if row else 0
            return {
                "status": "ok",
                "memory_count": memory_count,
                "pipeline": self._engine.get_status(),
                "scheduler": self._scheduler.get_status(),
                "kg": self._kg.get_stats(),
            }
        finally:
            await self._pool.release(conn)

    # ---- 新增工具实现 ----

    async def _memory_search(self, query: str, top_k: int = 20) -> dict:
        """多策略检索。"""
        results = await self._temp_engine.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.score,
                    "sources": r.sources,
                }
                for r in results
            ],
            "total": len(results),
        }

    async def _memory_ingest(self, content: str, source: str = "") -> dict:
        """管线摄取。"""
        msg_id = await self._engine.ingest(content, source=source)
        return {"id": msg_id, "status": "accepted" if msg_id else "duplicate"}

    async def _kg_query(self, entity: str) -> dict:
        """知识图谱查询。"""
        context = await self._kg.get_entity_context(entity)
        if not context["entity"]:
            return {"error": f"entity '{entity}' not found"}
        return {
            "entity": {
                "name": context["entity"].name,
                "type": context["entity"].entity_type,
                "metadata": context["entity"].metadata,
            },
            "relations": [
                {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                for r in context["relations"]
            ],
            "neighbors": context["neighbors"],
        }

    async def _pipeline_run(self, content: str, source: str = "") -> dict:
        """手动触发管线。"""
        msg_id = await self._engine.ingest(content, source=source)
        return {"id": msg_id, "status": "accepted" if msg_id else "duplicate"}

    async def _reflect_trigger(self) -> dict:
        """手动触发 reflect。"""
        result = await self._scheduler.trigger_reflect()
        return {"status": "completed", "insights": len(result.get("insights", [])), "gaps": len(result.get("gaps", []))}

    async def _consolidate_trigger(self) -> dict:
        """手动触发 consolidation。"""
        result = await self._scheduler.trigger_consolidate()
        return {"status": "completed", "details": result}

    async def _system_health(self) -> dict:
        """系统健康检查。"""
        status = "ok"
        conn = None
        try:
            conn = await self._pool.acquire()
            await conn.execute("SELECT 1")
        except Exception as e:
            status = f"error: {e}"
        finally:
            if conn is not None:
                try:
                    await self._pool.release(conn)
                except Exception:
                    pass
        return {
            "status": status,
            "chroma": "ok" if self._chroma is not None else "unavailable",
            "kg_entities": self._kg.get_stats()["entities"],
            "pipeline_running": self._engine.get_status()["running"],
        }

    async def _system_stats(self) -> dict:
        """系统统计。"""
        stats = {
            "pipeline": self._engine.get_status(),
            "scheduler": self._scheduler.get_status(),
            "kg": self._kg.get_stats(),
        }
        if self._mental_models:
            stats["mental_models"] = self._mental_models.get_stats()
        return stats

    # ---- v3.0 新增工具实现 ----

    async def _mental_models_query(self, category: str = "", key: str = "") -> dict:
        """查询用户心智模型/信念（Hindsight: Mental Models）。"""
        if not self._mental_models:
            return {"error": "mental models not enabled"}
        if category and key:
            belief = await self._mental_models.get_belief(category, key)
            if not belief:
                return {"error": "belief not found"}
            return {
                "category": belief.category,
                "key": belief.key,
                "value": belief.value,
                "confidence": belief.confidence,
                "evidence_count": belief.evidence_count,
            }
        elif category:
            beliefs = await self._mental_models.get_beliefs_by_category(category)
        else:
            beliefs = await self._mental_models.get_all_beliefs()
        return {
            "beliefs": [
                {
                    "category": b.category,
                    "key": b.key,
                    "value": b.value,
                    "confidence": b.confidence,
                    "evidence_count": b.evidence_count,
                }
                for b in beliefs
            ],
            "total": len(beliefs),
        }

    async def _mental_models_strong(self, min_confidence: float = 0.7) -> dict:
        """获取高置信度信念（用于注入 Agent 上下文）。"""
        if not self._mental_models:
            return {"error": "mental models not enabled"}
        beliefs = await self._mental_models.get_strong_beliefs(
            min_confidence=min_confidence, limit=20
        )
        return {
            "beliefs": [
                {
                    "category": b.category,
                    "key": b.key,
                    "value": b.value,
                    "confidence": b.confidence,
                    "evidence_count": b.evidence_count,
                }
                for b in beliefs
            ],
            "total": len(beliefs),
        }

    async def _memory_compress(self) -> dict:
        """手动触发记忆压缩（MemPalace: AAAK-inspired）。"""
        if not self._compressor:
            return {"error": "compression not enabled"}
        result = await self._compressor.compress()
        return {
            "status": "completed",
            "compressed": result.get("compressed", 0),
            "archived": result.get("archived", 0),
            "groups": result.get("groups", 0),
        }

    async def _verbatim_recall(self, query: str, limit: int = 10) -> dict:
        """逐字回溯原始记忆（MemPalace: Verbatim Storage）。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                """SELECT id, memory_id, raw_content, source, created_at
                   FROM verbatim
                   WHERE raw_content LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{query}%", limit),
            )
            rows = await cursor.fetchall()
            results = [
                {
                    "id": row["id"],
                    "memory_id": row["memory_id"],
                    "content": row["raw_content"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            return {"results": results, "total": len(results)}
        finally:
            await self._pool.release(conn)

    # ========================================================================
    # 增强版 memory_context：精准注入 + per-session 去重 + 动态预算 + 权威排序
    # ========================================================================

    async def _memory_context(self, query: str, top_k: int = 10,
                              session_id: str = "",
                              auto_inject: bool = True,
                              budget: int = 4000) -> dict:
        """获取完整上下文（增强版）：支持自动注入、per-session 去重、动态预算、阈值过滤、权威排序、摘要压缩。

        auto_inject=True 时，返回注入文本（可直接插入 system prompt），
        auto_inject=False 时返回原始检索结果。
        """
        # 1. 记忆检索
        results = await self._engine.search(query, top_k=top_k)

        # 2. 心智模型
        mental_models_text = ""
        if self._mental_models:
            mental_models_text = await self._mental_models.format_for_context(max_items=10)

        if not auto_inject:
            return {
                "memories": [
                    {"id": r.id, "text": r.text, "score": r.score, "sources": r.sources}
                    for r in results
                ],
                "mental_models": mental_models_text,
                "total_memories": len(results),
            }

        # ---- 增强注入模式 ----

        # 3. per-session 去重
        self._cleanup_expired_sessions()
        injected_ids = self._session_injected.get(session_id, set()) if session_id else set()
        total_candidates = len(results)

        # 4. 动态预算：按输入长度分配
        input_len = len(query)
        if input_len < 50:
            max_items = 2
        elif input_len < 200:
            max_items = 4
        else:
            max_items = 6

        # 5. 阈值过滤 + 去重 + 预算截断
        filtered = []
        for r in results:
            if session_id and r.id in injected_ids:
                continue
            if r.score < self._threshold_override:
                continue
            filtered.append(r)
            if len(filtered) >= max_items:
                break

        # 6. Bug fix (BUG-6): 批量获取权威等级 + trust score，单条 IN 查询取回全部，
        #   避免每条记忆独立 acquire/release 连接（原实现 6 条记忆=12 次独立查询）。
        auth_map, trust_map = await self._get_rank_meta([r.id for r in filtered])

        ranked = []
        for r in filtered:
            authority = auth_map.get(r.id, "medium")
            trust_score = trust_map.get(r.id, 0.5)
            ranked.append((r, authority, trust_score))

        auth_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        ranked.sort(key=lambda x: auth_order.get(x[1], 99))

        # 7. 组装注入文本
        lines = []
        token_cost = 0
        for r, authority, _ in ranked:
            label = {"critical": "[权威]", "high": "[高]", "medium": "[中]", "low": "[低]"}.get(authority, "[中]")
            text = r.text[:300]
            line = f"{label} {text}"
            if token_cost + len(line) > budget:
                break
            lines.append(line)
            token_cost += len(line)
            if session_id:
                if session_id not in self._session_injected:
                    self._session_injected[session_id] = set()
                self._session_injected[session_id].add(r.id)
                self._session_last_active[session_id] = time.time()

        # 8. 组装注入文本
        if lines:
            context_parts = [
                "## 记忆上下文",
                "以下信息来自 pudica-memory，按权威等级排列：",
                "",
            ]
            context_parts.extend(lines)
            context_parts.append("")
            context_parts.append("请优先使用 [权威] 和 [高] 等级的记忆。[低] 等级的记忆可能不准确，仅供参考。")
            context = "\n".join(context_parts)
        else:
            context = ""

        return {
            "context": context,
            "injected_ids": [r.id for r, _, _ in ranked],
            "stats": {
                "total_candidates": total_candidates,
                "filtered_out": total_candidates - len(ranked),
                "injected": len(ranked),
                "token_cost": token_cost,
            },
            "mental_models": mental_models_text,
        }

    # ---- 注入层辅助方法 ----

    def _cleanup_expired_sessions(self) -> None:
        """清理超时会话（30 分钟无活动）。"""
        now = time.time()
        expired = [
            sid for sid, last in self._session_last_active.items()
            if now - last > self._SESSION_TTL
        ]
        for sid in expired:
            self._session_injected.pop(sid, None)
            self._session_last_active.pop(sid, None)

    async def _get_rank_meta(self, memory_ids: list[str]) -> tuple[dict[str, str], dict[str, float]]:
        """批量获取记忆的 authority 与 trust_score（Bug fix BUG-6）。

        单条 IN 查询取回全部，替代原先逐 id 各 acquire/release 一次连接的实现。

        Args:
            memory_ids: 记忆 id 列表

        Returns:
            (authority_map, trust_map): {id: authority}, {id: trust_score}
        """
        if not memory_ids:
            return {}, {}
        auth_map: dict[str, str] = {}
        trust_map: dict[str, float] = {}
        try:
            conn = await self._pool.acquire()
            try:
                placeholders = ",".join("?" * len(memory_ids))
                cursor = await conn.execute(
                    f"SELECT id, authority, trust_score FROM memories WHERE id IN ({placeholders})",
                    memory_ids,
                )
                rows = await cursor.fetchall()
                for row in rows:
                    auth_map[row["id"]] = row["authority"] or "medium"
                    trust = row["trust_score"]
                    trust_map[row["id"]] = trust if trust is not None else 0.5
            finally:
                await self._pool.release(conn)
        except Exception:
            pass  # 失败时调用方用默认值兜底
        return auth_map, trust_map