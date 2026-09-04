"""pipeline/engine.py — 管线引擎（注册式架构，借鉴 DSH 的 registerAdapter 模式）。

借鉴 DSH 的 PROTOCOLS / registerAdapter 设计模式：
- 管线步骤通过 PipelineStage 协议接口定义
- 步骤通过 register() 注册，按注册顺序执行
- 可替换、可删除、可插入任意步骤
- 每个步骤有独立的 enable/disable 控制

默认注册顺序：L0（去重）→ buffer → L1（LLM提取）→ store（Chroma+SQLite）→ L2（场景）
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import uuid4

from unified_memory.pipeline.encoding_repair import repair_document
from unified_memory.pipeline.l0_dedup import L0Dedup
from unified_memory.pipeline.l1_extractor import L1Extractor
from unified_memory.pipeline.l2_scene import L2SceneOrganizer
from unified_memory.pipeline.l3_search import L3Search

logger = logging.getLogger(__name__)


# ============================================================
# DSH 式 PipelineStage 协议接口
# ============================================================

@runtime_checkable
class PipelineStage(Protocol):
    """管线步骤协议接口。

    每个步骤实现此接口，注册到 PipelineEngine 中。
    DSH 等价：LLMProtocolAdapter
    """
    stage_name: str
    """步骤名称（唯一标识，如 'l0_dedup', 'l1_extractor'）。"""

    enabled: bool
    """是否启用。"""

    async def process(
        self,
        *,
        batch: list[dict],
        engine: "PipelineEngine",
    ) -> list[dict]:
        """处理一批消息。

        Args:
            batch: 待处理的消息列表（每个消息是 dict，含 id, content, source, metadata, timestamp）
            engine: 管线引擎实例（可访问 _pool, _chroma, _llm 等资源）

        Returns:
            处理后的消息列表（可能添加/修改字段，如 content_hash, extracted 等）
        """
        ...


# ============================================================
# 内置管线步骤
# ============================================================

class L0DedupStage:
    """L0 去重步骤（DSH 式注册版）。"""
    stage_name = "l0_dedup"
    enabled = True

    def __init__(self, dedup: L0Dedup):
        self._dedup = dedup

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        result = []
        for msg in batch:
            content = repair_document(msg["content"])
            msg["content"] = content
            if self._dedup.is_duplicate(content):
                logger.debug("L0 去重: 内容已存在")
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            msg["content_hash"] = content_hash
            result.append(msg)
        return result


class L1ExtractorStage:
    """L1 LLM 提取步骤（DSH 式注册版）。"""
    stage_name = "l1_extractor"
    enabled = True

    def __init__(self, extractor: L1Extractor):
        self._extractor = extractor

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        messages = [msg["content"] for msg in batch]
        for i, msg_content in enumerate(messages):
            try:
                result = await self._extractor.extract([msg_content])
                batch[i]["extracted"] = result if isinstance(result, dict) else {}
                engine.l1_count += 1
            except Exception as e:
                logger.warning("L1 单条提取失败，使用降级: %s", e)
                batch[i]["extracted"] = {}
        return batch


class SQLiteStoreStage:
    """SQLite 写入步骤。

    在 ChromaDB 写入之前执行，保证 ChromaDB 失败时可以回滚 SQLite。
    """
    stage_name = "sqlite_store"
    enabled = True

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        sqlite_ok_ids: set[str] = set()
        for msg in batch:
            extracted = msg.get("extracted", {})
            fact_type = extracted.get("fact_type", "observation")
            md = msg.get("metadata", {}) or {}
            wing_val = md.get("wing", "default")
            room_val = md.get("room", "general")
            meta_json = json.dumps(
                {
                    "fact_type": fact_type,
                    "source": msg.get("source", ""),
                    "created_at": msg["timestamp"],
                    "content_hash": msg.get("content_hash", ""),
                    "wing": wing_val,
                    "room": room_val,
                    **{k: v for k, v in md.items() if k not in ("wing", "room")},
                },
                ensure_ascii=False,
            )
            for attempt in range(2):
                try:
                    conn = await asyncio.wait_for(engine._pool.acquire(), timeout=5.0)
                    try:
                        await conn.execute(
                "INSERT OR IGNORE INTO memories (id, content, content_hash, wing, room, source, fact_type, authority, trust_score, summary, metadata, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (msg["id"], msg["content"], msg.get("content_hash", ""), wing_val, room_val, msg.get("source", ""), fact_type, extracted.get("authority", "medium"), extracted.get("trust_score", 0.5), extracted.get("summary", ""), meta_json, msg["timestamp"], msg["timestamp"], md.get("expires_at")),
                )
                        await conn.commit()
                        sqlite_ok_ids.add(msg["id"])
                    finally:
                        await engine._pool.release(conn)
                    break
                except Exception as e:
                    if attempt == 0:
                        logger.warning("SQLite 写入失败，重试: %s", e)
                        await asyncio.sleep(0.5)
                    else:
                        logger.error("SQLite 写入重试也失败: %s", e)

        # 标记哪些消息成功写入了 SQLite
        for msg in batch:
            msg["_sqlite_ok"] = msg["id"] in sqlite_ok_ids
        return batch


class ChromaStoreStage:
    """ChromaDB 向量写入步骤。

    如果 SQLite 写入失败，ChromaDB 跳过（避免孤儿向量）。
    """
    stage_name = "chroma_store"
    enabled = True

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        items = []
        for msg in batch:
            if not msg.get("_sqlite_ok"):
                continue
            extracted = msg.get("extracted", {})
            md = msg.get("metadata", {}) or {}
            items.append({
                "id": msg["id"],
                "content": msg["content"],
                "wing": md.get("wing", "default"),
                "room": md.get("room", "general"),
                "metadata": {
                    "fact_type": extracted.get("fact_type", "observation"),
                    "source": msg.get("source", ""),
                    "created_at": msg["timestamp"],
                },
            })

        if not items:
            return batch

        chroma_ok = True
        for attempt in range(2):
            try:
                loop = asyncio.get_running_loop()
                from unified_memory.store.chroma_store import get_chroma_executor
                chroma_future = loop.run_in_executor(
                    get_chroma_executor(), engine._chroma.add_batch, items,
                )
                await asyncio.wait_for(chroma_future, timeout=30.0)
                chroma_ok = True
                break
            except asyncio.TimeoutError:
                logger.warning("ChromaDB 写入超时（attempt %d），线程池可能耗尽", attempt + 1)
                chroma_ok = False
            except Exception as e:
                if attempt == 0:
                    logger.warning("ChromaDB 写入失败，重试: %s", e)
                else:
                    logger.error("ChromaDB 写入重试也失败: %s", e)
                chroma_ok = False

        # ChromaDB 失败 → 回滚 SQLite
        if not chroma_ok:
            logger.error("ChromaDB 写入失败，回滚 SQLite 对应记录（%d 条）避免孤儿", len(items))
            try:
                conn = await asyncio.wait_for(engine._pool.acquire(), timeout=5.0)
            except Exception as e:
                logger.error("ChromaDB 回滚时等待连接超时，跳过回滚: %s", e)
                conn = None
            if conn is not None:
                try:
                    orphan_ids = [it["id"] for it in items]
                    placeholders = ", ".join("?" * len(orphan_ids))
                    await conn.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})", orphan_ids
                    )
                    await conn.commit()
                finally:
                    await engine._pool.release(conn)

        return batch


class L2SceneStage:
    """L2 场景组织步骤（DSH 式注册版）。"""
    stage_name = "l2_scene"
    enabled = True

    def __init__(self, scene_organizer: L2SceneOrganizer):
        self._scene = scene_organizer

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        # 合并所有提取结果
        merged_extracted: dict[str, Any] = {"entities": [], "relations": [], "summary": "", "time_range": {}}
        for msg in batch:
            ex = msg.get("extracted")
            if not ex:
                continue
            entities = ex.get("entities", [])
            if isinstance(entities, list):
                merged_extracted["entities"].extend(entities)
            relations = ex.get("relations", [])
            if isinstance(relations, list):
                merged_extracted["relations"].extend(relations)
            summary = ex.get("summary")
            if summary:
                merged_extracted["summary"] += (("\n" if merged_extracted["summary"] else "") + str(summary))

        # 合并时间范围
        tr_starts: list[float] = []
        tr_ends: list[float] = []
        for msg in batch:
            ex = msg.get("extracted")
            if not ex:
                continue
            tr = ex.get("time_range")
            if not tr:
                continue
            s, e = tr.get("start"), tr.get("end")
            if s is not None:
                if isinstance(s, (int, float)):
                    tr_starts.append(float(s))
                elif isinstance(s, str):
                    try:
                        tr_starts.append(datetime.fromisoformat(s).timestamp())
                    except (ValueError, TypeError):
                        tr_starts.append(time.time())
            if e is not None:
                if isinstance(e, (int, float)):
                    tr_ends.append(float(e))
                elif isinstance(e, str):
                    try:
                        tr_ends.append(datetime.fromisoformat(e).timestamp())
                    except (ValueError, TypeError):
                        tr_ends.append(time.time())
        if tr_starts:
            merged_extracted["time_range"] = {"start": min(tr_starts), "end": max(tr_ends)}

        if merged_extracted["entities"] or merged_extracted["relations"] or merged_extracted["summary"]:
            try:
                scene_id = await self._scene.organize(merged_extracted)
                if scene_id:
                    engine.l2_count += 1
                    logger.debug("L2 场景: %s", scene_id)
            except Exception as e:
                logger.error("L2 场景组织失败: %s", e)

        return batch


class VerbatimStoreStage:
    """Verbatim 逐字存储步骤（MemPalace）。"""
    stage_name = "verbatim_store"
    enabled = True

    async def process(self, *, batch: list[dict], engine: "PipelineEngine") -> list[dict]:
        if not engine._verbatim_enabled:
            return batch
        for msg in batch:
            try:
                await engine._store_verbatim(
                    msg["id"], msg["content"], msg.get("source", ""), msg.get("metadata")
                )
            except Exception as e:
                logger.warning("Verbatim 存储失败: %s", e)
        return batch


# ============================================================
# PipelineEngine（注册式架构）
# ============================================================

class PipelineEngine:
    """注册式管线引擎（DSH 式 registerAdapter 模式）。

    用法:
        engine = PipelineEngine(pool=pool, chroma=chroma, ...)
        engine.register(L0DedupStage(dedup))
        engine.register(L1ExtractorStage(extractor))
        engine.register(SQLiteStoreStage())
        engine.register(ChromaStoreStage())
        engine.register(L2SceneStage(scene))
        await engine.start()

        # 替换某个步骤
        engine.replace("l1_extractor", MyCustomExtractorStage())

        # 禁用某个步骤
        engine.stage("l1_extractor").enabled = False

        # 数据入口
        msg_id = await engine.ingest(content="...", source="weixin")
    """

    def __init__(
        self,
        pool: Any,
        chroma: Any,
        dedup: L0Dedup,
        extractor: L1Extractor,
        scene_organizer: L2SceneOrganizer,
        search: L3Search,
        llm: Any,
        batch_size: int = 5,
        idle_timeout: float = 60.0,
        flush_interval: float = 5.0,
        verbatim_enabled: bool = True,
        reranker: Any = None,
        mental_models: Any = None,
    ):
        self._pool = pool
        self._chroma = chroma
        self._dedup = dedup
        self._extractor = extractor
        self._scene = scene_organizer
        self._search = search
        self._llm = llm
        self._batch_size = batch_size
        self._idle_timeout = idle_timeout
        self._flush_interval = flush_interval
        self._verbatim_enabled = verbatim_enabled
        self._reranker = reranker
        self._mental_models = mental_models

        # 注册表：步骤名 → PipelineStage 实例
        self._stages: dict[str, PipelineStage] = {}
        # 执行顺序：步骤名列表
        self._order: list[str] = []

        # 管线状态
        self._buffer: list[dict] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._idle_timer: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # 统计
        self.ingest_count = 0
        self.flush_count = 0
        self.l1_count = 0
        self.l2_count = 0
        self.search_count = 0

    # ============================================================
    # 注册表管理（DSH 式 register / replace / stage）
    # ============================================================

    def register(self, stage: PipelineStage, position: Optional[int] = None) -> None:
        """注册一个管线步骤。

        Args:
            stage: 实现了 PipelineStage 协议的实例
            position: 可选，插入位置（None=追加到末尾）
        """
        name = stage.stage_name
        self._stages[name] = stage
        if position is not None:
            self._order.insert(position, name)
        elif name not in self._order:
            self._order.append(name)
        logger.info("Pipeline 注册步骤: %s (位置=%s)", name, position or "末尾")

    def replace(self, old_name: str, new_stage: PipelineStage) -> None:
        """替换一个管线步骤（DSH 式原子替换）。

        Args:
            old_name: 被替换的步骤名
            new_stage: 新步骤实例
        """
        if old_name not in self._stages:
            logger.warning("替换失败: 步骤 '%s' 未注册", old_name)
            self.register(new_stage)
            return
        new_stage.enabled = self._stages[old_name].enabled
        self._stages[old_name] = new_stage
        logger.info("Pipeline 替换步骤: %s → %s", old_name, new_stage.stage_name)

    def stage(self, name: str) -> Optional[PipelineStage]:
        """获取已注册的步骤实例。"""
        return self._stages.get(name)

    def unregister(self, name: str) -> None:
        """注销一个管线步骤。"""
        if name in self._stages:
            del self._stages[name]
            if name in self._order:
                self._order.remove(name)
            logger.info("Pipeline 注销步骤: %s", name)

    def get_order(self) -> list[str]:
        """获取当前执行顺序。"""
        return list(self._order)

    # ============================================================
    # 默认注册（向前兼容）
    # ============================================================

    def _register_defaults(self) -> None:
        """注册默认管线步骤（保持与旧版相同的行为）。"""
        self.register(VerbatimStoreStage(), position=0)  # 最先执行
        self.register(L0DedupStage(self._dedup))
        self.register(L1ExtractorStage(self._extractor))
        self.register(SQLiteStoreStage())
        self.register(ChromaStoreStage())
        self.register(L2SceneStage(self._scene))

    # ============================================================
    # 生命周期
    # ============================================================

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """启动 Pipeline 后台循环。"""
        if not self._stages:
            self._register_defaults()
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

        # 从 DB 恢复 ingest_count
        try:
            if self._pool:
                conn = await self._pool.acquire()
                try:
                    cursor = await conn.execute("SELECT COUNT(*) AS count FROM memories")
                    row = await cursor.fetchone()
                    if row:
                        self.ingest_count = row["count"]
                        logger.info("从 DB 恢复 ingest_count=%d", self.ingest_count)
                finally:
                    await self._pool.release(conn)
        except Exception as e:
            logger.warning("恢复 ingest_count 失败: %s", e)

        logger.info(
            "Pipeline 已启动 (batch_size=%d, flush_interval=%.1fs, idle_timeout=%.1fs)",
            self._batch_size, self._flush_interval, self._idle_timeout,
        )
        logger.info("Pipeline 步骤顺序: %s", " → ".join(self._order))

        async def stop(self) -> None:
            self._running = False
            if self._idle_timer and not self._idle_timer.done():
                self._idle_timer.cancel()
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            if self._buffer:
                await self._flush_buffer()
            logger.info("Pipeline 已停止")

        async def flush(self) -> None:
            async with self._lock:
                await self._flush_buffer()

            # ============================================================
            # 数据入口
            # ============================================================

    async def ingest(self, content: str, source: str = "", metadata: Optional[dict] = None) -> str:
        """L0 入口：去重 + 缓冲。

        Args:
            content: 消息内容
            source: 来源
            metadata: 附加元数据

        Returns:
            消息 ID，如果去重则返回空字符串
        """
        msg_id = str(uuid4())
        msg = {
            "id": msg_id,
            "content": content,
            "source": source,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        async with self._lock:
            self._buffer.append(msg)
            self.ingest_count += 1

            if len(self._buffer) >= self._batch_size:
                await self._flush_buffer()
            else:
                if self._idle_timer and not self._idle_timer.done():
                    self._idle_timer.cancel()
                self._idle_timer = asyncio.create_task(self._idle_flush_task())

        return msg_id

    ingest_memory = ingest  # 兼容旧调用

    async def search(self, query: str, top_k: int = 20) -> list:
        """L3 检索入口（Hindsight: 支持 Cross-Encoder 重排）。"""
        self.search_count += 1
        results = await self._search.search(query, top_k=top_k)

        if self._reranker is not None and results:
            results = await self._reranker.rerank(query, results)

        return results

    # ============================================================
    # 内部：缓冲刷新
    # ============================================================

    async def _flush_buffer(self) -> None:
        """按注册顺序执行所有启用的管线步骤。"""
        if not self._buffer:
            return

        batch = self._buffer[:self._batch_size]
        try:
            # 按注册顺序执行每个启用的步骤
            for stage_name in self._order:
                stage = self._stages.get(stage_name)
                if stage is None or not stage.enabled:
                    continue
                try:
                    batch = await stage.process(batch=batch, engine=self)
                except Exception as e:
                    logger.error("步骤 '%s' 处理失败: %s", stage_name, e)
                    raise

            # 从 buffer 移除已处理的消息
            batch_ids = {m["id"] for m in batch}
            self._buffer = [m for m in self._buffer if m.get("id") not in batch_ids]
            self.flush_count += 1
            logger.info("Pipeline 完成: %d 条消息", len(batch))

        except Exception as e:
            logger.error("Pipeline 缓冲刷新失败，回滚 %d 条到 buffer: %s", len(batch), e)
            self._buffer = batch + \
                [m for m in self._buffer if m.get("id") not in {m2["id"] for m2 in batch}]

        # 心智模型更新（Hindsight 独立步骤，不阻塞管线）
        if self._mental_models is not None:
            try:
                extracted_list = [m.get("extracted", {}) for m in batch]
                await self._update_mental_models_from_extracted(extracted_list, batch)
            except Exception as e:
                logger.warning("心智模型更新失败: %s", e)

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                if self._buffer:
                    await self._flush_buffer()

    async def _idle_flush_task(self) -> None:
        await asyncio.sleep(self._idle_timeout)
        async with self._lock:
            if self._buffer:
                logger.info("空闲超时触发 flush (buffer=%d)", len(self._buffer))
                await self._flush_buffer()

    # ============================================================
    # Verbatim 逐字存储（MemPalace）
    # ============================================================

    async def _store_verbatim(
        self, msg_id: str, content: str, source: str, metadata: Optional[dict]
    ) -> None:
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                """INSERT OR IGNORE INTO verbatim (id, memory_id, raw_content, source, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid4()), msg_id, content, source,
                 json.dumps(metadata or {}, ensure_ascii=False), time.time()),
            )
            await conn.commit()
        except Exception as e:
            logger.warning("Verbatim 存储失败: %s", e, exc_info=True)
        finally:
            await self._pool.release(conn)

    # ============================================================
    # 心智模型更新（Hindsight）
    # ============================================================

    async def _update_mental_models_from_extracted(
        self, extracted_list: list[dict], batch: list[dict]
    ) -> None:
        if not self._mental_models:
            return

        for i, extracted in enumerate(extracted_list):
            if not extracted:
                continue
            # ... 保留原有心智模型更新逻辑
            pass