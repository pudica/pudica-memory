"""pipeline/engine.py — 管线引擎（缓冲写入 + 定时 flush）。

串行管线引擎，concurrency=1。
参考 TencentDB pipeline-manager.ts 的 SerialQueue 模式。
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from unified_memory.pipeline.l0_dedup import L0Dedup
from unified_memory.pipeline.l1_extractor import L1Extractor
from unified_memory.pipeline.l2_scene import L2SceneOrganizer
from unified_memory.pipeline.l3_search import L3Search

logger = logging.getLogger(__name__)


class PipelineEngine:
    """串行管线引擎，缓冲写入 + 定时 flush。

    数据流：L0（去重）→ 缓冲 → L1（LLM提取）→ 写入 Chroma+SQLite + L2（场景）
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
        """
        Args:
            pool: SQLitePool 实例
            chroma: ChromaStore 实例
            dedup: L0Dedup 实例
            extractor: L1Extractor 实例
            scene_organizer: L2SceneOrganizer 实例
            search: L3Search 实例
            llm: LLM 客户端
            batch_size: L1 批处理大小
            idle_timeout: 空闲超时（秒），超过此时间自动触发 flush
            flush_interval: 定时 flush 间隔（秒）
            verbatim_enabled: 是否启用 Verbatim 逐字存储（MemPalace）
            reranker: Reranker 实例（Hindsight Cross-Encoder），可选
            mental_models: MentalModelStore 实例（Hindsight），可选
        """
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

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Pipeline 是否在运行中。"""
        return self._running

    async def start(self) -> None:
        """启动 Pipeline 后台循环。"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Pipeline 已启动 (batch_size=%d, flush_interval=%.1fs, idle_timeout=%.1fs)",
                     self._batch_size, self._flush_interval, self._idle_timeout)

    async def stop(self) -> None:
        """停止管线引擎。"""
        self._running = False
        if self._idle_timer and not self._idle_timer.done():
            self._idle_timer.cancel()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # 刷剩余缓冲区
        if self._buffer:
            await self._flush_buffer()
        logger.info("Pipeline 已停止")

    async def flush(self) -> None:
        """手动触发缓冲区刷新。"""
        async with self._lock:
            await self._flush_buffer()

    # ------------------------------------------------------------------
    # 数据入口
    # ------------------------------------------------------------------

    async def ingest(self, content: str, source: str = "", metadata: Optional[dict] = None) -> str:
        """L0 入口：去重 + 缓冲。别名 ingest_memory 兼容旧调用。

        Args:
            content: 消息内容
            source: 来源 (如 weixin, test)
            metadata: 附加元数据

        Returns:
            消息 ID，如果去重则返回空字符串
        """
        # L0：去重
        if self._dedup.is_duplicate(content):
            logger.debug("L0 去重: 内容已存在")
            return ""

        msg_id = str(uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        msg = {
            "id": msg_id,
            "content": content,
            "content_hash": content_hash,
            "source": source,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        # MemPalace: Verbatim 逐字存储（在管线处理前保存原始输入）
        if self._verbatim_enabled:
            await self._store_verbatim(msg_id, content, source, metadata)

        # 在锁内把消息加入 buffer，避免与 _flush_loop / _idle_flush_task 竞态
        async with self._lock:
            self._buffer.append(msg)
            self.ingest_count += 1

            if len(self._buffer) >= self._batch_size:
                await self._flush_buffer()
            else:
                # 在锁内重置空闲定时器，消除竞态
                if self._idle_timer and not self._idle_timer.done():
                    self._idle_timer.cancel()
                self._idle_timer = asyncio.create_task(self._idle_flush_task())

        return msg_id

    ingest_memory = ingest  # 兼容旧调用

    async def search(self, query: str, top_k: int = 20) -> list:
        """L3 检索入口（Hindsight: 支持 Cross-Encoder 重排）。

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            检索结果列表
        """
        self.search_count += 1  # 统计检索次数
        results = await self._search.search(query, top_k=top_k)

        # Hindsight: Cross-Encoder 重排
        if self._reranker is not None and results:
            results = await self._reranker.rerank(query, results)

        return results

    # ------------------------------------------------------------------
    # 内部：缓冲刷新
    # ------------------------------------------------------------------

    async def _flush_buffer(self) -> None:
        """L1：批量提取 + 写入 Chroma + SQLite + L2 场景组织。"""
        if not self._buffer:
            return

        # 取消空闲定时器（防止与 _flush_loop 并发触发）
        if self._idle_timer and not self._idle_timer.done():
            self._idle_timer.cancel()

        batch = self._buffer[:self._batch_size]
        self._buffer = self._buffer[self._batch_size:]

        logger.info("L1 缓冲写入: %d 条消息", len(batch))
        self.flush_count += 1

        # 1. 调用 LLM 提取（每条消息独立提取，避免整批共享一个 fact_type）
        messages = [msg["content"] for msg in batch]
        per_msg_extracted: list[dict] = []
        for msg_content in messages:
            try:
                result = await self._extractor.extract([msg_content])
                per_msg_extracted.append(result if isinstance(result, dict) else {})
            except Exception as e:
                logger.warning("L1 单条提取失败，使用降级: %s", e)
                per_msg_extracted.append({})
            self.l1_count += 1

        # 构建 ChromaDB 写入项（稍后写入）
        items = []
        for i, msg in enumerate(batch):
            extracted = per_msg_extracted[i] if i < len(per_msg_extracted) else {}
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

        # 2. 写入 SQLite（先写 SQLite，再写 ChromaDB；避免 ChromaDB 成功但 SQLite 失败产生孤儿向量）
        sqlite_ok_ids: set[str] = set()
        for i, msg in enumerate(batch):
            extracted = per_msg_extracted[i] if i < len(per_msg_extracted) else {}
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
                    conn = await self._pool.acquire()
                    try:
                        await conn.execute(
                            "INSERT OR IGNORE INTO memories (id, content, content_hash, wing, room, source, fact_type, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (msg["id"], msg["content"], msg.get("content_hash", ""), wing_val, room_val, msg.get("source", ""), fact_type, meta_json, msg["timestamp"], msg["timestamp"]),
                        )
                        await conn.commit()
                        sqlite_ok_ids.add(msg["id"])
                    finally:
                        await self._pool.release(conn)
                    break
                except Exception as e:
                    if attempt == 0:
                        logger.warning("SQLite 写入失败，重试: %s", e)
                        await asyncio.sleep(0.5)
                    else:
                        logger.error("SQLite 写入重试也失败: %s", e)

        # 3. 写入 ChromaDB（仅写入 SQLite 成功的消息，失败重试 1 次）
        chroma_items = [it for it in items if it["id"] in sqlite_ok_ids]
        if chroma_items:
            for attempt in range(2):
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._chroma.add_batch, chroma_items)
                    break
                except Exception as e:
                    if attempt == 0:
                        logger.warning("ChromaDB 写入失败，重试: %s", e)
                    else:
                        logger.error("ChromaDB 写入重试也失败: %s", e)

        # 4. 触发 L2：场景组织（仅当 SQLite 写入成功后才更新 KG，避免孤儿实体）
        # 将所有提取结果中的实体/关系/摘要合并后传给 scene organizer
        merged_extracted: dict[str, Any] = {"entities": [], "relations": [], "summary": "", "time_range": {}}
        for ex in per_msg_extracted:
            if ex:
                entities = ex.get("entities", [])
                if isinstance(entities, list):
                    merged_extracted["entities"].extend(entities)
                relations = ex.get("relations", [])
                if isinstance(relations, list):
                    merged_extracted["relations"].extend(relations)
                summary = ex.get("summary")
                if summary:
                    merged_extracted["summary"] += (("\n" if merged_extracted["summary"] else "") + str(summary))
        # 合并时间范围（取最早 start 和最晚 end）
        # 支持 ISO 8601 字符串和 float 时间戳
        tr_starts: list[float] = []
        tr_ends: list[float] = []
        for ex in per_msg_extracted:
            tr = ex.get("time_range") if ex else None
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
                    self.l2_count += 1
                    logger.debug("L2 场景: %s", scene_id)
            except Exception as e:
                logger.error("L2 场景组织失败: %s", e)

        # Hindsight: 从提取结果更新心智模型
        if self._mental_models is not None:
            try:
                await self._update_mental_models_from_extracted(
                    per_msg_extracted, batch
                )
            except Exception as e:
                logger.warning("心智模型更新失败: %s", e)

    async def _flush_loop(self) -> None:
        """定时 flush 循环：每 _flush_interval 秒检查一次缓冲区。"""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                if self._buffer:
                    await self._flush_buffer()

    async def _idle_flush_task(self) -> None:
        """空闲超时后自动 flush。"""
        await asyncio.sleep(self._idle_timeout)
        async with self._lock:
            if self._buffer:
                logger.info("空闲超时触发 flush (buffer=%d)", len(self._buffer))
                await self._flush_buffer()

    # ------------------------------------------------------------------
    # Verbatim 逐字存储（MemPalace）
    # ------------------------------------------------------------------

    async def _store_verbatim(
        self, msg_id: str, content: str, source: str, metadata: Optional[dict]
    ) -> None:
        """存储原始输入到 verbatim 表（MemPalace: 逐字存储）。

        在管线处理前保存原始内容，支持精确回溯。

        Args:
            msg_id: 消息 ID
            content: 原始内容
            source: 来源
            metadata: 附加元数据
        """
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

    # ------------------------------------------------------------------
    # 心智模型更新（Hindsight）
    # ------------------------------------------------------------------

    async def _update_mental_models_from_extracted(
        self, extracted_list: list[dict], batch: list[dict]
    ) -> None:
        """从 L1 提取结果中更新心智模型（Hindsight: Mental Models）。

        规则提取：
        - preference: 从 opinion 类型事实中提取偏好
        - behavior: 从 experience 类型事实中提取行为模式
        - belief: 从 world 类型事实中提取信念
        - knowledge_level: 从实体类型推断知识水平

        Args:
            extracted_list: L1 提取结果列表
            batch: 原始消息批次
        """
        if not self._mental_models:
            return

        for i, extracted in enumerate(extracted_list):
            if not extracted:
                continue
            msg_id = batch[i]["id"] if i < len(batch) else ""
            fact_type = extracted.get("fact_type", "observation")
            summary = extracted.get("summary", "")
            entities = extracted.get("entities", [])

            if not summary:
                continue

            # 根据事实类型推断心智模型类别
            if fact_type == "opinion":
                # 偏好类：从摘要中提取关键偏好
                for entity in entities[:3]:
                    await self._mental_models.upsert_belief(
                        category="preference",
                        key=entity["name"],
                        value=summary[:100],
                        source_memory_id=msg_id,
                        confidence_delta=0.15,
                    )
            elif fact_type == "experience":
                # 行为类：记录用户做过的事
                for entity in entities[:3]:
                    await self._mental_models.upsert_belief(
                        category="behavior",
                        key=entity["name"],
                        value=summary[:100],
                        source_memory_id=msg_id,
                        confidence_delta=0.1,
                    )
            elif fact_type == "world":
                # 信念类：用户持有的知识/观点
                for entity in entities[:2]:
                    await self._mental_models.upsert_belief(
                        category="belief",
                        key=entity["name"],
                        value=summary[:100],
                        source_memory_id=msg_id,
                        confidence_delta=0.1,
                    )

            # 知识水平：根据实体数量推断
            if len(entities) >= 5:
                await self._mental_models.upsert_belief(
                    category="knowledge_level",
                    key="domain_expertise",
                    value="高（单次对话涉及 %d 个实体）" % len(entities),
                    source_memory_id=msg_id,
                    confidence_delta=0.05,
                )

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """获取管线状态信息。

        Returns:
            {"running": bool, "buffer_size": int, "ingest_count": N,
             "flush_count": N, "l1_count": N, "l2_count": N}
        """
        return {
            "running": self._running,
            "buffer_size": len(self._buffer),
            "ingest_count": self.ingest_count,
            "flush_count": self.flush_count,
            "l1_count": self.l1_count,
            "l2_count": self.l2_count,
            "search_count": self.search_count,
        }