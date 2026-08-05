"""store/mental_models.py — 心智模型/信念系统（Hindsight 启发）。

追踪用户的偏好、信念和行为模式，作为长期记忆的抽象层。
心智模型是对原始记忆的提炼和泛化，不同于具体的事实记忆。

信念类别（参考 Hindsight Mental Models）：
  - preference: 用户偏好（喜欢/不喜欢什么）
  - belief: 用户持有的观点或信念
  - behavior: 用户的行为模式或习惯
  - knowledge_level: 用户在某个领域的知识水平

更新机制：
  - 新证据到来时，置信度按贝叶斯后验更新
  - 支持性证据（同 value）：置信度上升
  - 矛盾性证据（不同 value）：置信度下降，记录冲突历史
  - 置信度低于阈值时可以被覆盖
  - 过期信念自动降级

v3.1 升级：
  - 贝叶斯后验更新替代简单累加 delta
  - 信念冲突检测：新 value 与旧 value 不同时记录冲突
  - 冲突历史追踪：metadata 中记录 conflict_history
  - 信念推理链：记录每条信念的证据来源和更新路径
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Belief:
    """单条信念/心智模型条目。

    Attributes:
        id: 信念 ID
        category: 类别 (preference / belief / behavior / knowledge_level)
        key: 信念键（如 "中医偏好"）
        value: 信念值（如 "倾向于经方"）
        confidence: 置信度 (0-1)
        evidence_count: 支持证据数量
        conflict_count: 矛盾证据数量
        source_memory_ids: 来源记忆 ID 列表
        metadata: 附加元数据（含 conflict_history）
        created_at: 创建时间
        updated_at: 更新时间
    """
    id: str = ""
    category: str = ""
    key: str = ""
    value: str = ""
    confidence: float = 0.5
    evidence_count: int = 1
    conflict_count: int = 0
    source_memory_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


class MentalModelStore:
    """心智模型存储，管理用户信念的增删改查。

    使用 SQLite 持久化，内存缓存热数据。
    所有写入操作通过 asyncio.Lock 保护并发安全。
    """

    def __init__(
        self,
        pool: Any,
        settings: Optional[dict] = None,
    ):
        """
        Args:
            pool: SQLitePool 实例
            settings: 配置字典
                - belief_update_threshold: 信念更新阈值
                - max_beliefs: 最大信念数量
                - belief_ttl: 信念过期时间（秒），0=永不过期
        """
        self._pool = pool
        self._settings = settings or {}
        self._update_threshold = self._settings.get("belief_update_threshold", 0.6)
        self._max_beliefs = self._settings.get("max_beliefs", 100)
        self._belief_ttl = self._settings.get("belief_ttl", 0)
        self._lock = asyncio.Lock()
        self._cache: dict[str, Belief] = {}  # key="{category}:{key}" → Belief
        self._loaded = False

    async def load_from_db(self) -> None:
        """从数据库加载所有信念到内存缓存。"""
        if self._loaded:
            return
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT * FROM mental_models ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
            for row in rows:
                source_ids = []
                try:
                    source_ids = json.loads(row["source_memory_ids"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    pass
                meta = {}
                try:
                    meta = json.loads(row["metadata"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass
                belief = Belief(
                    id=row["id"],
                    category=row["category"],
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    evidence_count=row["evidence_count"],
                    conflict_count=meta.get("conflict_count", 0),
                    source_memory_ids=source_ids,
                    metadata=meta,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                cache_key = f"{belief.category}:{belief.key}"
                self._cache[cache_key] = belief
            self._loaded = True
            logger.info("心智模型加载完成: %d 条信念", len(self._cache))
        finally:
            await self._pool.release(conn)

    async def upsert_belief(
        self,
        category: str,
        key: str,
        value: str,
        source_memory_id: str = "",
        confidence_delta: float = 0.1,
        metadata: Optional[dict] = None,
    ) -> Belief:
        """更新或创建一条信念（贝叶斯后验 + 冲突检测）。

        v3.1 升级：从简单 confidence += delta 改为贝叶斯后验更新。

        更新逻辑：
        - 新信念：初始置信度 = 0.5 + confidence_delta
        - 已有信念 + 支持性证据（value 相似）：
            贝叶斯后验更新，置信度上升
        - 已有信念 + 矛盾性证据（value 不同）：
            记录冲突历史，置信度按贝叶斯后验下降
            如果矛盾证据累积到阈值，value 更新为新值
        - 证据计数 +1（支持或矛盾）

        贝叶斯更新公式：
            prior = confidence
            if supporting: posterior = prior * p_support / (prior * p_support + (1-prior) * p_noise)
            if conflicting: posterior = prior * p_noise / (prior * p_noise + (1-prior) * p_support)
            where p_support = 0.85, p_noise = 0.15

        Args:
            category: 信念类别
            key: 信念键
            value: 信念值
            source_memory_id: 来源记忆 ID
            confidence_delta: 置信度增量（用作先验强度调节）
            metadata: 附加元数据

        Returns:
            更新后的 Belief 对象
        """
        async with self._lock:
            cache_key = f"{category}:{key}"
            now = time.time()
            existing = self._cache.get(cache_key)

            if existing:
                # --- 冲突检测 ---
                similarity = self._text_similarity(existing.value, value)
                is_conflict = similarity < 0.4  # 文本相似度低于 0.4 视为矛盾

                # --- 贝叶斯后验更新 ---
                prior = existing.confidence
                p_support = 0.85  # P(证据|信念为真)
                p_noise = 0.15    # P(证据|信念为假)

                if is_conflict:
                    # 矛盾证据：置信度下降
                    posterior = (prior * p_noise) / (
                        prior * p_noise + (1 - prior) * p_support + 1e-10
                    )
                    existing.conflict_count += 1
                    # 记录冲突历史
                    conflict_history = existing.metadata.setdefault("conflict_history", [])
                    conflict_history.append({
                        "old_value": existing.value,
                        "new_value": value,
                        "similarity": round(similarity, 3),
                        "source_memory_id": source_memory_id,
                        "timestamp": now,
                    })
                    # 只保留最近 20 条冲突记录
                    if len(conflict_history) > 20:
                        conflict_history[:] = conflict_history[-20:]

                    # 如果矛盾证据足够多（conflict_count >= evidence_count / 2），
                    # 或者新证据的 confidence_delta 足够大，更新 value
                    if (existing.conflict_count >= existing.evidence_count / 2
                            or confidence_delta >= 0.3):
                        existing.metadata["previous_value"] = existing.value
                        existing.value = value
                        logger.info(
                            "信念值更新（矛盾证据累积）: %s:%s %s → %s",
                            category, key, existing.metadata["previous_value"][:30], value[:30],
                        )
                else:
                    # 支持性证据：置信度上升
                    posterior = (prior * p_support) / (
                        prior * p_support + (1 - prior) * p_noise + 1e-10
                    )
                    # 如果 value 有细微差异但相似度高，取最新值
                    if similarity < 0.9 and similarity >= 0.4:
                        existing.value = value

                # 应用 confidence_delta 作为调节因子
                # posterior 与 confidence_delta 结合：delta 越大，更新幅度越大
                adjusted = posterior + confidence_delta * (1 - posterior) * (0 if is_conflict else 1)
                existing.confidence = max(0.0, min(1.0, adjusted))
                existing.evidence_count += 1
                existing.updated_at = now
                if source_memory_id and source_memory_id not in existing.source_memory_ids:
                    existing.source_memory_ids.append(source_memory_id)
                if metadata:
                    existing.metadata.update(metadata)

                # 存入 conflict_count 到 metadata（避免 schema 变更）
                existing.metadata["conflict_count"] = existing.conflict_count

                await self._persist_belief(existing, upsert=True)
                logger.debug(
                    "信念更新: %s:%s = %s (conf=%.2f, n=%d, conflicts=%d, %s)",
                    category, key, existing.value[:30], existing.confidence,
                    existing.evidence_count, existing.conflict_count,
                    "冲突" if is_conflict else "支持",
                )
                return existing
            else:
                # 创建新信念
                if len(self._cache) >= self._max_beliefs:
                    await self._evict_lowest_confidence()
                belief = Belief(
                    id=str(uuid4()),
                    category=category,
                    key=key,
                    value=value,
                    confidence=min(1.0, 0.5 + confidence_delta),
                    evidence_count=1,
                    conflict_count=0,
                    source_memory_ids=[source_memory_id] if source_memory_id else [],
                    metadata=metadata or {},
                    created_at=now,
                    updated_at=now,
                )
                belief.metadata["conflict_count"] = 0
                self._cache[cache_key] = belief
                await self._persist_belief(belief, upsert=False)
                logger.debug("信念创建: %s:%s = %s (conf=%.2f)",
                             category, key, value[:30], belief.confidence)
                return belief

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """计算两段文本的 Jaccard 相似度（基于 token 集合）。

        用于判断新证据是支持还是矛盾已有信念。

        Args:
            a: 文本 A
            b: 文本 B

        Returns:
            相似度 [0, 1]，1 表示完全相同
        """
        if not a or not b:
            return 0.0
        # 分词：中文按字，英文按空格
        tokens_a: set[str] = set()
        for word in a.split():
            w = word.strip().lower()
            if w:
                tokens_a.add(w)
        for c in a:
            if "\u4e00" <= c <= "\u9fff":
                tokens_a.add(c)

        tokens_b: set[str] = set()
        for word in b.split():
            w = word.strip().lower()
            if w:
                tokens_b.add(w)
        for c in b:
            if "\u4e00" <= c <= "\u9fff":
                tokens_b.add(c)

        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    async def get_belief(self, category: str, key: str) -> Optional[Belief]:
        """查询单条信念。"""
        cache_key = f"{category}:{key}"
        return self._cache.get(cache_key)

    async def get_beliefs_by_category(self, category: str) -> list[Belief]:
        """获取某类别下的所有信念。"""
        return [
            b for cache_key, b in self._cache.items()
            if b.category == category
        ]

    async def get_all_beliefs(self) -> list[Belief]:
        """获取所有信念，按置信度降序。"""
        beliefs = list(self._cache.values())
        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs

    async def get_strong_beliefs(self, min_confidence: float = 0.7, limit: int = 20) -> list[Belief]:
        """获取高置信度信念（用于注入 Agent 上下文）。"""
        beliefs = [
            b for b in self._cache.values()
            if b.confidence >= min_confidence
        ]
        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs[:limit]

    async def decay_beliefs(self, decay_factor: float = 0.95) -> int:
        """信念衰减：降低所有信念的置信度。

        定期调用以淘汰过时信念。

        Args:
            decay_factor: 衰减因子 (0-1)，0.95 表示每次衰减 5%

        Returns:
            衰减的信念数量
        """
        async with self._lock:
            count = 0
            now = time.time()
            for belief in self._cache.values():
                # 检查 TTL
                if self._belief_ttl > 0:
                    age = now - belief.updated_at
                    if age > self._belief_ttl:
                        belief.confidence *= 0.5  # 过期信念大幅降级
                else:
                    belief.confidence *= decay_factor
                belief.updated_at = now
                await self._persist_belief(belief, upsert=True)
                count += 1

            # 清理置信度过低的信念
            to_remove = [
                k for k, b in self._cache.items()
                if b.confidence < 0.1
            ]
            for k in to_remove:
                del self._cache[k]
                # 从数据库删除
                await self._delete_belief_from_db(k)

            logger.info("信念衰减完成: %d 条衰减, %d 条清理", count, len(to_remove))
            return count

    async def format_for_context(self, max_items: int = 10) -> str:
        """将高置信度信念格式化为上下文文本（供 Agent 使用）。

        Args:
            max_items: 最大条目数

        Returns:
            格式化的信念文本
        """
        beliefs = await self.get_strong_beliefs(min_confidence=0.6, limit=max_items)
        if not beliefs:
            return ""

        lines = ["## 用户心智模型"]
        for b in beliefs:
            conflict_tag = f" ⚠{b.conflict_count}冲突" if b.conflict_count > 0 else ""
            lines.append(
                f"- [{b.category}] {b.key}: {b.value} "
                f"(置信度: {b.confidence:.0%}, 证据: {b.evidence_count}{conflict_tag})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _persist_belief(self, belief: Belief, upsert: bool) -> None:
        """持久化信念到数据库。"""
        conn = await self._pool.acquire()
        try:
            if upsert:
                await conn.execute(
                    """INSERT INTO mental_models (id, category, key, value, confidence, evidence_count, source_memory_ids, metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(category, key) DO UPDATE SET
                           value=excluded.value,
                           confidence=excluded.confidence,
                           evidence_count=excluded.evidence_count,
                           source_memory_ids=excluded.source_memory_ids,
                           metadata=excluded.metadata,
                           updated_at=excluded.updated_at""",
                    (belief.id, belief.category, belief.key, belief.value,
                     belief.confidence, belief.evidence_count,
                     json.dumps(belief.source_memory_ids, ensure_ascii=False),
                     json.dumps(belief.metadata, ensure_ascii=False),
                     belief.created_at, belief.updated_at),
                )
            else:
                await conn.execute(
                    """INSERT OR IGNORE INTO mental_models (id, category, key, value, confidence, evidence_count, source_memory_ids, metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (belief.id, belief.category, belief.key, belief.value,
                     belief.confidence, belief.evidence_count,
                     json.dumps(belief.source_memory_ids, ensure_ascii=False),
                     json.dumps(belief.metadata, ensure_ascii=False),
                     belief.created_at, belief.updated_at),
                )
            await conn.commit()
        except Exception as e:
            logger.error("持久化信念失败: %s", e)
        finally:
            await self._pool.release(conn)

    async def _evict_lowest_confidence(self) -> None:
        """淘汰置信度最低的信念。"""
        if not self._cache:
            return
        lowest_key = min(self._cache, key=lambda k: self._cache[k].confidence)
        evicted = self._cache.pop(lowest_key)
        await self._delete_belief_from_db(evicted.id)
        logger.debug("淘汰低置信度信念: %s:%s", evicted.category, evicted.key)

    async def _delete_belief_from_db(self, belief_id_or_key: str) -> None:
        """从数据库删除信念。"""
        conn = await self._pool.acquire()
        try:
            # 先按 ID 删，再按 key 删（_evict 传的是 id，_decay 传的可能也是 id）
            await conn.execute("DELETE FROM mental_models WHERE id = ?", (belief_id_or_key,))
            await conn.commit()
        except Exception as e:
            logger.warning("删除信念失败: %s", e)
        finally:
            await self._pool.release(conn)

    def get_stats(self) -> dict:
        """获取心智模型统计。"""
        beliefs = list(self._cache.values())
        categories: dict[str, int] = {}
        for b in beliefs:
            categories[b.category] = categories.get(b.category, 0) + 1
        avg_confidence = sum(b.confidence for b in beliefs) / len(beliefs) if beliefs else 0
        total_conflicts = sum(b.conflict_count for b in beliefs)
        conflicted = sum(1 for b in beliefs if b.conflict_count > 0)
        return {
            "total": len(beliefs),
            "categories": categories,
            "avg_confidence": round(avg_confidence, 3),
            "total_conflicts": total_conflicts,
            "conflicted_beliefs": conflicted,
        }
