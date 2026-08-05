"""tasks/consolidation.py — Consolidation 任务：四策略 + 证据链。

参考文档 7.4 节 Consolidator 实现 + hindsight reflect/agent.py 的 consolidation flow。
四种策略：dedup / merge / link / upgrade，全部带证据链（evidence chain）。
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from unified_memory.store.kg import Relation

logger = logging.getLogger(__name__)


class Consolidator:
    """Consolidation 任务：四策略 + 证据链。

    参考文档 7.4 节和历史 hindsight reflect/agent.py 的 consolidation 流程。

    四种策略：
    1. dedup: 删除重复实体（基于相似度）
    2. merge: 合并相似实体
    3. link: 为实体添加跨场景链接
    4. upgrade: 升级实体类型（由 AI 推测转确认为 knowledge）

    所有操作带证据链（evidence chain），保留 provenance。
    """

    def __init__(
        self,
        kg: Any,
        chroma: Any,
        pool: Any,
        settings: Optional[dict] = None,
    ):
        """
        Args:
            kg: KnowledgeGraph 实例
            chroma: ChromaStore 实例
            pool: SQLitePool 实例
            settings: 可选配置（similarity_threshold, batch_size 等）
        """
        self._kg = kg
        self._chroma = chroma
        self._pool = pool
        self._settings = settings or {}
        self._sim_threshold = self._settings.get("similarity_threshold", 0.85)
        self._batch_size = self._settings.get("batch_size", 10)

    async def consolidate(self) -> dict:
        """执行一次完整的 Consolidation 流程。

        Returns:
            统计结果字典，包含各策略的操作数和证据链
        """
        logger.info("开始 Consolidation 任务")

        dedup_result = await self._dedup()
        merge_result = await self._merge()
        link_result = await self._link()
        upgrade_result = await self._upgrade()

        result = {
            "dedup": dedup_result,
            "merge": merge_result,
            "link": link_result,
            "upgrade": upgrade_result,
            "timestamp": time.time(),
        }

        # 记录
        await self._log_consolidation(result)

        total = dedup_result["count"] + merge_result["count"] + link_result["count"] + upgrade_result["count"]
        logger.info("Consolidation 完成: 共 %d 操作 (dedup=%d, merge=%d, link=%d, upgrade=%d)",
                     total, dedup_result["count"], merge_result["count"],
                     link_result["count"], upgrade_result["count"])
        return result

    async def _dedup(self) -> dict:
        """策略一：去重 — 删除重复实体。

        基于实体名称的相似度，删除相似度超过阈值的实体。

        Returns:
            {"count": N, "deleted": [str], "evidence": [...]}
        """
        deleted: list[str] = []
        evidence: list[dict] = []

        all_entities = await self._kg.get_all_entities()

        # 预批量加载所有实体上下文（一次锁获取，避免 O(n²) 次锁获取）
        entity_contexts: dict[str, dict] = {}
        for e in all_entities:
            entity_contexts[e] = await self._kg.get_entity_context(e)

        # 计算两两相似度
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                e1, e2 = all_entities[i], all_entities[j]
                if e1 in deleted or e2 in deleted:
                    continue
                sim = self._name_similarity(e1, e2)
                if sim >= self._sim_threshold:
                    e1_ctx = entity_contexts.get(e1, {"relations": []})
                    e2_ctx = entity_contexts.get(e2, {"relations": []})
                    if len(e1_ctx["relations"]) >= len(e2_ctx["relations"]):
                        keep, remove = e1, e2
                    else:
                        keep, remove = e2, e1
                    # 实际删除：将 remove 的关系重定向到 keep
                    await self._reassign_relations(remove, keep)
                    deleted.append(remove)
                    evidence.append({
                        "type": "dedup",
                        "entity_a": keep,
                        "entity_b": remove,
                        "similarity": round(sim, 3),
                        "reason": f"名称相似度 {sim:.1%}，保留 {keep}，删除 {remove}",
                    })

        return {"count": len(deleted), "deleted": deleted, "evidence": evidence}

    async def _reassign_relations(self, from_entity: str, to_entity: str) -> None:
        """将 from_entity 的所有关系重新指向 to_entity，然后删除 from_entity。"""
        # 第一步：先持久化到 DB（成功后才更新内存，避免不一致）
        if self._pool:
            conn = await self._pool.acquire()
            try:
                # 删除自引用关系（避免 FK 冲突）
                await conn.execute(
                    "DELETE FROM relations WHERE subject = ? AND object = ?",
                    (to_entity, to_entity),
                )
                await conn.execute(
                    "UPDATE relations SET subject = ? WHERE subject = ?",
                    (to_entity, from_entity),
                )
                await conn.execute(
                    "UPDATE relations SET object = ? WHERE object = ?",
                    (to_entity, from_entity),
                )
                await conn.execute("DELETE FROM entities WHERE name = ?", (from_entity,))
                await conn.commit()
            except Exception as e:
                logger.warning("持久化关系重定向失败: %s", e)
                return  # DB 失败则不更新内存，保持一致性
            finally:
                await self._pool.release(conn)
        # 第二步：DB 成功后更新内存（创建新 Relation 对象，避免突变共享对象）
        async with self._kg._lock:
            new_relations = []
            for rel in self._kg._relations:
                if rel.subject == from_entity or rel.object == from_entity:
                    # 创建新的 Relation 对象替换旧的，避免共享状态突变
                    new_rel = Relation(
                        id=rel.id,
                        subject=to_entity if rel.subject == from_entity else rel.subject,
                        predicate=rel.predicate,
                        object=to_entity if rel.object == from_entity else rel.object,
                        weight=rel.weight,
                        source=rel.source,
                        created_at=rel.created_at,
                    )
                    new_relations.append(new_rel)
                else:
                    new_relations.append(rel)
            self._kg._relations = new_relations
            # v3.1：重建关系索引
            self._kg._relation_index.clear()
            for i, rel in enumerate(new_relations):
                self._kg._relation_index.setdefault(rel.subject, []).append(i)
                self._kg._relation_index.setdefault(rel.object, []).append(i)
            # 从内存和名称集合删除实体
            self._kg._entities.pop(from_entity, None)
            self._kg._all_entity_names.discard(from_entity)
            self._kg._relation_index.pop(from_entity, None)
    async def _merge(self) -> dict:
        """策略二：合并 — 将相似实体合并。

        合并时保留所有关系，内容合并，证据链记录 provenance。

        Returns:
            {"count": N, "merges": [{"from": str, "into": str, "relations": N}], "evidence": [...]}
        """
        merges: list[dict] = []
        evidence: list[dict] = []

        all_entities = await self._kg.get_all_entities()
        merged: set[str] = set()

        # 预批量加载所有实体上下文
        entity_contexts: dict[str, dict] = {}
        for e in all_entities:
            entity_contexts[e] = await self._kg.get_entity_context(e)

        for i in range(len(all_entities)):
            if all_entities[i] in merged:
                continue
            for j in range(i + 1, len(all_entities)):
                if all_entities[j] in merged:
                    continue
                e1, e2 = all_entities[i], all_entities[j]
                sim = self._name_similarity(e1, e2)
                if 0.7 <= sim < self._sim_threshold:
                    # 实际合并：将 e2 的关系重定向到 e1
                    await self._reassign_relations(e2, e1)
                    merges.append({
                        "from": e2,
                        "into": e1,
                        "relations": 0,  # 已重定向
                    })
                    evidence.append({
                        "type": "merge",
                        "from": e2,
                        "into": e1,
                        "similarity": round(sim, 3),
                        "reason": f"实体相似度 {sim:.1%}，合并 {e2} → {e1}",
                        "provenance": {"target_source": "auto", "confidence": sim},
                    })
                    merged.add(e2)

        return {"count": len(merges), "merges": merges, "evidence": evidence}

    async def _link(self) -> dict:
        """策略三：链接 — 为实体添加跨场景链接。

        基于实体在多场景中的共现关系，自动添加链接。

        Returns:
            {"count": N, "links": [{"subject": str, "predicate": str, "object": str}], "evidence": [...]}
        """
        links: list[dict] = []
        evidence: list[dict] = []

        all_entities = await self._kg.get_all_entities()
        # 预批量加载所有实体邻居（一次锁获取，避免 O(n²) 次锁获取）
        entity_neighbors: dict[str, set[str]] = {}
        for e in all_entities:
            neighbors = await self._kg.get_neighbors(e)
            entity_neighbors[e] = set(n for n, _ in neighbors)

        for i in range(len(all_entities)):
            e1 = all_entities[i]
            # 获取 e1 的邻居
            e1_neighbors = entity_neighbors.get(e1, set())
            for j in range(i + 1, len(all_entities)):
                e2 = all_entities[j]
                if e2 in e1_neighbors:
                    continue
                # 检查是否有共同邻居
                e2_neighbors = entity_neighbors.get(e2, set())
                common = e1_neighbors & e2_neighbors
                if len(common) >= 2:
                    # 有共同邻居，添加隐式关联并写入知识图谱
                    await self._kg.add_relation(e1, "related_to", e2, source="consolidation")
                    links.append({
                        "subject": e1,
                        "predicate": "related_to",
                        "object": e2,
                    })
                    evidence.append({
                        "type": "link",
                        "subject": e1,
                        "predicate": "related_to",
                        "object": e2,
                        "common_neighbors": list(common),
                        "reason": f"共享 {len(common)} 个共同邻居",
                    })

        return {"count": len(links), "links": links, "evidence": evidence}

    async def _upgrade(self) -> dict:
        """策略四：升级 — 实体类型升级。

        实体类型从 AI 推测 (observation) → 确认 (knowledge)，
        当实体出现在多个场景且有稳定关系链时。

        Returns:
            {"count": N, "upgrades": [{"entity": str, "from_type": str, "to_type": str}], "evidence": [...]}
        """
        upgrades: list[dict] = []
        evidence: list[dict] = []

        # 在 KG 锁内获取实体名称快照，避免迭代时并发修改
        async with self._kg._lock:
            entity_names = list(self._kg._all_entity_names)

        for entity_name in entity_names:
            entity = await self._kg.get_entity(entity_name)
            if not entity:
                continue
            context = await self._kg.get_entity_context(entity_name)
            # 升级条件：至少 3 条关系且出现在多个场景
            if len(context["relations"]) >= 3:
                # 实际修改内存中的实体类型
                old_type = entity.entity_type
                entity.entity_type = "knowledge"
                # 持久化到数据库
                if self._pool:
                    conn = None
                    try:
                        conn = await self._pool.acquire()
                        await conn.execute(
                            "UPDATE entities SET entity_type = ? WHERE name = ?",
                            ("knowledge", entity_name),
                        )
                        await conn.commit()
                    except Exception as e:
                        logger.warning("持久化实体类型升级失败: %s", e)
                        entity.entity_type = old_type  # 回滚
                    finally:
                        if conn is not None:
                            await self._pool.release(conn)
                upgrades.append({
                    "entity": entity_name,
                    "from_type": old_type,
                    "to_type": "knowledge",
                })
                evidence.append({
                    "type": "upgrade",
                    "entity": entity_name,
                    "from_type": old_type,
                    "to_type": "knowledge",
                    "relations_count": len(context["relations"]),
                    "reason": f"实体出现在 {len(context['relations'])} 条关系中，可升级为 knowledge",
                })

        return {"count": len(upgrades), "upgrades": upgrades, "evidence": evidence}

    def _name_similarity(self, a: str, b: str) -> float:
        """计算两个实体名称的相似度。

        基于 Jaccard 字符集相似度：交集字符数 / 并集字符数。
        相比"字符出现比例"，Jaccard 是有效的相似度度量，
        "云南" vs "南云" 不再等于 1.0。

        Args:
            a: 实体名称 A
            b: 实体名称 B

        Returns:
            0-1 之间的相似度分数
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        set_a = set(a)
        set_b = set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    async def _log_consolidation(self, result: dict) -> None:
        """记录 consolidation 结果到日志表。"""
        conn = await self._pool.acquire()
        try:
            # 自动建表（如果不存在）
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_logs (
                    id TEXT PRIMARY KEY,
                    result TEXT,
                    created_at REAL
                )
            """)
            await conn.execute(
                "INSERT INTO consolidation_logs (id, result, created_at) VALUES (?, ?, ?)",
                (str(uuid4()), json.dumps(result, ensure_ascii=False), time.time()),
            )
            await conn.commit()
        except Exception as e:
            logger.warning("写入 consolidation 日志失败: %s", e)
        finally:
            await self._pool.release(conn)