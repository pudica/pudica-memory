"""store/kg.py — 轻量知识图谱，LRU 缓存 + SQLite 持久化。

支持实体、关系、三元组 CRUD，支持 get_neighbors() 和 get_similar_entities()。
参考文档 6.3 节 KnowledgeGraph 实现 + hindsight link_expansion_retrieval.py 的图链接扩展模式。

v3.1 升级：从全内存缓存改为 LRU 缓存 + SQLite 按需加载。
  - 实体使用 OrderedDict LRU 缓存，cache miss 时从 SQLite 按需加载
  - 轻量级 _all_entity_names 集合用于存在性检查（不占大量内存）
  - 关系使用 _relation_index 索引实现 O(1) 邻居查找（替代 O(n) 全扫描）
  - load_from_db() 只加载实体名称和关系索引，不加载完整 Entity 对象
  - max_entity_cache 参数控制缓存上限（默认 10000），防止 OOM
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# 弱关联谓词：co_occurs_with（L2 关联图自动填充的共现边）与 related_to
# （consolidation 合并边）都是自动生成的噪声关联，不是 LLM 语义关系。
# 在 kg_query 用户可见结果中过滤掉，但 get_neighbors() 仍需保留给图检索召回用。
WEAK_PREDICATES = frozenset({"co_occurs_with", "related_to"})


@dataclass
class Entity:
    """知识图谱实体。"""
    id: str
    name: str
    entity_type: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Relation:
    """知识图谱关系（三元组）。"""
    id: str
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    source: str = "auto"
    created_at: float = field(default_factory=time.time)


class KnowledgeGraph:
    """轻量知识图谱，LRU 缓存 + SQLite 持久化。

    v3.1 升级：实体使用 LRU 缓存，关系使用索引加速查找。
    支持大规模知识图谱（10万+实体）而不会 OOM。

    参考文档 6.3 节实现 + mempalace knowledge_graph.py 的实体关系模式。
    """

    def __init__(self, pool: Optional[Any] = None, max_entity_cache: int = 10000):
        """
        Args:
            pool: 可选，SQLite 连接池，用于持久化
            max_entity_cache: 实体 LRU 缓存上限，默认 10000。
                超过后自动淘汰最久未访问的实体。设为 0 表示无限制（兼容旧行为）。
        """
        self._pool = pool
        self._max_entity_cache = max_entity_cache
        # LRU 缓存：name → Entity（ OrderedDict 保持插入/访问顺序）
        self._entities: OrderedDict[str, Entity] = OrderedDict()
        # 轻量级名称集合：用于存在性检查，不存储完整 Entity 对象
        self._all_entity_names: set[str] = set()
        # 关系列表（全量加载，因为关系体积小且需要频繁遍历）
        self._relations: list[Relation] = []
        # 关系索引：entity_name → [relation_indices]，O(1) 查找替代 O(n) 扫描
        self._relation_index: dict[str, list[int]] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # 实体操作
    # ------------------------------------------------------------------

    async def _add_entity_unlocked(
        self, name: str, entity_type: str, metadata: Optional[dict] = None
    ) -> Entity:
        """添加实体（如果已存在则返回已有实体）。调用者必须已持有 self._lock。

        v3.1：使用 _all_entity_names 做存在性检查，LRU 缓存存储 Entity 对象。

        Args:
            name: 实体名称
            entity_type: 实体类型（person/org/location/product）
            metadata: 附加元数据

        Returns:
            实体对象
        """
        # 先检查名称集合（O(1)，不触发缓存加载）
        if name in self._all_entity_names:
            # 尝试从缓存获取
            entity = self._entities.get(name)
            if entity:
                # LRU：移动到末尾（最近使用）
                self._entities.move_to_end(name)
                return entity
            # 缓存 miss：从 DB 加载
            entity = await self._load_entity_from_db(name)
            if entity:
                self._cache_entity(name, entity)
                return entity
            # DB 中也没有（可能刚创建但未持久化），创建新的
        entity = Entity(
            id=str(uuid4()),
            name=name,
            entity_type=entity_type,
            metadata=metadata or {},
            created_at=time.time(),
        )
        self._all_entity_names.add(name)
        self._cache_entity(name, entity)
        if self._pool:
            await self._persist_entity(entity)
        return entity

    def _cache_entity(self, name: str, entity: Entity) -> None:
        """将实体放入 LRU 缓存，超限时淘汰最久未访问的。"""
        self._entities[name] = entity
        if self._max_entity_cache > 0 and len(self._entities) > self._max_entity_cache:
            # 淘汰最久未访问的（OrderedDict 第一个）
            evicted_name, _ = self._entities.popitem(last=False)
            logger.debug("KG LRU 淘汰实体: %s (cache=%d)", evicted_name, len(self._entities))

    async def _load_entity_from_db(self, name: str) -> Optional[Entity]:
        """从 SQLite 按需加载单个实体。"""
        if not self._pool:
            return None
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            )
            row = await cursor.fetchone()
            if row:
                return Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                )
            return None
        except Exception as e:
            logger.warning("从 DB 加载实体失败 %s: %s", name, e)
            return None
        finally:
            await self._pool.release(conn)

    async def add_entity(
        self, name: str, entity_type: str, metadata: Optional[dict] = None
    ) -> Entity:
        """添加实体（如果已存在则返回已有实体）。

        Args:
            name: 实体名称
            entity_type: 实体类型（person/org/location/product）
            metadata: 附加元数据

        Returns:
            实体对象
        """
        async with self._lock:
            return await self._add_entity_unlocked(name, entity_type, metadata)

    async def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体（LRU 缓存 + 按需加载）。

        v3.1：cache miss 时从 SQLite 按需加载，自动加入 LRU 缓存。

        Args:
            name: 实体名称

        Returns:
            实体对象，不存在则返回 None
        """
        async with self._lock:
            # 先查缓存
            entity = self._entities.get(name)
            if entity:
                self._entities.move_to_end(name)  # LRU 更新
                return entity
            # cache miss：检查名称集合
            if name not in self._all_entity_names:
                return None
            # 从 DB 加载
            entity = await self._load_entity_from_db(name)
            if entity:
                self._cache_entity(name, entity)
                return entity
            return None

    async def get_all_entities(self) -> list[str]:
        """获取所有实体名称列表（从轻量级名称集合获取，不触发缓存加载）。"""
        return list(self._all_entity_names)

    async def get_hot_entities(self, min_relations: int = 3, hours: int = 24) -> list[Entity]:
        """获取高频活跃实体。

        v3.1：使用 _relation_index 替代 O(n) 全扫描。

        Args:
            min_relations: 最小关系数
            hours: 时间窗口（小时）

        Returns:
            活跃实体列表
        """
        cutoff = time.time() - hours * 3600
        async with self._lock:
            # 使用关系索引统计每个实体的关系数
            relation_count: dict[str, int] = {}
            for name, indices in self._relation_index.items():
                count = sum(
                    1 for i in indices
                    if self._relations[i].created_at >= cutoff
                )
                if count > 0:
                    relation_count[name] = count

            # 筛选并加载实体
            result = []
            for name, count in relation_count.items():
                if count >= min_relations:
                    entity = self._entities.get(name)
                    if entity:
                        result.append(entity)
                    else:
                        # cache miss：从 DB 加载
                        loaded = await self._load_entity_from_db(name)
                        if loaded:
                            self._cache_entity(name, loaded)
                            result.append(loaded)

            result.sort(key=lambda e: relation_count.get(e.name, 0), reverse=True)
            return result

    async def get_entity_context(self, name: str) -> dict:
        """获取实体的完整上下文（属性 + 关系 + 邻居）。

        v3.1：使用关系索引 O(1) 查找，替代 O(n) 全扫描。

        Args:
            name: 实体名称

        Returns:
            {"entity": Entity, "relations": [Relation], "neighbors": [str]}
        """
        async with self._lock:
            # 获取实体（可能触发 cache miss 加载）
            entity = self._entities.get(name)
            if not entity and name in self._all_entity_names:
                entity = await self._load_entity_from_db(name)
                if entity:
                    self._cache_entity(name, entity)
            if not entity:
                return {"entity": None, "relations": [], "neighbors": []}

            # 使用关系索引查找
            indices = self._relation_index.get(name, [])
            relations = [self._relations[i] for i in indices]
            neighbors = set()
            for r in relations:
                if r.subject == name:
                    neighbors.add(r.object)
                if r.object == name:
                    neighbors.add(r.subject)

            return {
                "entity": entity,
                "relations": relations,
                "neighbors": list(neighbors),
            }

    # ------------------------------------------------------------------
    # 关系操作
    # ------------------------------------------------------------------

    async def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 1.0,
        source: str = "auto",
    ) -> Relation:
        """添加关系（三元组）。

        v3.1：维护 _relation_index 实现后续 O(1) 查找。

        Args:
            subject: 主体实体名称
            predicate: 关系谓词
            obj: 客体实体名称
            weight: 关系权重
            source: 来源

        Returns:
            关系对象
        """
        # 确保实体存在（在锁内创建，避免锁分裂导致重复关系）
        async with self._lock:
            if subject not in self._all_entity_names:
                await self._add_entity_unlocked(subject, "unknown")
            if obj not in self._all_entity_names:
                await self._add_entity_unlocked(obj, "unknown")

            rel = Relation(
                id=str(uuid4()),
                subject=subject,
                predicate=predicate,
                object=obj,
                weight=weight,
                source=source,
                created_at=time.time(),
            )
            rel_idx = len(self._relations)
            self._relations.append(rel)
            # 维护关系索引
            self._relation_index.setdefault(subject, []).append(rel_idx)
            self._relation_index.setdefault(obj, []).append(rel_idx)
            if self._pool:
                await self._persist_relation(rel)
            return rel

    async def add_relation_if_absent(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 1.0,
        source: str = "auto",
    ) -> Relation:
        """添加关系，若同 (subject, predicate, object) 已存在则不重复插入。

        关联图自动填充用：同一 ingest 的多个实体两两建边，避免重复三元组膨胀。
        已存在时权重累加（体现共现强度），返回已有关系对象。

        Args:
            subject: 主体实体名称
            predicate: 关系谓词
            obj: 客体实体名称
            weight: 新增权重（已存在时累加）
            source: 来源

        Returns:
            关系对象（新增或已存在的）
        """
        async with self._lock:
            for rel in self._relations:
                if (
                    rel.subject == subject
                    and rel.predicate == predicate
                    and rel.object == obj
                ):
                    # 已存在：累加权重体现共现强度
                    new_weight = rel.weight + weight
                    rel.weight = new_weight
                    if self._pool:
                        await self._persist_relation(rel)
                    return rel
            # 不存在：调用底层插入（使用 _add_entity_unlocked 需锁，这里已持有锁）
            if subject not in self._all_entity_names:
                await self._add_entity_unlocked(subject, "unknown")
            if obj not in self._all_entity_names:
                await self._add_entity_unlocked(obj, "unknown")
            rel = Relation(
                id=str(uuid4()),
                subject=subject,
                predicate=predicate,
                object=obj,
                weight=weight,
                source=source,
                created_at=time.time(),
            )
            rel_idx = len(self._relations)
            self._relations.append(rel)
            self._relation_index.setdefault(subject, []).append(rel_idx)
            self._relation_index.setdefault(obj, []).append(rel_idx)
            if self._pool:
                await self._persist_relation(rel)
            return rel

    # ------------------------------------------------------------------
    # P1: 事实冲突检测 + 合并
    # ------------------------------------------------------------------

    async def get_entity_type_conflicts(self) -> list[dict]:
        """检测同一实体在 DB 中存在多个不同 entity_type 的冲突。

        Returns:
            [{"entity": name, "types": [type1, type2, ...], "counts": {type: count}}, ...]
        """
        if not self._pool:
            return []
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                """SELECT name, entity_type, COUNT(*) as cnt
                   FROM entities
                   GROUP BY name, entity_type
                   HAVING cnt > 0
                   ORDER BY name"""
            )
            # 解析分组结果
            entity_types: dict[str, dict[str, int]] = {}
            rows = await cursor.fetchall()
            for row in rows:
                name = row["name"]
                etype = row["entity_type"]
                cnt = row["cnt"]
                if name not in entity_types:
                    entity_types[name] = {}
                entity_types[name][etype] = entity_types[name].get(etype, 0) + cnt

            conflicts = []
            for name, types in entity_types.items():
                if len(types) > 1:
                    conflicts.append({
                        "entity": name,
                        "types": list(types.keys()),
                        "counts": types,
                    })
            return conflicts
        finally:
            await self._pool.release(conn)

    async def merge_entity_conflicts(self, dry_run: bool = True) -> dict:
        """合并同一实体的多个 type 冲突（保留最高频 type）。

        Args:
            dry_run: True=只报告不执行，False=实际合并

        Returns:
            {"conflicts": N, "merged": N, "details": [...]}
        """
        conflicts = await self.get_entity_type_conflicts()
        if not conflicts:
            return {"conflicts": 0, "merged": 0, "details": []}

        merged = 0
        details = []
        for c in conflicts:
            # 选出现次数最多的 type
            best_type = max(c["counts"], key=c["counts"].get)
            old_types = [t for t in c["types"] if t != best_type]
            if not old_types:
                continue
            if not dry_run:
                async with self._lock:
                    # 更新缓存中已加载的实体 type
                    entity = self._entities.get(c["entity"])
                    if entity and entity.entity_type != best_type:
                        entity.entity_type = best_type
                    # 更新 DB 中所有冲突的 type 行
                    if self._pool:
                        conn = await self._pool.acquire()
                        try:
                            await conn.execute(
                                "UPDATE entities SET entity_type = ? WHERE name = ?",
                                (best_type, c["entity"]),
                            )
                            await conn.commit()
                        finally:
                            await self._pool.release(conn)
            merged += 1
            details.append({
                "entity": c["entity"],
                "from_types": old_types,
                "to_type": best_type,
            })

        return {
            "conflicts": len(conflicts),
            "merged": merged,
            "dry_run": dry_run,
            "details": details,
        }

    async def delete_entity(self, name: str) -> None:
        """删除实体及其所有关系（按名称）。

        同步删除 SQLite 中该实体行，以及所有以它为端点的关系行，
        并维护内存索引（_all_entity_names / _relations / _relation_index）。
        """
        async with self._lock:
            # 收集所有以该实体为端点的关系索引
            removed_indices = {
                idx for idx, rel in enumerate(self._relations)
                if rel.subject == name or rel.object == name
            }
            if removed_indices:
                # 重建关系列表与关系索引（跳过被删关系）
                new_relations: list[Relation] = []
                new_index: dict[str, list[int]] = {}
                for idx, rel in enumerate(self._relations):
                    if idx in removed_indices:
                        continue
                    new_idx = len(new_relations)
                    new_relations.append(rel)
                    new_index.setdefault(rel.subject, []).append(new_idx)
                    new_index.setdefault(rel.object, []).append(new_idx)
                self._relations = new_relations
                self._relation_index = new_index

            # 删除实体缓存与名称集合
            self._entities.pop(name, None)
            self._all_entity_names.discard(name)

            if self._pool:
                conn = await self._pool.acquire()
                try:
                    await conn.execute(
                        "DELETE FROM relations WHERE subject=? OR object=?",
                        (name, name),
                    )
                    await conn.execute("DELETE FROM entities WHERE name=?", (name,))
                    await conn.commit()
                except Exception as e:
                    logger.error("删除实体失败: %s", e)
                finally:
                    await self._pool.release(conn)

    async def delete_relation(self, subject: str, obj: str) -> None:
        """删除指定 (subject, object) 对的全部关系（不限谓词）。

        重提取场景用：先按端点对删掉旧关系，再写入新关系，避免重复三元组累积。
        """
        async with self._lock:
            removed_indices = {
                idx for idx, rel in enumerate(self._relations)
                if rel.subject == subject and rel.object == obj
            }
            if not removed_indices:
                return
            # 重建关系列表与关系索引（跳过被删关系）
            new_relations: list[Relation] = []
            new_index: dict[str, list[int]] = {}
            for idx, rel in enumerate(self._relations):
                if idx in removed_indices:
                    continue
                new_idx = len(new_relations)
                new_relations.append(rel)
                new_index.setdefault(rel.subject, []).append(new_idx)
                new_index.setdefault(rel.object, []).append(new_idx)
            self._relations = new_relations
            self._relation_index = new_index

            if self._pool:
                conn = await self._pool.acquire()
                try:
                    await conn.execute(
                        "DELETE FROM relations WHERE subject=? AND object=?",
                        (subject, obj),
                    )
                    await conn.commit()
                except Exception as e:
                    logger.error("删除关系失败: %s", e)
                finally:
                    await self._pool.release(conn)

    async def get_relations(
        self, subject: Optional[str] = None, predicate: Optional[str] = None
    ) -> list[Relation]:
        """查询关系。

        v3.1：有 subject 参数时使用关系索引 O(1) 查找。

        Args:
            subject: 可选，主体过滤
            predicate: 可选，谓词过滤

        Returns:
            匹配的关系列表
        """
        async with self._lock:
            if subject:
                # 使用索引
                indices = self._relation_index.get(subject, [])
                results = [self._relations[i] for i in indices]
            else:
                results = list(self._relations)
            if predicate:
                results = [r for r in results if r.predicate == predicate]
            return results

    # ------------------------------------------------------------------
    # 图遍历
    # ------------------------------------------------------------------

    def _get_neighbors_unlocked(self, entity_name: str) -> list[tuple[str, str]]:
        """获取实体的直接邻居及其关系类型（调用者持有锁）。

        v3.1：使用 _relation_index O(1) 查找，替代 O(n) 全扫描。
        """
        neighbors: list[tuple[str, str]] = []
        indices = self._relation_index.get(entity_name, [])
        for i in indices:
            rel = self._relations[i]
            if rel.subject == entity_name:
                neighbors.append((rel.object, rel.predicate))
            elif rel.object == entity_name:
                neighbors.append((rel.subject, rel.predicate))
        return neighbors

    async def get_neighbors(self, entity_name: str) -> list[tuple[str, str]]:
        """获取实体的直接邻居及其关系类型。

        Args:
            entity_name: 实体名称

        Returns:
            [(邻居名称, 关系谓词), ...] 列表
        """
        async with self._lock:
            return self._get_neighbors_unlocked(entity_name)

    def _get_similar_entities_unlocked(
        self, entity_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """获取语义相似的实体（基于共享关系数）。调用者持有锁。

        v3.1：使用 _relation_index 加速邻居查找，避免 O(n) 全扫描。
        """
        # 使用索引获取目标实体的邻居
        target_indices = self._relation_index.get(entity_name, [])
        target_neighbors = set()
        for i in target_indices:
            rel = self._relations[i]
            if rel.subject == entity_name:
                target_neighbors.add(rel.object)
            elif rel.object == entity_name:
                target_neighbors.add(rel.subject)

        if not target_neighbors:
            return []

        # 遍历所有有关系的实体（从索引 key 获取，而非全量实体）
        scores: dict[str, float] = {}
        for name in self._relation_index:
            if name == entity_name:
                continue
            # 使用索引获取该实体的邻居
            indices = self._relation_index.get(name, [])
            entity_neighbors = set()
            for i in indices:
                rel = self._relations[i]
                if rel.subject == name:
                    entity_neighbors.add(rel.object)
                elif rel.object == name:
                    entity_neighbors.add(rel.subject)
            if not entity_neighbors:
                continue
            intersection = target_neighbors & entity_neighbors
            union = target_neighbors | entity_neighbors
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard > 0:
                scores[name] = jaccard

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    async def query(self, entity: str) -> dict:
        """查询实体的完整知识图谱信息。

        Args:
            entity: 实体名称

        Returns:
            包含 entity 详情、邻居、相似实体的字典
        """
        entity_obj = await self.get_entity(entity)
        if entity_obj is None:
            return {"entity": entity, "found": False, "neighbors": [], "similar": []}

        neighbors = await self.get_neighbors(entity)
        similar = await self.get_similar_entities(entity, top_k=5)

        # 区分语义关系与弱关联：co_occurs_with / related_to 是自动生成的共现/合并边，
        # 不是 LLM 语义关系，不应淹没 kg_query 的语义结果。
        # 仅过滤用户可见输出；get_neighbors() 本身保留给图检索召回用。
        semantic_neighbors = []
        weak_neighbors = []
        for n, r in neighbors:
            item = {"name": n, "relation": r}
            if r in WEAK_PREDICATES:
                weak_neighbors.append(item)
            else:
                semantic_neighbors.append(item)

        return {
            "entity": entity,
            "found": True,
            "type": entity_obj.entity_type,
            "metadata": entity_obj.metadata,
            "neighbors": semantic_neighbors,
            "weak_neighbors": weak_neighbors,
            "similar": [{"name": s[0], "score": s[1]} for s in similar],
        }

    async def get_similar_entities(
        self, entity_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """获取语义相似的实体（基于共享关系数）。

        Args:
            entity_name: 实体名称
            top_k: 返回相似实体数

        Returns:
            [(实体名称, 相似度分数), ...]
        """
        async with self._lock:
            return self._get_similar_entities_unlocked(entity_name, top_k)

    async def expand_from_entities(
        self,
        entities: list[str],
        max_depth: int = 2,
        max_results: int = 50,
    ) -> list[tuple[str, float, str]]:
        """图链接扩展：从种子实体出发，沿实体共现/语义链接/因果链扩展。

        参考 hindsight link_expansion_retrieval.py 的三重链接扩展。

        Args:
            entities: 种子实体列表
            max_depth: 最大递归深度
            max_results: 最大返回结果数

        Returns:
            [(实体名称, 得分, 扩展类型)] 列表
        """
        expanded: list[tuple[str, float, str]] = []
        seen: set[str] = set(entities)

        async with self._lock:
            for entity in entities:
                # 实体共现：直接邻居
                for neighbor, rel_type in self._get_neighbors_unlocked(entity):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        expanded.append((neighbor, 0.8, "cooccurrence"))
                        # 因果链扩展
                        if rel_type in ("cause", "effect", "prevent", "enable"):
                            if max_depth > 1:
                                sub = self._expand_recursive_unlocked(
                                    [neighbor], max_depth - 1, seen
                                )
                                for e, s, t in sub:
                                    if e not in seen:
                                        seen.add(e)
                                        expanded.append((e, s * 0.7, "causal"))

                # 语义 kNN：相似实体
                for similar_entity, sim_score in self._get_similar_entities_unlocked(entity, top_k=5):
                    if similar_entity not in seen:
                        seen.add(similar_entity)
                        expanded.append((similar_entity, sim_score * 0.6, "semantic_knn"))

        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded[:max_results]

    def _expand_recursive_unlocked(
        self, entities: list[str], depth: int, seen: set[str]
    ) -> list[tuple[str, float, str]]:
        """递归因果链扩展（调用者持有锁）。"""
        results: list[tuple[str, float, str]] = []
        if depth <= 0:
            return results
        for entity in entities:
            for neighbor, rel_type in self._get_neighbors_unlocked(entity):
                if neighbor in seen:
                    continue
                if rel_type in ("cause", "effect", "prevent", "enable"):
                    seen.add(neighbor)
                    results.append((neighbor, 0.6 * depth / 2, "causal"))
                    if depth > 1:
                        results.extend(
                            self._expand_recursive_unlocked([neighbor], depth - 1, seen)
                        )
        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    async def _persist_entity(self, entity: Entity) -> None:
        """将实体持久化到 SQLite。"""
        if not self._pool:
            return
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                """INSERT INTO entities (id, name, entity_type, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       entity_type = excluded.entity_type,
                       metadata = excluded.metadata""",
                (entity.id, entity.name, entity.entity_type,
                 json.dumps(entity.metadata), entity.created_at),
            )
            await conn.commit()
        except Exception as e:
            logger.error("持久化实体失败: %s", e)
        finally:
            await self._pool.release(conn)

    async def _persist_relation(self, relation: Relation) -> None:
        """将关系持久化到 SQLite。"""
        if not self._pool:
            return
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                """INSERT INTO relations (id, subject, predicate, object, weight, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       subject = excluded.subject,
                       predicate = excluded.predicate,
                       object = excluded.object,
                       weight = excluded.weight,
                       source = excluded.source""",
                (relation.id, relation.subject, relation.predicate,
                 relation.object, relation.weight, relation.source, relation.created_at),
            )
            await conn.commit()
        except Exception as e:
            logger.error("持久化关系失败: %s", e)
        finally:
            await self._pool.release(conn)

    async def load_from_db(self) -> None:
        """从 SQLite 加载到内存。

        v3.1 升级：
        - 只加载实体名称到 _all_entity_names（轻量级，不占大量内存）
        - 完整 Entity 对象按需从 DB 加载（LRU 缓存）
        - 加载全部关系并构建 _relation_index 索引
        """
        if not self._pool:
            return
        if self._loaded:
            return
        conn = await self._pool.acquire()
        try:
            # 只加载实体名称（不加载完整 Entity 对象）
            cursor = await conn.execute("SELECT name FROM entities")
            rows = await cursor.fetchall()
            for row in rows:
                self._all_entity_names.add(row["name"])

            # 加载全部关系并构建索引
            cursor = await conn.execute("SELECT * FROM relations")
            rows = await cursor.fetchall()
            for row in rows:
                idx = len(self._relations)
                rel = Relation(
                    id=row["id"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    weight=row["weight"],
                    source=row["source"],
                    created_at=row["created_at"],
                )
                self._relations.append(rel)
                # 构建关系索引
                self._relation_index.setdefault(rel.subject, []).append(idx)
                self._relation_index.setdefault(rel.object, []).append(idx)
                # 确保实体名称在集合中
                self._all_entity_names.add(rel.subject)
                self._all_entity_names.add(rel.object)

            self._loaded = True
            logger.info(
                "KG 加载完成: %d 实体名称, %d 关系, %d 索引条目 (LRU cache=%d)",
                len(self._all_entity_names), len(self._relations),
                len(self._relation_index), self._max_entity_cache,
            )
        finally:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # 代码符号索引（v3.4.0：KG 代码符号索引）
    # 记录函数/类/模块调用关系，从 TencentDB Memory v2.0 的 Code-Graph 资产启发
    # ------------------------------------------------------------------

    CODE_SYMBOL_TYPES = frozenset({"function", "class", "module", "method", "variable", "api_endpoint"})

    async def add_code_symbol(
        self, name: str, symbol_type: str, file_path: str,
        metadata: Optional[dict] = None,
    ) -> Entity:
        """添加代码符号实体。

        Args:
            name: 符号名称（如 "ingest()", "UnifiedMemoryApp", "main.py"）
            symbol_type: 符号类型（function/class/module/method/variable/api_endpoint）
            file_path: 源文件路径
            metadata: 附加元数据（如行号、文档字符串摘要、参数列表等）

        Returns:
            实体对象
        """
        assert symbol_type in self.CODE_SYMBOL_TYPES, f"不支持的符号类型: {symbol_type}"
        m = metadata or {}
        m["kind"] = "code_symbol"
        m["symbol_type"] = symbol_type
        m["file_path"] = file_path
        entity = await self.add_entity(name, f"code_{symbol_type}", metadata=m)
        return entity

    async def add_code_call_relation(
        self, caller: str, callee: str, weight: float = 1.0,
    ) -> Relation:
        """添加代码调用关系（caller → calls → callee）。

        Args:
            caller: 调用方符号名
            callee: 被调用方符号名
            weight: 调用频次权重

        Returns:
            关系对象
        """
        return await self.add_relation_if_absent(caller, "calls", callee, weight=weight, source="code_index")

    async def add_code_contain_relation(
        self, container: str, contained: str, weight: float = 1.0,
    ) -> Relation:
        """添加代码包含关系（container → contains → contained）。

        Args:
            container: 容器符号名（如模块名、类名）
            contained: 被包含符号名（如类中的方法、模块中的函数）
            weight: 包含关系权重

        Returns:
            关系对象
        """
        return await self.add_relation_if_absent(container, "contains", contained, weight=weight, source="code_index")

    async def get_code_symbols_by_file(self, file_path: str) -> list[dict]:
        """获取某个文件的所有代码符号。

        Args:
            file_path: 源文件路径

        Returns:
            [{"name": ..., "type": ..., "metadata": ...}, ...]
        """
        # 从所有实体中筛选出 metadata.path == file_path 的代码符号
        async with self._lock:
            results = []
            for name in self._all_entity_names:
                entity = self._entities.get(name)
                if entity and entity.metadata.get("kind") == "code_symbol":
                    if entity.metadata.get("file_path") == file_path:
                        results.append({
                            "name": entity.name,
                            "type": entity.metadata.get("symbol_type", "unknown"),
                            "metadata": entity.metadata,
                        })
            return results

    async def get_code_call_graph(self, symbol_name: str, depth: int = 2) -> dict:
        """获取代码符号的调用图（调用者和被调用者）。

        Args:
            symbol_name: 符号名称
            depth: 递归深度

        Returns:
            {"symbol": ..., "callers": [...], "callees": [...], "call_graph": {...}}
        """
        entity = await self.get_entity(symbol_name)
        if not entity:
            return {"symbol": symbol_name, "found": False}

        # 获取所有关系
        neighbors = await self.get_neighbors(symbol_name)
        callers = []
        callees = []
        for n, r in neighbors:
            if r == "calls":
                callees.append({"name": n, "relation": "calls"})
            # calls 的反向：谁是调用者？需要通过关系索引查找
        # 从关系列表中反查调用者
        all_rels = await self.get_relations()
        for rel in all_rels:
            if rel.predicate == "calls" and rel.object == symbol_name:
                callers.append({"name": rel.subject, "relation": "calls"})

        return {
            "symbol": symbol_name,
            "found": True,
            "type": entity.entity_type,
            "file_path": entity.metadata.get("file_path", ""),
            "callers": callers,
            "callees": callees,
        }

    def get_stats(self) -> dict:
        """获取知识图谱统计信息。

        v3.1：实体计数从 _all_entity_names 获取（不依赖缓存）。
        v3.4.0：新增代码符号统计。

        Returns:
            {"entities": N, "relations": N, "entity_types": {...}, "cache_hit_ratio": float,
             "code_symbols": N, "code_relations": N}
        """
        # 实体类型统计需要从缓存获取（未缓存的不统计类型）
        # 实体类型统计需要从缓存获取（未缓存的不统计类型）
        type_counts: dict[str, int] = {}
        for entity in self._entities.values():
            t = entity.entity_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "entities": len(self._all_entity_names),
            "relations": len(self._relations),
            "entity_types": type_counts,
            "cached_entities": len(self._entities),
            "max_cache": self._max_entity_cache,
        }