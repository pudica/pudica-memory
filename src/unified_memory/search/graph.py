"""search/graph.py — 图链接扩展检索。

基于内存知识图谱的三重链接扩展：
1. 实体共现 — 直接邻居
2. 语义 kNN — 相似实体
3. 因果链 — cause/effect 关系链

参考 hindsight `link_expansion_retrieval.py` 实现（文档 7.3 节）。
"""

from typing import Any, Optional

from unified_memory.search.types import ScoredResult


class GraphRetriever:
    """基于内存知识图谱的链接扩展检索。

    参考 hindsight `link_expansion_retrieval.py` 的三重链接扩展：
    1. 实体共现 — 直接邻居
    2. 语义 kNN — 相似实体
    3. 因果链 — cause/effect 关系链
    """

    def __init__(self, kg: Any):
        """
        Args:
            kg: KnowledgeGraph 实例
        """
        self._kg = kg
        self._entity_pattern_cache: Optional[tuple[frozenset, Any]] = None  # (entity_set, compiled_pattern)

    async def search(
        self, query: str, top_k: int = 20
    ) -> list[ScoredResult]:
        """图链接扩展检索。

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            ScoredResult 列表
        """
        # 1. 从查询中提取实体
        entities = await self._extract_entities(query)
        if not entities:
            return []

        # 2. 三重链接扩展
        expanded = await self._kg.expand_from_entities(entities, max_depth=2, max_results=top_k)

        # 3. 转换为 ScoredResult
        results: list[ScoredResult] = []
        for entity_id, score, exp_type in expanded:
            # 获取实体上下文
            context = await self._kg.get_entity_context(entity_id)
            content = ""
            if context["entity"]:
                content = f"实体: {context['entity'].name} (类型: {context['entity'].entity_type})"
                if context["relations"]:
                    rel_summary = "; ".join(
                        f"{r.subject} {r.predicate} {r.object}"
                        for r in context["relations"][:3]
                    )
                    content += f"\n关系: {rel_summary}"
                if context["neighbors"]:
                    content += f"\n邻居: {', '.join(context['neighbors'][:5])}"

            if content:
                type_weight = {
                    "causal": 0.9,
                    "semantic_knn": 0.7,
                    "cooccurrence": 0.5,
                }.get(exp_type, 0.5)
                results.append(ScoredResult(
                    id=f"graph_{entity_id}",
                    text=content,
                    score=score * type_weight,
                    source="graph",
                    metadata={
                        "entity_id": entity_id,
                        "expansion_type": exp_type,
                    },
                ))

        # 按得分排序，截取 top_k
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results[:top_k]):
            r.rank = i
        return results[:top_k]

    async def _extract_entities(self, query: str) -> list[str]:
        """从查询中提取实体名。

        使用单个编译好的正则交替模式匹配所有实体，避免 O(n) 次正则编译。
        缓存编译结果，仅在实体集合变化时重新编译。

        Args:
            query: 查询文本

        Returns:
            匹配的实体名称列表
        """
        import re
        all_entities = await self._kg.get_all_entities()
        if not all_entities:
            return []

        # 检查缓存是否仍然有效
        entity_set = frozenset(all_entities)
        if self._entity_pattern_cache is None or self._entity_pattern_cache[0] != entity_set:
            # 按长度降序排列，确保最长实体优先匹配
            sorted_entities = sorted(all_entities, key=len, reverse=True)
            pattern_str = "|".join(re.escape(e) for e in sorted_entities)
            compiled = re.compile(pattern_str, re.IGNORECASE)
            self._entity_pattern_cache = (entity_set, compiled)

        compiled = self._entity_pattern_cache[1]
        query_lower = query.lower()
        # 用字典做 O(1) 查找，避免 O(n) 遍历 all_entities
        entity_lower_map: dict[str, str] = {e.lower(): e for e in all_entities}
        entities = []
        for match in compiled.finditer(query_lower):
            matched_text = match.group()
            original = entity_lower_map.get(matched_text)
            if original and original not in entities:
                entities.append(original)
        return entities