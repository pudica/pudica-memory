"""search/fusion.py — RRF 融合 + 4 路并行检索引擎 + 自适应查询分类。

参考文档 7.3 节实现：
- RRFusion: Reciprocal Rank Fusion，公式 score(d) = sum(1/(k + rank(d)))
- interleave: 交替取各策略的 top 结果（备选方案）
- QueryClassifier: 查询分类器，根据查询特征动态调整检索路权重
- TEMPREngine: 4 路并行检索 + 融合

参考 hindsight `fusion.py` 的 reciprocal_rank_fusion 实现。

v3.1 升级：新增 QueryClassifier，根据查询文本特征动态调整 RRF 权重。
  - 时间查询（"上周"/"昨天"/"2024年"）→ 提权 temporal，降权 graph
  - 实体查询（短词/专有名词）→ 提权 graph，降权 temporal
  - 关键词查询（短文本/词组）→ 提权 bm25，降权 semantic
  - 语义查询（长句/自然语言）→ 提权 semantic，降权 bm25
"""

import asyncio
import re
from collections import defaultdict
from typing import Any, Optional

from unified_memory.search.types import ScoredResult, FusionResult


class RRFusion:
    """Reciprocal Rank Fusion 融合器。

    公式：score(d) = sum(1 / (k + rank(d)))
    参考 hindsight `fusion.py` 的 reciprocal_rank_fusion() 实现。
    """

    def __init__(self, k: int = 30, weights: Optional[dict[str, float]] = None):
        """
        Args:
            k: RRF 常数，默认 30（小数据集区分度更好）
            weights: 各策略权重，默认 semantic=1.0, bm25=1.0, graph=0.8, temporal=0.6
        """
        self._k = k
        self._weights = weights or {
            "semantic": 1.0,
            "bm25": 1.0,
            "graph": 0.8,
            "temporal": 0.6,
        }

    def fuse(
        self, results: dict, top_k: int = 20,
        weights: Optional[dict[str, float]] = None,
    ) -> list[FusionResult]:
        """RRF 融合 + 分数归一化。

        k=30（小数据集区分度更好），结果 top_k 截断 + 分数归一化到 [0,1]。

        Args:
            results: {策略名: [ScoredResult, ...]} 字典
            top_k: 返回条数，默认 20
            weights: 可选权重覆盖（Bug fix: 用于自适应权重，不修改共享状态避免并发问题）

        Returns:
            融合后按 RRF 得分降序排列的 FusionResult 列表
        """
        effective_weights = weights if weights is not None else self._weights
        accumulator: dict[str, dict] = defaultdict(lambda: {
            "score": 0.0,
            "texts": [],
            "sources": [],
            "metadata": {},
        })

        for strategy, scored_list in results.items():
            weight = effective_weights.get(strategy, 1.0)
            for sr in scored_list:
                doc_id = sr.id
                rank = sr.rank + 1  # 1-indexed
                rrf_score = weight * (1.0 / (self._k + rank))
                accumulator[doc_id]["score"] += rrf_score
                if sr.text not in accumulator[doc_id]["texts"]:
                    accumulator[doc_id]["texts"].append(sr.text or "")
                if strategy not in accumulator[doc_id]["sources"]:
                    accumulator[doc_id]["sources"].append(strategy)
                if sr.metadata:
                    accumulator[doc_id]["metadata"].update(sr.metadata)

        sorted_items = sorted(
            accumulator.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]

        # 分数归一化到 [0, 1]
        max_score = max(info["score"] for _, info in sorted_items) if sorted_items else 1.0
        if max_score <= 0:
            max_score = 1.0

        return [
            FusionResult(
                id=doc_id,
                text="\n".join(info["texts"]),
                score=info["score"] / max_score if max_score > 0 else 0.0,
                sources=info["sources"],
                metadata=info["metadata"],
            )
            for doc_id, info in sorted_items
        ]

    def interleave(
        self, results: dict[str, list[ScoredResult]], top_k: int = 20
    ) -> list[FusionResult]:
        """Interleave 融合（备选策略）：交替取各策略的 top 结果。

        当 RRF 区分度不足时使用，参考 hindsight fusion.py 的 interleave 模式。

        Args:
            results: {策略名: [ScoredResult, ...]} 字典
            top_k: 返回条数

        Returns:
            融合后的 FusionResult 列表
        """
        seen: set[str] = set()
        fused: list[FusionResult] = []

        strategy_names = sorted(results.keys())
        max_len = max(len(v) for v in results.values())

        for i in range(max_len):
            for name in strategy_names:
                lst = results.get(name, [])
                if i < len(lst):
                    sr = lst[i]
                    if sr.id not in seen:
                        seen.add(sr.id)
                        lst_len = len(lst)
                        fused.append(FusionResult(
                            id=sr.id,
                            text=sr.text,
                            score=1.0 - (i / lst_len) if lst_len > 0 else 1.0,
                            sources=[name],
                            metadata=sr.metadata,
                        ))
                    if len(fused) >= top_k:
                        return fused
        return fused


class QueryClassifier:
    """查询分类器：根据查询文本特征动态调整 RRF 权重。

    v3.1 新增。检测查询类型并返回调整后的权重字典。

    分类规则：
    1. temporal（时间查询）：包含时间词（上周/昨天/2024年/last week 等）
    2. entity（实体查询）：短文本（≤8字）且不含时间词，或包含专有名词模式
    3. keyword（关键词查询）：短文本（≤15字/词），词组形式，无完整句式
    4. semantic（语义查询）：长句、自然语言、包含疑问词

    权重调整策略（在默认权重基础上浮动）：
    - temporal: temporal ×1.6, graph ×0.7
    - entity:   graph ×1.5, temporal ×0.7
    - keyword:  bm25 ×1.5, semantic ×0.7
    - semantic: semantic ×1.3, bm25 ×0.8
    """

    # 时间词模式
    _TEMPORAL_PATTERNS = [
        r'上周|上个月|昨天|前天|今天|明天|上周|下周|最近|刚才|之前|以前',
        r'去年|前年|今年|明年|\d{4}年|\d{1,2}月|\d{1,2}日',
        r'last\s+week|yesterday|tomorrow|today|ago|recent',
        r'\d+\s*(day|week|month|year)s?\s+ago',
    ]

    # 疑问词模式（语义查询特征）
    _QUESTION_PATTERNS = [
        r'为什么|怎么|如何|什么是|是什么|请问|能不能|可以吗|是不是|对不对',
        r'why|how|what|when|where|who|can you|is it|are there',
    ]

    # 专有名词模式（实体查询特征）
    _ENTITY_PATTERNS = [
        r'[\u4e00-\u9fff]{2,4}(先生|女士|博士|教授|医生|老师)',  # 人名
        r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # 英文人名
    ]

    def __init__(self, base_weights: Optional[dict[str, float]] = None):
        """
        Args:
            base_weights: 基础权重，默认 semantic=1.0, bm25=1.0, graph=0.8, temporal=0.6
        """
        self._base_weights = base_weights or {
            "semantic": 1.0,
            "bm25": 1.0,
            "graph": 0.8,
            "temporal": 0.6,
        }
        self._compiled_temporal = [re.compile(p, re.IGNORECASE) for p in self._TEMPORAL_PATTERNS]
        self._compiled_question = [re.compile(p, re.IGNORECASE) for p in self._QUESTION_PATTERNS]
        self._compiled_entity = [re.compile(p) for p in self._ENTITY_PATTERNS]

    def classify(self, query: str) -> str:
        """分类查询类型。

        Args:
            query: 查询文本

        Returns:
            "temporal" | "entity" | "keyword" | "semantic"
        """
        query = query.strip()
        if not query:
            return "semantic"

        # 1. 检测时间查询
        for pattern in self._compiled_temporal:
            if pattern.search(query):
                return "temporal"

        # 2. 计算文本长度（中文按字，英文按词）
        chinese_len = sum(1 for c in query if "\u4e00" <= c <= "\u9fff")
        english_words = len([w for w in query.split() if w.strip()])
        effective_len = chinese_len + english_words

        # 3. 检测关键词查询：短文本 + 无疑问词 + 无完整句式
        # Bug fix: 先检测关键词再检测实体。原来短文本（≤8字符）一律判为 entity，
        # 导致 "Python教程"、"内存管理" 等关键词查询被错误路由到 graph 路。
        has_question = any(p.search(query) for p in self._compiled_question)

        # Bug fix: 对 9..15 字符区间的查询也检查实体模式，避免 "张三教授"、
        # "Dr. Smith" 等中短实体名被误判为 keyword。
        if 9 <= effective_len <= 15 and not has_question:
            for pattern in self._compiled_entity:
                if pattern.search(query):
                    return "entity"

        if effective_len <= 15 and not has_question:
            return "keyword"

        # 4. 检测实体查询：短文本 + 专有名词模式
        if effective_len <= 8:
            for pattern in self._compiled_entity:
                if pattern.search(query):
                    return "entity"
            # 其他短文本默认为关键词
            if not has_question:
                return "keyword"

        # 5. 默认：语义查询
        return "semantic"

    def get_adaptive_weights(self, query: str) -> dict[str, float]:
        """根据查询类型返回调整后的权重。

        Args:
            query: 查询文本

        Returns:
            调整后的权重字典
        """
        qtype = self.classify(query)
        weights = dict(self._base_weights)  # 复制基础权重

        if qtype == "temporal":
            weights["temporal"] *= 1.6
            weights["graph"] *= 0.7
        elif qtype == "entity":
            weights["graph"] *= 1.5
            weights["temporal"] *= 0.7
        elif qtype == "keyword":
            weights["bm25"] *= 1.5
            weights["semantic"] *= 0.7
        elif qtype == "semantic":
            weights["semantic"] *= 1.3
            weights["bm25"] *= 0.8

        return weights

    def get_classification_info(self, query: str) -> dict:
        """获取分类详情（用于日志和调试）。

        Args:
            query: 查询文本

        Returns:
            {"type": str, "weights": dict}
        """
        qtype = self.classify(query)
        weights = self.get_adaptive_weights(query)
        return {"type": qtype, "weights": weights}


class TEMPREngine:
    """TEMPR 多策略并行检索引擎。

    4 路并行检索：语义 + BM25 + 图链接 + 时间 → RRF 融合。
    """

    def __init__(
        self,
        semantic: Any,
        bm25: Any,
        graph: Any,
        temporal: Any,
        fusion: RRFusion,
        adaptive: bool = True,
    ):
        """
        Args:
            semantic: SemanticRetriever 实例
            bm25: BM25Retriever 实例
            graph: GraphRetriever 实例
            temporal: TemporalRetriever 实例
            fusion: RRFusion 实例
            adaptive: 是否启用自适应检索（v3.1），默认 True。
                启用后根据查询类型动态调整 RRF 权重。
        """
        self._semantic = semantic
        self._bm25 = bm25
        self._graph = graph
        self._temporal = temporal
        self._fusion = fusion
        self._adaptive = adaptive
        self._classifier = QueryClassifier(fusion._weights) if adaptive else None

    async def search(
        self,
        query: str,
        top_k: int = 20,
        time_range: Optional[tuple[float, float]] = None,
        fusion_mode: str = "rrf",
    ) -> list[FusionResult]:
        """4 路并行检索 + RRF 融合（支持自适应权重）。

        v3.1 升级：启用 adaptive 时，根据查询类型动态调整 RRF 权重。
        - 时间查询 → 提权 temporal 路
        - 实体查询 → 提权 graph 路
        - 关键词查询 → 提权 bm25 路
        - 语义查询 → 提权 semantic 路

        Args:
            query: 查询文本
            top_k: 每路检索的 top_k
            time_range: 时间检索的时间范围
            fusion_mode: 融合模式 — "rrf" 或 "interleave"

        Returns:
            融合后的检索结果列表
        """
        # 4 路并行检索
        semantic_task = self._semantic.search(query, top_k)
        bm25_task = self._bm25.search(query, top_k)
        graph_task = self._graph.search(query, top_k)
        temporal_task = self._temporal.search(query, top_k, time_range=time_range)

        semantic_results, bm25_results, graph_results, temporal_results = await asyncio.gather(
            semantic_task, bm25_task, graph_task, temporal_task,
            return_exceptions=True,
        )

        # 处理异常（某路失败不影响其他路）
        results: dict[str, list[ScoredResult]] = {}
        if not isinstance(semantic_results, Exception):
            results["semantic"] = semantic_results
        if not isinstance(bm25_results, Exception):
            results["bm25"] = bm25_results
        if not isinstance(graph_results, Exception):
            results["graph"] = graph_results
        if not isinstance(temporal_results, Exception):
            results["temporal"] = temporal_results

        if not results:
            return []

        # 自适应权重调整（Bug fix: 通过 fuse() 参数传递权重，不修改共享 self._fusion._weights）
        adaptive_weights = None
        if self._adaptive and self._classifier and fusion_mode == "rrf":
            adaptive_weights = self._classifier.get_adaptive_weights(query)

            cls_info = self._classifier.get_classification_info(query)
            import logging
            logging.getLogger(__name__).debug(
                "自适应检索: query='%s' → type=%s, weights=%s",
                query[:50], cls_info["type"],
                {k: round(v, 2) for k, v in adaptive_weights.items()},
            )

        # 融合
        if fusion_mode == "interleave":
            result = self._fusion.interleave(results, top_k=top_k)
        else:
            result = self._fusion.fuse(results, top_k=top_k, weights=adaptive_weights)

        return result