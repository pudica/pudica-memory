"""pipeline/l3_search.py — L3：多策略检索 + 上下文压缩。

参考 TencentDB l3-search.ts 的搜索策略和 hindsight retrieval.py 的 TEMPR 架构。
文档 6.1 节 L3Search 实现。
"""

import logging
from typing import Any, Optional

from unified_memory.search.fusion import TEMPREngine
from unified_memory.search.types import FusionResult

logger = logging.getLogger(__name__)


class L3Search:
    """L3：多策略检索 + 上下文压缩。

    4 路并行检索 + RRF 融合，token budget 控制。
    参考 TencentDB l3-search.ts 的搜索策略和 hindsight TEMPR 架构。
    """

    def __init__(
        self,
        temp_engine: TEMPREngine,
        default_budget: int = 4000,
    ):
        """
        Args:
            temp_engine: TEMPREngine 实例
            default_budget: 默认 token budget
        """
        self._engine = temp_engine
        self._default_budget = default_budget

    async def search(
        self,
        query: str,
        budget: int = 0,
        top_k: int = 20,
        time_range: Optional[tuple[float, float]] = None,
    ) -> list[FusionResult]:
        """4 路并行检索 + RRF 融合，token budget 控制。

        Args:
            query: 查询文本
            budget: Token budget（0 = 使用默认 4000）
            top_k: 每路检索的 top_k
            time_range: 时间检索范围

        Returns:
            融合后的检索结果列表
        """
        budget = budget or self._default_budget

        # 4 路并行检索 + RRF 融合
        results = await self._engine.search(query, top_k=top_k, time_range=time_range)

        # Token budget 裁剪
        results = self._trim_to_budget(results, budget)

        return results

    async def recall(
        self,
        query: str,
        budget: int = 4000,
        top_k: int = 20,
    ) -> list[FusionResult]:
        """Agent 优化检索：带 token budget 的上下文检索。

        Args:
            query: 查询文本
            budget: Token budget
            top_k: 每路检索的 top_k

        Returns:
            检索结果列表
        """
        return await self.search(query, budget=budget, top_k=top_k)

    def _trim_to_budget(
        self, results: list[FusionResult], budget: int
    ) -> list[FusionResult]:
        """Token budget 裁剪。

        渐进式裁剪策略：
        - 轻度：替换可替代性高的条目
        - 中度：删除最旧条目
        - 紧急：截断超长消息

        Args:
            results: 融合后的结果列表
            budget: Token budget

        Returns:
            裁剪后的结果列表
        """
        # 简单估算：中文约 1.5 token/字，英文约 0.75 token/字符
        # 按 1 token/字 粗略估算
        total_tokens = sum(len(r.text) for r in results)

        if total_tokens <= budget:
            return results

        # 需要裁剪：从得分最低的开始删除
        trimmed = list(results)
        while trimmed and sum(len(r.text) for r in trimmed) > budget:
            # 移除得分最低的条目
            trimmed.pop()

        # 如果裁剪后为空，至少保留得分最高的
        if not trimmed and results:
            top = results[0]
            # 创建新的 FusionResult 而非原地修改，避免影响原始对象
            trimmed_text = top.text[:budget] + "..." if len(top.text) > budget else top.text
            trimmed = [FusionResult(
                id=top.id,
                text=trimmed_text,
                score=top.score,
                sources=top.sources.copy(),
                metadata=top.metadata.copy(),
            )]

        logger.debug("Token budget 裁剪: %d → %d 条 (%d tokens)",
                      len(results), len(trimmed), sum(len(r.text) for r in trimmed))
        return trimmed