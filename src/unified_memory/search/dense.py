"""search/dense.py — 语义检索（ChromaDB）。

基于 ChromaDB 的余弦相似度语义检索。
参考文档 7.3 节 SemanticRetriever 实现。
"""

from typing import Any, Optional

import asyncio

from unified_memory.search.types import ScoredResult


class SemanticRetriever:
    """基于 ChromaDB 的语义检索。

    通过 ChromaStore 的 search 方法进行向量相似度检索。
    """

    def __init__(self, chroma_store: Any):
        """
        Args:
            chroma_store: ChromaStore 实例
        """
        self._chroma = chroma_store

    async def search(
        self,
        query: str,
        top_k: int = 20,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[ScoredResult]:
        """语义检索，返回余弦相似度得分。

        Args:
            query: 查询文本
            top_k: 返回条数
            wing: 可选，按 wing 过滤
            room: 可选，按 room 过滤

        Returns:
            ScoredResult 列表，按得分降序排列
        """
        loop = asyncio.get_running_loop()
        from unified_memory.store.chroma_store import get_chroma_executor
        results = await loop.run_in_executor(
            get_chroma_executor(),
            lambda: self._chroma.search(query, n_results=top_k, wing=wing, room=room),
        )
        scored: list[ScoredResult] = []
        for i, r in enumerate(results):
            scored.append(ScoredResult(
                id=r["id"],
                text=r["content"],
                score=r["score"],
                source="semantic",
                rank=i,
                metadata=r.get("metadata", {}),
            ))
        return scored