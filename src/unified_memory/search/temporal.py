"""search/temporal.py — 时间检索。

时间范围过滤 + 时间衰减加权。
参考文档 7.3 节 TemporalRetriever 实现。
"""

import json
import math
import time
from typing import Any, Optional

from unified_memory.search.types import ScoredResult


class TemporalRetriever:
    """时间检索：时间范围过滤 + 时间衰减加权。

    近期信息权重更高，使用指数衰减函数：exp(-decay_rate * age_days)。
    """

    def __init__(self, pool: Any):
        """
        Args:
            pool: SQLitePool 实例
        """
        self._pool = pool

    async def search(
        self,
        query: str = "",
        top_k: int = 20,
        decay_rate: float = 0.1,
        time_range: Optional[tuple[float, float]] = None,
    ) -> list[ScoredResult]:
        """时间检索：时间范围过滤 + 时间衰减加权。

        Args:
            query: 查询文本（当前未使用，保留接口兼容性）
            top_k: 返回条数
            decay_rate: 衰减率，默认 0.1
            time_range: 可选，时间范围过滤 (start_ts, end_ts)

        Returns:
            ScoredResult 列表
        """
        if time_range is None:
            time_range = (time.time() - 7 * 86400, time.time())

        now = time.time()
        conn = await self._pool.acquire()
        try:
            sql = """
                SELECT id, content, created_at, metadata
                FROM memories
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            cursor = await conn.execute(sql, (time_range[0], time_range[1], top_k * 2))
            rows = await cursor.fetchall()

            scored: list[ScoredResult] = []
            for i, row in enumerate(rows):
                age_days = (now - row["created_at"]) / 86400
                # 时间衰减权重：exp(-decay_rate * age_days)
                temporal_score = math.exp(-decay_rate * max(0, age_days))

                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        meta = {}

                scored.append(ScoredResult(
                    id=row["id"],
                    text=row["content"],
                    score=temporal_score,
                    source="temporal",
                    rank=i,
                    metadata={
                        "created_at": row["created_at"],
                        "age_days": round(age_days, 2),
                        **meta,
                    },
                ))

            scored.sort(key=lambda r: r.score, reverse=True)
            for i, r in enumerate(scored[:top_k]):
                r.rank = i
            return scored[:top_k]
        finally:
            await self._pool.release(conn)