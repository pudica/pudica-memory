"""search/sparse.py — BM25 全文检索（SQLite FTS5）。

基于 SQLite FTS5 的 BM25 全文检索，支持 wing/room 元数据过滤。
"""

import json
import re
from typing import Any, Optional

from unified_memory.search.types import ScoredResult


class BM25Retriever:
    """基于 SQLite FTS5 的 BM25 全文检索。

    使用 SQLite 内置的 FTS5 虚拟表进行全文搜索，返回 BM25 得分。
    FTS5 的 rank 是负值（越小越相关），转换为 0-1 得分。
    支持 wing/room 两级元数据过滤。
    """

    def __init__(self, pool: Any):
        """
        Args:
            pool: SQLitePool 实例
        """
        self._pool = pool

    async def search(
        self,
        query: str,
        top_k: int = 20,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[ScoredResult]:
        """FTS5 全文搜索，返回 BM25 得分。

        Args:
            query: 搜索关键词
            top_k: 返回条数
            wing: 可选，按 wing 过滤
            room: 可选，按 room 过滤

        Returns:
            ScoredResult 列表，按得分降序排列
        """
        # 对查询进行 FTS5 转义处理
        def escape_fts(text: str) -> str:
            """FTS5 字符串字面量转义。

            FTS5 的 MATCH 语法里 : ^ * " ( ) 等都是特殊字符。最稳妥的做法是把每个
            词条包成双引号字符串字面量，字面量内部唯一需要转义的是双引号本身，
            按 FTS5 规范用两个双引号表示。原实现的正则实际是个空操作（匹配非特殊
            字符再原样替换），遇到特殊字符反而会破坏查询。
            """
            return '"' + text.replace('"', '""') + '"'
        # 动态过滤阈值：中文字符 >= 2 即可（FTS5 trigram 可匹配 2-gram），英文 >= 3
        def _min_len(w: str) -> bool:
            stripped = w.strip()
            if not stripped:
                return False
            # 含中文字符的词条放宽到 2 字
            has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in stripped)
            return len(stripped) >= (2 if has_cjk else 3)

        fts_query = " OR ".join(escape_fts(w) for w in re.split(r"\s+", query) if _min_len(w))
        if not fts_query:
            return []

        # 构建过滤条件：FTS5 表(memories_fts)仅有 id/content/wing/room 四列
        # 没有 metadata 列。通过 JOIN memories 基表获取全部元数据，然后用 wing/room 过滤
        # 注意：FTS5 的 rank 是负值（越小越相关），memories 基表有 metadata（JSON 字符串）
        conditions: list[str] = []
        params: list[Any] = [fts_query]

        if wing is not None:
            conditions.append("m.wing = ?")
            params.append(wing)
        if room is not None:
            conditions.append("m.room = ?")
            params.append(room)

        where_clause = ""
        if conditions:
            where_clause = " AND " + " AND ".join(conditions)

        sql = f"""
            SELECT fts.id, fts.content, m.metadata, bm25(memories_fts) AS rank
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.id
            WHERE memories_fts MATCH ?{where_clause}
            ORDER BY rank
            LIMIT ?
        """
        params.append(top_k)

        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            scored: list[ScoredResult] = []
            for i, row in enumerate(rows):
                # FTS5 rank 是负值（越小越相关），转换为 0-1 得分
                bm25_score = max(0, 1.0 / (1.0 + abs(row["rank"])))
                metadata_raw = row["metadata"]
                if isinstance(metadata_raw, str):
                    try:
                        metadata_raw = json.loads(metadata_raw)
                    except (json.JSONDecodeError, TypeError):
                        metadata_raw = {}
                elif metadata_raw is None:
                    metadata_raw = {}
                scored.append(ScoredResult(
                    id=row["id"],
                    text=row["content"],
                    score=bm25_score,
                    source="bm25",
                    rank=i,
                    metadata=dict(metadata_raw),
                ))
            return scored
        finally:
            await self._pool.release(conn)