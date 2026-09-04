"""search/fuzzy_matching.py — 模糊标签匹配与拼写纠错。

从 Hindsight 的 pg_trgm 实现移植（纯 Python 版），
支持 trigram 相似度计算和模糊标签匹配。

用法:
    matcher = FuzzyMatcher()
    # 对单个标签做模糊匹配
    match = matcher.fuzzy_match_tag("typ", ["type", "typo", "tap"])
    # 对标签列表做模糊搜索
    matches = matcher.search_tags("mempalce", ["mempalace", "memory", "palace"])
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# 相似度阈值：低于此值不算匹配
DEFAULT_SIMILARITY_THRESHOLD = 0.3


def _trigram_set(text: str) -> set[str]:
    """生成字符串的 trigram 集合。

    每个字符串前加两个空格做 padding，确保短字符串也能生成 trigram。
    示例: 'cat' -> ['  c', ' ca', 'cat', 'at ']
    """
    normalized = text.lower().strip()
    padded = f"  {normalized} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


@lru_cache(maxsize=4096)
def _cached_trigram_set(text: str) -> frozenset[str]:
    """缓存版本的 trigram 集合。"""
    return frozenset(_trigram_set(text))


def trigram_similarity(a: str, b: str) -> float:
    """计算两个字符串的 trigram 相似度。

    使用 Dice 系数: 2 * |intersection| / (|a| + |b|)
    返回 0.0 ~ 1.0 之间的浮点数。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    set_a = _cached_trigram_set(a)
    set_b = _cached_trigram_set(b)

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    return 2.0 * len(intersection) / (len(set_a) + len(set_b))


class FuzzyMatcher:
    """模糊标签匹配器。

    用 trigram 相似度在候选列表中找最匹配的标签。
    支持中文：中文按字符做 trigram（每个汉字是一个字符），
    对中文短标签效果较好。
    """

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    def fuzzy_match_tag(self, query: str, candidates: list[str]) -> Optional[str]:
        """在候选列表中找最匹配的标签。

        Args:
            query: 用户输入的标签（可能拼写错误）
            candidates: 候选标签列表

        Returns:
            最匹配的候选标签，如果都没有超过阈值则返回 None
        """
        if not query or not candidates:
            return None

        # 完全匹配优先
        query_lower = query.lower().strip()
        for c in candidates:
            if c.lower().strip() == query_lower:
                return c

        best_match = None
        best_score = self._threshold

        for c in candidates:
            score = trigram_similarity(query, c)
            if score > best_score:
                best_score = score
                best_match = c

        if best_match:
            logger.debug("fuzzy match '%s' -> '%s' (score=%.3f)", query, best_match, best_score)
        return best_match

    def search_tags(self, query: str, candidates: list[str],
                    top_k: int = 5) -> list[tuple[str, float]]:
        """对候选标签做模糊搜索，返回按相似度排序的结果。

        Args:
            query: 搜索词
            candidates: 候选标签列表
            top_k: 返回前 N 个结果

        Returns:
            [(标签, 相似度), ...] 按相似度降序排列
        """
        if not query or not candidates:
            return []

        scored = []
        for c in candidates:
            score = trigram_similarity(query, c)
            if score >= self._threshold:
                scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# 全局实例
_default_matcher = FuzzyMatcher()


def fuzzy_match_tag(query: str, candidates: list[str]) -> Optional[str]:
    """全局函数：模糊匹配标签。"""
    return _default_matcher.fuzzy_match_tag(query, candidates)


def fuzzy_search_tags(query: str, candidates: list[str],
                      top_k: int = 5) -> list[tuple[str, float]]:
    """全局函数：模糊搜索标签。"""
    return _default_matcher.search_tags(query, candidates, top_k)