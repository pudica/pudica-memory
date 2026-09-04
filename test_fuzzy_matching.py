"""测试 search/fuzzy_matching.py 的 trigram 相似度和模糊匹配。"""

import sys
import os

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unified_memory.search.fuzzy_matching import (
    trigram_similarity,
    FuzzyMatcher,
    fuzzy_match_tag,
    fuzzy_search_tags,
)


def test_trigram_similarity():
    """测试 trigram 相似度计算。"""
    # 完全一致
    assert trigram_similarity("hello", "hello") == 1.0
    assert trigram_similarity("", "") == 0.0
    assert trigram_similarity("a", "") == 0.0

    # 大小写不敏感
    assert trigram_similarity("Hello", "hello") == 1.0

    # 相似但不相同
    sim = trigram_similarity("mempalace", "mempalace")
    assert sim == 1.0, f"mempalace==mempalace should be 1.0, got {sim}"

    sim = trigram_similarity("mempalce", "mempalace")
    assert sim > 0.5, f"mempalce~mempalace should be >0.5, got {sim}"

    sim = trigram_similarity("cat", "car")
    assert sim > 0.0, f"cat~car should be >0.0, got {sim}"

    # 完全不相似
    sim = trigram_similarity("abc", "xyz")
    assert sim == 0.0, f"abc~xyz should be 0.0, got {sim}"

    # 中文
    sim = trigram_similarity("中医", "中医")
    assert sim == 1.0, f"中医==中医 should be 1.0, got {sim}"
    print("  PASS test_trigram_similarity")


def test_fuzzy_match_tag():
    """测试模糊匹配标签。"""
    matcher = FuzzyMatcher(threshold=0.3)
    candidates = ["mempalace", "memory", "palace", "wing", "drawer"]

    # 完全匹配
    assert matcher.fuzzy_match_tag("mempalace", candidates) == "mempalace"
    assert matcher.fuzzy_match_tag("MEMORY", candidates) == "memory"

    # 近似匹配
    match = matcher.fuzzy_match_tag("mempalce", candidates)
    assert match == "mempalace", f"mempalce should match mempalace, got {match}"

    # 提高阈值，匹配变严格
    matcher.threshold = 0.9
    assert matcher.fuzzy_match_tag("mempalce", candidates) is None

    # 降低阈值
    matcher.threshold = 0.1
    assert matcher.fuzzy_match_tag("mempalce", candidates) is not None

    # 空输入
    assert matcher.fuzzy_match_tag("", ["a", "b"]) is None
    assert matcher.fuzzy_match_tag("a", []) is None

    # 中文
    matcher.threshold = 0.3
    candidates_zh = ["中医", "中药", "针灸", "脉诊"]
    assert matcher.fuzzy_match_tag("中医", candidates_zh) == "中医"
    match = matcher.fuzzy_match_tag("中Y", candidates_zh)
    assert match == "中医", f"中Y should match 中医, got {match}"
    print("  PASS test_fuzzy_match_tag")


def test_fuzzy_search_tags():
    """测试模糊搜索标签（返回排序结果）。"""
    matcher = FuzzyMatcher(threshold=0.3)
    candidates = ["mempalace", "memory", "palace", "wing", "drawer"]

    results = matcher.search_tags("mempalce", candidates)
    assert len(results) >= 1, f"should have at least 1 result, got {len(results)}"
    # 最匹配的应该是 mempalace
    assert results[0][0] == "mempalace", f"top match should be mempalace, got {results[0][0]}"

    # top_k 限制
    results = matcher.search_tags("mempalce", candidates, top_k=1)
    assert len(results) == 1, f"top_k=1 should return 1 result, got {len(results)}"

    # 无匹配
    results = matcher.search_tags("zzzzz", candidates)
    assert len(results) == 0, f"should have 0 matches, got {len(results)}"

    print("  PASS test_fuzzy_search_tags")


def test_global_functions():
    """测试全局函数版。"""
    candidates = ["mempalace", "memory", "palace"]

    match = fuzzy_match_tag("mempalce", candidates)
    assert match == "mempalace", f"global: mempalce should match mempalace, got {match}"

    results = fuzzy_search_tags("mempalce", candidates, top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "mempalace"

    print("  PASS test_global_functions")


if __name__ == "__main__":
    print("Running fuzzy_matching tests...")
    test_trigram_similarity()
    test_fuzzy_match_tag()
    test_fuzzy_search_tags()
    test_global_functions()
    print("ALL TESTS PASSED")