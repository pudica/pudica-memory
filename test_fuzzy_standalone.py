"""测试 fuzzy_matching（纯函数，无依赖）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "unified_memory", "search"))

# 直接导入 modules，绕过 __init__.py 的依赖链
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fuzzy_matching",
    os.path.join(os.path.dirname(__file__), "src", "unified_memory", "search", "fuzzy_matching.py")
)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

# 测试
assert fm.trigram_similarity("hello", "hello") == 1.0
assert fm.trigram_similarity("", "") == 0.0
assert fm.trigram_similarity("mempalace", "mempalace") == 1.0
sim = fm.trigram_similarity("mempalce", "mempalace")
assert sim > 0.5, f"mempalce~mempalace should be >0.5, got {sim}"
sim = fm.trigram_similarity("abc", "xyz")
assert sim == 0.0, f"abc~xyz should be 0.0, got {sim}"
sim = fm.trigram_similarity("中医", "中医")
assert sim == 1.0, f"中医==中医 should be 1.0, got {sim}"
sim = fm.trigram_similarity("中医", "中药")
assert sim > 0.0, f"中医~中药 should be >0.0, got {sim}"

matcher = fm.FuzzyMatcher(threshold=0.3)
candidates = ["mempalace", "memory", "palace", "wing", "drawer"]
assert matcher.fuzzy_match_tag("mempalace", candidates) == "mempalace"
assert matcher.fuzzy_match_tag("MEMORY", candidates) == "memory"
match = matcher.fuzzy_match_tag("mempalce", candidates)
assert match == "mempalace", f"mempalce should match mempalace, got {match}"

# 提高阈值
matcher.threshold = 0.9
assert matcher.fuzzy_match_tag("mempalce", candidates) is None
matcher.threshold = 0.3

# 空输入
assert matcher.fuzzy_match_tag("", ["a"]) is None
assert matcher.fuzzy_match_tag("a", []) is None

# 中文
candidates_zh = ["中医", "中药", "针灸", "脉诊"]
assert matcher.fuzzy_match_tag("中医", candidates_zh) == "中医"
match = matcher.fuzzy_match_tag("中Y", candidates_zh)
assert match == "中医", f"中Y should match 中医, got {match}"

# fuzzy_search_tags
results = matcher.search_tags("mempalce", candidates)
assert len(results) >= 1, f"should have >=1 result, got {len(results)}"
assert results[0][0] == "mempalace", f"top: {results[0][0]}"

results = matcher.search_tags("mempalce", candidates, top_k=1)
assert len(results) == 1, f"top_k=1 should return 1, got {len(results)}"

results = matcher.search_tags("zzzzz", candidates)
assert len(results) == 0, f"should have 0 matches, got {len(results)}"

# 全局函数
match = fm.fuzzy_match_tag("mempalce", candidates)
assert match == "mempalace", f"global: {match}"
results = fm.fuzzy_search_tags("mempalce", candidates, top_k=2)
assert len(results) >= 1

print("ALL TESTS PASSED")