"""pipeline/l1_extractor.py — L1：本地规则提取（关键词实体 + 时间戳摘要）。

默认纯本地运行，不需要 LLM。LLM 作为可选增强模式。
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 默认关键词实体提取规则（中文命名实体关键词）
DEFAULT_ENTITY_PATTERNS: list[tuple[str, str]] = [
    # 人名模式：中文姓名（2-4字中文）
    (r"[\u4e00-\u9fff]{2,4}(?:医生|大夫|老师|先生|女士|同志)", "person"),
    # 组织模式：带"局/公司/医院/集团"等后缀
    (r"[\u4e00-\u9fff]{2,}(?:局|公司|医院|集团|部|委|办|院|中心|社|协会|基金会)", "org"),
    # 地点模式：带"省/市/县/区/镇/乡/村/路/街"等后缀
    (r"[\u4e00-\u9fff]{2,}(?:省|市|县|区|镇|乡|村|路|街|大道|桥|山|河|湖)", "location"),
    # 产品模式：带"系统/平台/软件/APP/工具/API"等后缀
    (r"[\u4e00-\u9fff]+(?:系统|平台|软件|APP|工具|API|协议|模型|框架|方案)", "product"),
    # 英文人名（首字母大写 + 空格）
    (r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "person"),
    # 英文组织名（全大写缩写）
    (r"[A-Z]{2,}(?:-[A-Z]+)*", "org"),
]

# 事实类型关键词（Hindsight 4 类结构化记忆）
FACT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "observation": ["观察", "发现", "看到", "注意到", "感觉", "觉得", "现象", "情况", "状态"],
    "experience": ["经历", "做过", "尝试", "体验", "用过", "试过", "实施", "执行", "完成"],
    "world": ["是", "属于", "位于", "包括", "包含", "有", "定义", "指", "代表", "构成"],
    "opinion": ["认为", "建议", "推荐", "应该", "值得", "好", "不好", "不错", "偏好", "倾向"],
}

# 事实类型置信度权重（Hindsight: 带置信度的事实提取）
FACT_TYPE_CONFIDENCE: dict[str, float] = {
    "world": 0.9,       # 世界知识，最可靠
    "observation": 0.7,  # 观察到的事实
    "experience": 0.8,   # 个人经历，较可靠
    "opinion": 0.5,      # 主观意见，置信度较低
}


class L1Extractor:
    """L1：本地规则提取器（关键词实体 + 时间戳摘要）。

    Args:
        llm: 可选的 LLM 客户端，提供后自动启用 LLM 增强模式
    """

    def __init__(self, llm: Any = None):
        self._llm = llm
        self._max_retries = 2

    async def extract(self, messages: list[str]) -> dict:
        """提取关键信息。

        优先使用 LLM 增强（如果配置了），否则走纯本地规则。

        Args:
            messages: 消息文本列表

        Returns:
            提取结果字典，包含 entities, relations, summary, fact_type 等字段
        """
        if self._llm is not None:
            try:
                return await self._llm_extract(messages)
            except Exception as e:
                logger.warning("LLM 提取失败，降级到本地规则: %s", e)

        return self._local_extract(messages)

    async def _llm_extract(self, messages: list[str]) -> dict:
        """LLM 增强提取。"""
        prompt = _build_llm_prompt(messages)
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm.call(
                    prompt,
                    response_format={"type": "json_object"},
                )
                import json
                result = json.loads(response)
                result.setdefault("entities", [])
                result.setdefault("relations", [])
                result.setdefault("summary", "")
                result.setdefault("time_range", {})
                result.setdefault("fact_type", "observation")
                result.setdefault("confidence", 0.7)
                return result
            except Exception as e:
                last_error = e
                logger.warning("L1 LLM 提取失败 (尝试 %d/%d): %s",
                               attempt + 1, self._max_retries, e)
                if attempt < self._max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1)

        logger.error("LLM 提取全部失败: %s", last_error)
        raise last_error  # 让上层降级走本地规则

    def _local_extract(self, messages: list[str]) -> dict:
        """纯本地规则提取：关键词实体 + 时间戳摘要。

        Returns:
            提取结果字典
        """
        combined = "\n".join(messages)

        # 1. 关键词实体提取
        entities = self._extract_entities(combined)

        # 2. 事实类型判断
        fact_type = self._classify_fact_type(combined)

        # 3. 摘要（截取前 200 字，取完整句子）
        summary = self._make_summary(combined)

        # 4. 时间戳
        now = datetime.now(timezone.utc)
        time_range = {
            "start": now.isoformat(),
            "end": now.isoformat(),
        }

        return {
            "entities": entities,
            "relations": [],  # 本地规则不提取关系
            "summary": summary,
            "time_range": time_range,
            "fact_type": fact_type,
            "confidence": self._compute_confidence(fact_type, len(entities), len(combined)),
        }

    def _extract_entities(self, text: str) -> list[dict]:
        """关键词实体提取。

        Returns:
            [{"name": "...", "type": "person|org|location|product"}, ...]
        """
        seen: set[str] = set()
        entities: list[dict] = []

        for pattern, ent_type in DEFAULT_ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group().strip()
                if len(name) < 2:
                    continue
                name = self._clean_entity_name(name, ent_type)
                if len(name) < 2:
                    continue
                if name not in seen:
                    seen.add(name)
                    entities.append({"name": name, "type": ent_type})

        # 去重后取前 20 个
        return entities[:20]

    # 实体名清洗：剥离常见非实体前缀（用户/正在/评估等），并按后缀词定位裁剪，
    # 避免把整句（如"用户参与了云南天海科技有限公司"）当作实体名。
    _NOISE_PREFIXES: tuple[str, ...] = (
        "用户", "我们", "我", "正在", "已经", "开始", "继续", "需要", "可以",
        "希望", "评估", "测试", "进行", "使用", "通过", "完成", "负责",
        "参与了", "参加了", "参与", "参加", "加入", "关于", "对于", "因为",
        "所以", "然后", "但是", "以及", "还有", "包括", "根据", "针对",
        "表示", "认为", "建议", "推荐", "喜欢", "正在使用", "正在学习",
        "讨论", "商量", "今天", "明天", "昨天", "去", "来", "开会",
        "开会讨论", "发布", "推出", "上线", "合作", "签署", "达成", "宣布",
        "准备", "打算", "计划", "考虑", "看", "看看", "最近", "前", "后",
        "学习", "研究", "开发", "建设", "管理", "运营", "服务", "提供",
        "支持", "实现", "打造", "构建", "推动", "促进", "加强", "与",
        "和", "及", "向", "为", "给", "对", "把", "从", "在", "跟", "同",
        "还有", "也是", "就是", "作为", "成为", "属于", "位于",
    )
    # 各实体类型的后缀词（避免跨类型串扰，如 person 的"老师"误裁剪 location 实体）
    _TYPE_SUFFIXES: dict[str, tuple[str, ...]] = {
        "person": ("医生", "大夫", "老师", "先生", "女士", "同志"),
        "org": ("有限公司", "研究院", "基金会", "中心", "公司", "集团", "医院",
                "大学", "学院", "协会", "局", "部", "委", "办", "院", "社"),
        "location": ("省", "市", "县", "区", "镇", "乡", "村", "路", "街", "大道",
                     "桥", "山", "河", "湖"),
        "product": ("系统", "平台", "软件", "工具", "协议", "模型", "框架",
                    "方案", "APP", "API"),
    }
    _SUFFIX_WORDS: tuple[str, ...] = tuple(
        sorted({s for v in _TYPE_SUFFIXES.values() for s in v}, key=len, reverse=True)
    )

    def _clean_entity_name(self, name: str, ent_type: str = "org") -> str:
        """剥离常见非实体前缀 + 按实体类型后缀词裁剪，保留紧凑实体名。"""
        # 1. 循环剥离非实体前缀（"用户参与了X公司" -> "X公司"）
        changed = True
        while changed and len(name) > 2:
            changed = False
            for p in self._NOISE_PREFIXES:
                if name.startswith(p) and len(name) > len(p) + 1:
                    name = name[len(p):]
                    changed = True
                    break
        # 2. 按类型后缀定位：只保留最后一个后缀词及其前最多 8 个汉字
        suffixes = self._TYPE_SUFFIXES.get(ent_type, self._SUFFIX_WORDS)
        for suffix in sorted(suffixes, key=len, reverse=True):
            idx = name.rfind(suffix)
            if idx > 0:
                head = name[max(0, idx - 8):idx]
                # 实体名通常从"的/在"等分隔词之后开始（如"云岭集团的量子计算平台"、
                # "王老师在云南省昆明市"），截断到最后一个分隔符之后
                for sep in ("的", "在"):
                    if sep in head:
                        head = head.split(sep)[-1]
                cleaned = head + suffix
                # 若裁剪后反而更短则接受，否则保留原名（避免过度裁剪）
                if len(cleaned) < len(name):
                    name = cleaned
                break
        return name.strip()

    def _classify_fact_type(self, text: str) -> str:
        """基于关键词判断事实类型。"""
        scores = {ft: 0 for ft in FACT_TYPE_KEYWORDS}
        for ft, keywords in FACT_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[ft] += 1
        if not any(scores.values()):
            return "observation"
        return max(scores, key=scores.get)

    def _make_summary(self, text: str, max_chars: int = 200) -> str:
        """截取前 max_chars 字，取完整句子。"""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # 找最后一个句号/问号/感叹号/换行
        last_punct = max(
            truncated.rfind("。"),
            truncated.rfind("？"),
            truncated.rfind("！"),
            truncated.rfind("\n"),
        )
        if last_punct > max_chars // 2:
            return truncated[:last_punct + 1]
        return truncated

    def _compute_confidence(self, fact_type: str, entity_count: int, text_length: int) -> float:
        """计算提取结果的置信度（Hindsight: 带置信度的事实提取）。

        综合考量：
        - 事实类型的基础置信度
        - 提取到的实体数量（越多越可靠）
        - 文本长度（过短可能信息不足）

        Args:
            fact_type: 事实类型
            entity_count: 提取到的实体数量
            text_length: 原始文本长度

        Returns:
            0-1 之间的置信度
        """
        base = FACT_TYPE_CONFIDENCE.get(fact_type, 0.5)
        # 实体数量加成（最多 +0.1）
        entity_bonus = min(0.1, entity_count * 0.02)
        # 文本长度加成（50-500 字为最佳区间）
        if 50 <= text_length <= 500:
            length_bonus = 0.05
        elif text_length < 20:
            length_bonus = -0.1  # 过短惩罚
        else:
            length_bonus = 0.0
        return max(0.1, min(1.0, base + entity_bonus + length_bonus))


def _build_llm_prompt(messages: list[str]) -> str:
    """构建 LLM 提取提示词。"""
    import json
    return f"""从以下对话中提取关键信息，返回 JSON 格式：

对话内容：
{json.dumps(messages, ensure_ascii=False)}

要求：
1. entities: 提取所有命名实体（人、组织、地点、产品）
2. relations: 实体之间的关系（如：工作于、位于、属于）
3. time_range: 事件发生的时间范围（ISO 8601）
4. summary: 150字以内的摘要，保留关键事实
5. fact_type: 事实类型，必须是以下之一：
   - "observation": 观察到的事实/现象
   - "experience": 个人经历/做过的事
   - "world": 世界知识/客观事实
   - "opinion": 主观观点/偏好/建议
6. confidence: 提取结果的置信度 (0.0-1.0)，基于信息明确程度

输出格式：
```json
{{
  "entities": [{{"name": "...", "type": "person|org|location|product"}}],
  "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}],
  "time_range": {{"start": "ISO8601", "end": "ISO8601"}},
  "summary": "...",
  "fact_type": "observation",
  "confidence": 0.8
}}
```"""