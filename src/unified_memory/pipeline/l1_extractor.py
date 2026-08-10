"""pipeline/l1_extractor.py — L1：本地规则提取（关键词实体 + 时间戳摘要）。

默认纯本地运行，不需要 LLM。LLM 作为可选增强模式。
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ========================================================================
# 中文姓氏前缀（覆盖常见单姓+复姓，用于人名识别）
# MemPalace zh-CN entity detection 提供的 100+ 姓氏
# ========================================================================
_CHINESE_SURNAMES: tuple[str, ...] = (
    "王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "胡", "朱", "郭", "何", "高", "林", "罗", "郑",
    "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹", "彭",
    "曾", "萧", "田", "董", "袁", "潘", "于", "蒋", "蔡", "余",
    "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈", "姚",
    "卢", "姜", "崔", "钟", "谭", "陆", "汪", "范", "金", "石",
    "廖", "贾", "夏", "韦", "方", "白", "邹", "孟", "熊", "秦",
    "邱", "江", "尹", "薛", "阎", "段", "雷", "侯", "龙", "史",
    "陶", "黎", "贺", "顾", "毛", "郝", "龚", "邵", "万", "钱",
    "严", "武", "戴", "莫", "孔", "向", "汤", "温", "庞", "殷",
    "章", "葛", "管", "甘", "卞", "冉", "蓝", "殷", "习",
)
_SURNAMES_PATTERN = "[" + "".join(_CHINESE_SURNAMES) + "]"

# 中文动词接续模式（MemPalace zh-CN：人名后跟动词时增强识别）
_CHINESE_PERSON_VERBS: tuple[str, ...] = (
    "说", "问", "答", "表示", "回答", "提出", "决定", "认为",
    "指出", "解释", "告诉", "写道", "想", "觉得", "知道", "喜欢",
    "讨厌", "确认", "提醒", "分享", "建议", "同意", "反对",
)

# 中文对话标记模式（MemPalace zh-CN）
_CHINESE_DIALOGUE_PREFIXES: tuple[str, ...] = (
    ">", ":", "：", "」", "」",
)

# 中文停用词（MemPalace zh-CN 140+ 词，用作实体过滤）
_CHINESE_STOPWORDS: set[str] = {
    "的", "了", "着", "过", "得", "地", "吗", "吧", "呢", "啊", "喔", "耶",
    "我", "你", "妳", "他", "她", "它", "您", "咱",
    "我们", "你们", "妳们", "他们", "她们", "它们", "咱们",
    "自己", "大家", "有人", "没人",
    "今天", "明天", "昨天", "前天", "后天", "今年", "明年", "去年",
    "早上", "下午", "晚上", "中午", "凌晨",
    "现在", "刚才", "刚刚", "等等", "等下", "待会",
    "最近", "以前", "之前", "之后", "以后", "后来",
    "什么", "为什么", "怎么", "怎样", "哪里", "哪个",
    "这个", "那个", "这里", "那里", "这些", "那些", "这样", "那样",
    "但是", "可是", "然后", "所以", "因为", "如果", "虽然",
    "而且", "或者", "或是", "还是", "不过", "只是", "不只",
    "既然", "不然", "否则", "此外", "另外",
    "很", "非常", "相当", "真的", "确实", "当然", "其实",
    "已经", "正在", "即将", "将要", "刚好", "恰好",
    "可能", "也许", "或许", "大概", "应该", "必须", "一定",
    "完成", "执行", "进行", "开始", "结束", "继续", "停止", "完毕",
    "没有", "有点", "有些", "一些", "许多", "很多",
    "问题", "答案", "原因", "结果", "情况", "状况",
    "主要", "重要", "基本", "简单", "复杂", "特别",
    "谢谢", "感谢", "对不起", "不好意思", "请问",
    "欢迎", "再见", "你好", "您好", "哈喽", "拜拜",
}

# 默认关键词实体提取规则（中文命名实体关键词）
# 整合 MemPalace zh-CN 的 entity detection 增强
DEFAULT_ENTITY_PATTERNS: list[tuple[str, str]] = [
    # 0. 姓氏 + 名字 + 言语动词（高精度，带动词后缀）
    (rf"{_SURNAMES_PATTERN}[\u4e00-\u9fff]{{1,2}}(?:{'|'.join(_CHINESE_PERSON_VERBS)})", "person"),
    # 1. 姓氏 + 名字（最多 3 字，避免吞入后续动词/助词，且后面不能跟汉字；排除常见产品名如"方案"）
    # 后瞻：排除"了/是/案"；下一个若是汉字则排除（但"也"助词例外，可在 _clean 中剥离）
    (rf"{_SURNAMES_PATTERN}[\u4e00-\u9fff]{{1,2}}(?![了是案]|(?<!也)[\u4e00-\u9fff])", "person"),
    # 2. 头衔后缀（老师/主任/局长等，前面不能有汉字，避免吞入前缀）
    (r"(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{1,4}(?:医生|大夫|老师|先生|女士|同志|经理|主任|教授|院长|局长)", "person"),
    # 3. 组织（局/公司/部委等，后缀前最多 8 字，且以汉字开头，前面不能是介词/动词）
    # 注意：排除"局长"避免与 person 头衔冲突（"赵局长"应为 person 非 org）
    (r"(?<![了的是在跟与])(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{1,8}(?:局(?!长)|公司|医院|集团|部|委|办|院|中心|社|协会|基金会|大学|学院|研究院)", "org"),
    # 4. 地点（省/市/县等，后缀前最多 8 字，前面不能是介词/动词）
    (r"(?<![了的是在跟与])(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{1,8}(?:省|市|县|区|镇|乡|村|路|街|大道|桥|山|河|湖|江|海)", "location"),
    # 5. 产品/方案（后缀前最多 4 字，且不能以动词开头）
    (r"(?<![是说了和跟与在])(?<![\u4e00-\u9fff]{3})[\u4e00-\u9fff]{1,4}(?:系统|平台|软件|APP|工具|API|协议|模型|框架|方案|管线|引擎)", "product"),
    # 6. 英文名（如 John Smith）
    (r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "person"),
    # 7. 英文缩写（如 NASA）
    (r"[A-Z]{2,}(?:-[A-Z]+)*", "org"),
]

# 事实类型关键词（Hindsight 4 类结构化记忆）
# Bug fix (BUG-10): 清理重复项。
# Bug fix (BUG-7): world 关键词移除极常见的"是/的/有/构成"等虚词，
#   避免任何带"的"的句子误判为 world。world 仅保留明确的客观事实动词。
FACT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "observation": ["观察", "发现", "看到", "注意到", "感觉", "觉得", "现象", "情况", "状态", "天气", "今天", "外面", "这里", "那里"],
    "experience": ["经历", "做过", "尝试", "体验", "用过", "试过", "实施", "执行", "完成", "去过", "来过", "去了", "来了", "做了", "吃过"],
    "world": ["属于", "位于", "包括", "包含", "定义", "指", "代表", "省会", "首都", "号称", "是来自", "名称为", "由构成"],
    "opinion": ["认为", "建议", "推荐", "应该", "值得", "偏好", "倾向", "不太"],
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
                # Bug fix (BUG-8): LLM 路径原先不产生 authority/trust_score，
                # 导致 engine.py 的 INSERT 用 .get() 兜底，全落 medium/0.5。
                # 这里用本地规则为 LLM 结果补充权威等级和信任分。
                authority = self._evaluate_authority("\n".join(messages), result["fact_type"], len(result.get("entities", [])))
                trust_score = self._compute_trust_score(
                    result["fact_type"],
                    len(result.get("entities", [])),
                    len("\n".join(messages)),
                )
                result.setdefault("authority", authority)
                result.setdefault("trust_score", trust_score)
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
            # Bug fix (BUG-5): confidence 与 trust_score 计算逻辑完全相同，
            # 且 confidence 无对应数据库列，白算不落库。统一收敛到 trust_score，
            # 避免冗余平行字段。上游 engine.py INSERT 只写 authority/trust_score。
            "authority": self._evaluate_authority(combined, fact_type, len(entities)),
            "trust_score": self._compute_trust_score(fact_type, len(entities), len(combined)),
        }

    def _extract_entities(self, text: str) -> list[dict]:
        """关键词实体提取（整合 MemPalace zh-CN 增强）。

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
                # MemPalace: 中文停用词过滤 —— 排除纯停用词组成的候选
                if name in _CHINESE_STOPWORDS or all(c in _CHINESE_STOPWORDS for c in name if '\u4e00' <= c <= '\u9fff'):
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
        "说", "讲", "谈", "聊", "强调", "指出", "提到", "提及", "开会讨论", "发布", "推出", "上线", "合作", "签署", "达成", "宣布",
        "准备", "打算", "计划", "考虑", "看", "看看", "最近", "前", "后",
        "学习", "研究", "开发", "建设", "管理", "运营", "服务", "提供",
        "支持", "实现", "打造", "构建", "推动", "促进", "加强", "与",
        "和", "及", "向", "为", "给", "对", "把", "从", "在", "跟", "同",
        "还有", "也是", "就是", "作为", "成为", "属于", "位于",
        "这个", "那个", "这些", "那些", "某个", "一些", "每个",
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
    # Bug fix (BUG-9): product 类型下这些词+产品后缀是合法真实产品名
    # （如"管理系统"、"评估平台"），不能被当作噪声前缀剥掉。
    # 原实现无限循环剥前缀 → "管理系统"被剥成"系统"，丢失真实实体名。
    _PRODUCT_SAFE_PREFIXES = frozenset({
        "评估", "测试", "管理", "开发", "服务", "支持", "使用", "研究",
        "建设", "运营", "提供", "实现", "打造", "学习", "分析", "监控",
        "预测", "营销", "客户", "智能", "数字",
    })

    def _clean_entity_name(self, name: str, ent_type: str = "org") -> str:
        """剥离常见非实体前缀 + 按实体类型后缀词裁剪，保留紧凑实体名。"""
        # 1. 循环剥离非实体前缀（"用户参与了X公司" -> "X公司"）
        # person 类型特判：单字姓氏+单字名但实际是常见产品名
        if ent_type == "person" and name in ("方案", "系统", "平台", "模型", "框架", "引擎", "管线", "器具", "工具"):
            return ""
        changed = True
        # 防死循环：最多剥 6 轮（Bug fix BUG-9）
        guard = 0
        while changed and len(name) > 2 and guard < 6:
            guard += 1
            changed = False
            for p in self._NOISE_PREFIXES:
                # Bug fix (BUG-9): product 类型跳过"产品安全前缀"（如"管理"），
                # 否则"管理系统" → "系统"。其余类型继续剥。
                if p in self._PRODUCT_SAFE_PREFIXES and ent_type == "product":
                    continue
                if name.startswith(p) and len(name) > len(p) + 1:
                    name = name[len(p):]
                    changed = True
                    break
        # 2. 按类型后缀定位：只保留最后一个后缀词及其前最多 8 个汉字
        # 特判：剥离言语动词（"王哥认为"->"王哥"，"何总工认为云南省"->"何总工"）
        for v in sorted(_CHINESE_PERSON_VERBS, key=len, reverse=True):
            if v in name and len(name) > len(v):
                idx = name.find(v)
                if idx > 0:
                    name = name[:idx]
                break
        # person 类型特判：尾部剥离"也"等助词（"王哥也" -> "王哥"）
        if ent_type == "person" and name.endswith("也"):
            name = name[:-1]
        suffixes = self._TYPE_SUFFIXES.get(ent_type, self._SUFFIX_WORDS)
        for suffix in sorted(suffixes, key=len, reverse=True):
            idx = name.rfind(suffix)
            if idx > 0:
                head = name[max(0, idx - 8):idx]
                # 实体名通常从"的/在/跟"等分隔词之后开始（如"云岭集团的量子计算平台"、
                # "王老师在云南省昆明市"、"陈主任跟赵局长"），截断到最后一个分隔符之后
                for sep in ("的", "在", "跟", "到", "去", "来"):
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

    def _evaluate_authority(self, text: str, fact_type: str, entity_count: int) -> str:
        """评估记忆的权威等级（Ground Truth 层级）。

        规则（Bug fix BUG-1/2: 提升区分度，摆脱对 fact_type 的过度依赖）：
        - 第一人称直接声明（我是/我叫/我的/我喜欢等）→ critical
        - 明确的个人经历动词（我做过/我用过/我吃过等）→ critical
        - 客观事实（world）或高实体数的具体陈述（实体≥3）→ high
        - 有实体的观察/经历（实体≥1）→ medium
        - 观察、或一般性陈述（observation）→ medium
        - 主观意见（opinion）、无实体的空泛观察 → low
        - 默认 → medium

        Returns:
            "critical" | "high" | "medium" | "low"
        """
        # 第一人称直接声明：用户自己说的，权威最高
        first_person_patterns = ["我是", "我叫", "我的", "我姓", "我住在", "我工作", "我今年", "我来自", "我喜欢", "我住", "我在"]
        for p in first_person_patterns:
            if p in text[:200]:
                return "critical"

        # 明确的个人经历动词 → critical（用户亲历可作第一手凭证）
        experience_verbs = ["我做过", "我用过", "我吃过", "我去过", "我试过", "我用了", "我做了", "我完成了"]
        for v in experience_verbs:
            if v in text[:300]:
                return "critical"

        # 客观事实（world）或具体实体陈述（≥3 个实体）→ high
        if fact_type == "world" and entity_count >= 1:
            return "high"
        if entity_count >= 3:
            return "high"
        if fact_type == "experience":
            return "high"

        # 有实体的观察/普通陈述 → medium
        if entity_count >= 1:
            return "medium"
        if fact_type == "observation":
            return "medium"

        # 主观意见、或完全无实体的空泛内容 → low
        if fact_type == "opinion":
            return "low"

        return "medium"

    def _compute_trust_score(self, fact_type: str, entity_count: int, text_length: int) -> float:
        """计算记忆的 trust score（0-1）。

        综合考量：
        - 事实类型的基础置信度
        - 实体数量（越多越可靠，但不超过 5 个）
        - 文本长度（50-500 字为最佳区间）

        Bug fix (BUG-2): 恢复区分度。原先因 world 关键词误判导致大量文本
        落 world(0.9)，trust 全被拉高到接近 0.9；现在 world 关键词收紧后
        分布自然回落。仍按类型*实体*长度加权。

        Returns:
            0-1 之间的 trust score
        """
        base = FACT_TYPE_CONFIDENCE.get(fact_type, 0.5)
        # 实体数量加成（最多 +0.15）
        entity_bonus = min(0.15, entity_count * 0.03)
        # 文本长度加成
        if 50 <= text_length <= 500:
            length_bonus = 0.1
        elif text_length < 20:
            length_bonus = -0.15  # 过短惩罚
        elif text_length > 2000:
            length_bonus = -0.05  # 过长噪声
        else:
            length_bonus = 0.0
        # 落到 2 位小数，便于区分（Bug fix: 避免出现 0.8099999 这类浮点尾差）
        return round(max(0.1, min(1.0, base + entity_bonus + length_bonus)), 2)


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