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

# ========================================================================
# 垃圾实体黑名单（KG 不写入，脚本和实时管线共享）
# 避免 L1 本地规则提取出工具名、代码路径、内部状态等非语义实体
# ========================================================================
GARBAGE_ENTITIES: set[str] = {
    "normal", "DB", "KG", "3", "5", "10", "OK", "N/A",
    "ingest_count", "search_count", "flush_count", "l1_count", "l2_count",
    "system_health", "system_stats", "mempalace_status", "mempalace_search",
    "mental_models_query", "memory_search", "memory_context", "memory_ingest",
    "kg_query", "verbatim_recall", "consolidate_trigger", "memory_compress",
    "pipeline_run", "reflect_trigger", "list_prompts", "get_prompt",
    "list_resources", "read_resource", "mempalace_list_wings", "mempalace_list_rooms",
    "mempalace_list_drawers", "mempalace_get_drawer", "mempalace_get_taxonomy",
    "mempalace_add_drawer",
    "scripts/l1_replay_all.py", "turn_context.py", "memory.provider",
    "sync_turn", "prefetch", "pudica-memory-development",
    "L1提取器", "L2场景组织", "引擎初始化", "MCP子进程",
    "HTTP 8420服务", "unified_memory collection",
    "dsh_execute_code", "dsh", "DeepSeek", "ARK", "Gateway",
    "47.116.76.149", "Windows", "MentalModelStore",
    "flush_count", "l1_count", "l2_count", "search_count",
    "scripts/l1_llm_replay.py", "pudica-memory",
    "Ark deepseek", "unified_memory",
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

# 事实类型关键词（Hindsight 4 类结构化记忆 + causation）
FACT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "observation": ["观察", "发现", "看到", "注意到", "感觉", "觉得", "现象", "情况", "状态", "天气", "今天", "外面", "这里", "那里"],
    "experience": ["做过", "去过", "吃过", "用过", "看过", "听过", "试过", "去过", "完成了", "做了", "经历了", "参加了", "体验了", "之前", "曾经", "以前"],
    "world": ["是", "属于", "位于", "定义", "指", "叫做", "就是", "指代", "概念", "方法", "原理", "规则", "标准", "系统", "机制", "结构", "功能", "分类"],
    "opinion": ["喜欢", "不喜欢", "觉得", "认为", "建议", "想要", "希望", "偏好", "愿意", "倾向", "应该", "最好", "推荐", "更愿意", "推荐", "我建议", "我觉得", "更喜欢", "不太喜欢", "个人认为", "我个人觉得"],
    "causation": ["因为", "所以", "导致", "造成", "引发", "引起", "使得", "促使", "源于", "起因", "结果", "于是", "因此", "由此", "从而", "以致", "触发", "带来", "产生", "诱发", "由于", "因而", "故此", "以至于", "正因为"],
}

# 事实类型置信度权重（Hindsight: 带置信度的事实提取）
FACT_TYPE_CONFIDENCE: dict[str, float] = {
    "world": 0.9,       # 世界知识，最可靠
    "observation": 0.7,  # 观察到的事实
    "experience": 0.8,   # 个人经历，较可靠
    "opinion": 0.5,      # 主观意见，置信度较低
    "causation": 0.6,    # 因果推理，依赖上下文准确性
}


class L1Extractor:
    """L1：本地规则提取器（关键词实体 + 时间戳摘要）。

    Args:
        llm: 可选的 LLM 客户端，提供后自动启用 LLM 增强模式
    """

    def __init__(self, llm: Any = None):
        self._llm = llm
        self._max_retries = 2
        self._local_only = llm is None
        self._registry: Optional["EntityRegistry"] = None

    def set_registry(self, registry: "EntityRegistry") -> None:
        """设置实体注册表（可选，有则启用 KG 白名单过滤）。"""
        self._registry = registry

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

        return await self._local_extract(messages)

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
                # ARK 等有些 provider 返回的 JSON 被 markdown 代码块包裹
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    first_newline = cleaned.find("\n")
                    if first_newline != -1:
                        cleaned = cleaned[first_newline + 1:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                result = json.loads(cleaned)
                result.setdefault("entities", [])
                result.setdefault("relations", [])
                result.setdefault("summary", "")
                if not result["summary"]:
                    result["summary"] = self._make_summary("\n".join(messages))
                    logger.info("L1 LLM 未返回 summary，本地规则补全: %s", result["summary"][:40])
                else:
                    logger.info("L1 LLM 正常返回 summary: %s", result["summary"][:40])
                result.setdefault("time_range", {})
                result.setdefault("fact_type", "observation")
                result.setdefault("confidence", 0.7)
                authority = self._evaluate_authority("\n".join(messages), result["fact_type"], len(result.get("entities", [])))
                trust_score = self._compute_trust_score(
                    result["fact_type"],
                    len(result.get("entities", [])),
                    len("\n".join(messages)),
                )
                result.setdefault("authority", authority)
                result.setdefault("trust_score", trust_score)
                # 实体注册表过滤
                if self._registry is not None:
                    result["entities"] = self._filter_entities_by_registry(result["entities"])
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

    async def _local_extract(self, messages: list[str]) -> dict:
        """纯本地规则提取：关键词实体 + 时间戳摘要。

        Returns:
            提取结果字典
        """
        combined = "\n".join(messages)

        # 1. 关键词实体提取
        entities = self._extract_entities(combined)

        # 2. 实体注册表过滤
        if self._registry is not None:
            entities = await self._filter_entities_by_registry(entities)

        # 3. 事实类型判断
        fact_type = self._classify_fact_type(combined)

        # 4. 摘要（截取前 200 字，取完整句子）
        summary = self._make_summary(combined)

        # 5. 时间戳
        now = datetime.now(timezone.utc)
        time_range = {
            "start": now.isoformat(),
            "end": now.isoformat(),
        }

        return {
            "entities": entities,
            "relations": [],
            "summary": summary,
            "time_range": time_range,
            "fact_type": fact_type,
            "authority": self._evaluate_authority(combined, fact_type, len(entities)),
            "trust_score": self._compute_trust_score(fact_type, len(entities), len(combined)),
        }

    async def _filter_entities_by_registry(self, entities: list[dict]) -> list[dict]:
        """根据实体注册表白名单过滤实体列表。

        只有注册表中 state='confirmed' 的实体才保留。
        不在注册表中的实体，如果被标记为 rejected 则移除。
        未知实体（不在注册表中）暂时保留，但标记为 unregistered 供后续自动发现。
        """
        if self._registry is None:
            return entities

        filtered = []
        for ent in entities:
            name = ent.get("name", "")
            if not name:
                continue
            # 先检查 GARBAGE_ENTITIES 黑名单
            if name in GARBAGE_ENTITIES:
                continue
            state = self._registry.get_state_sync(name)
            if state == "confirmed":
                filtered.append(ent)
            elif state == "rejected":
                continue  # 明确拒绝的实体不写入 KG
            else:
                # 未知 / candidate 实体：注册到注册表以递增计数
                # 异步注册，不阻塞过滤流程
                try:
                    await self._registry.register(name, ent.get("entity_type", "org"))
                except Exception:
                    pass  # 注册失败不影响过滤
                filtered.append(ent)
        return filtered

    def _extract_entities(self, text: str) -> list[dict]:
        """关键词实体提取（整合 MemPalace zh-CN 增强）。

        Bug fix: 英文通用词（Windows、Error、Installer、UTF 等）原本被
        英文人名/缩写规则误判为 person/org。现在对纯英文候选做停用词过滤，
        只在确属专有名词（首字母多词组、或大缩写）时才保留。

        P1 增强: 每个实体附加 signal-based confidence 评分（0.3-0.99），
        基于对话模式、动词上下文、出现次数等信号。

        Returns:
            [{"name": "...", "type": "person|org|location|product",
              "confidence": "0.0-1.0", "signals": [...]}, ...]
        """
        seen: set[str] = set()
        entities: list[dict] = []
        # 收集每个候选名的出现次数（用于频率信号）
        name_counts: dict[str, int] = {}
        text_lower = text.lower()

        for pattern, ent_type in DEFAULT_ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group().strip()
                if len(name) < 2:
                    continue
                # MemPalace: 中文停用词过滤
                if name in _CHINESE_STOPWORDS or all(c in _CHINESE_STOPWORDS for c in name if '\u4e00' <= c <= '\u9fff'):
                    continue
                # GARBAGE_ENTITIES 黑名单
                if name in GARBAGE_ENTITIES:
                    continue
                # 英文通用词过滤
                if self._is_english_generic(name):
                    continue
                name = self._clean_entity_name(name, ent_type)
                if len(name) < 2:
                    continue
                name_lower = name.lower()
                if name_lower not in seen:
                    seen.add(name_lower)
                    name_counts[name_lower] = 1
                    entities.append({"name": name, "type": ent_type})
                else:
                    name_counts[name_lower] += 1

        # P1: 信号评分 — 给每个实体算 confidence 和 signals
        for ent in entities:
            score = self._score_entity(ent["name"], ent["type"], text, text_lower, name_counts.get(ent["name"].lower(), 1))
            ent["confidence"] = score["confidence"]
            ent["signals"] = score["signals"]

        # 去重后取前 20 个
        return entities[:20]

    @staticmethod
    def _score_entity(name: str, ent_type: str, text: str, text_lower: str, frequency: int) -> dict:
        """对单个实体做信号评分，返回 confidence 和 signals。

        移植自 MemPalace score_entity/classify_entity 的核心思想（信号评分制），
        但适配中文场景：用对话标记、动词上下文、出现频率等信号。

        Args:
            name: 实体名
            ent_type: 实体类型（person/org/location/product）
            text: 原始文本
            text_lower: 小写文本（预计算）
            frequency: 实体在文本中出现次数

        Returns:
            {"confidence": 0.0-0.99, "signals": [...]}
        """
        signals = []
        score = 0.0
        name_lower = name.lower()

        # 信号1: 频率 — 出现次数越多越可信
        if frequency >= 3:
            score += 0.15
            signals.append(f"高频({frequency}x)")
        elif frequency >= 2:
            score += 0.08
            signals.append(f"中频({frequency}x)")

        # 信号2: 对话标记 — 实体名后跟冒号/说/问等
        # 检查 ":name" 或 "name:" 或 "name说" 等模式
        if ent_type == "person":
            name_in_text = name in text
            if name_in_text:
                # 对话标记：name: 或 name：或 name说
                dialogue_patterns = [
                    name + ":", name + "：", name + "说",
                    name + "问", name + "答", name + "表示",
                    name + "认为", name + "指出", name + "回答",
                    name + "解释", name + "告诉", name + "写道",
                    name + "想", name + "觉得", name + "知道",
                    name + "喜欢", name + "确认", name + "提醒",
                    name + "分享", name + "建议", name + "同意",
                    name + "反对", name + "决定", name + "提出",
                ]
                if any(p in text for p in dialogue_patterns):
                    score += 0.2
                    signals.append("对话标记")
                # 动词上下文：name + 做的/用了/完成了 等
                action_patterns = [
                    name + "做了", name + "用了", name + "完成了",
                    name + "参加了", name + "去过", name + "用过",
                    name + "吃过", name + "看过", name + "听过",
                    name + "负责", name + "管理", name + "处理",
                    name + "开发", name + "设计", name + "编写",
                    name + "创建", name + "修改", name + "修复",
                    name + "测试", name + "部署", name + "发布",
                ]
                if any(p in text for p in action_patterns):
                    score += 0.15
                    signals.append("动作上下文")
                # 代词邻近：实体名附近 3 行内有他/她/它
                name_idx = text.find(name)
                if name_idx >= 0:
                    window_start = max(0, text.rfind("\n", 0, name_idx) - 50)
                    window_end = min(len(text), text.find("\n", name_idx) + 50)
                    window = text[window_start:window_end]
                    pronouns = ["他", "她", "它", "他们", "她们", "它们", "他", "她", "其"]
                    if any(p in window for p in pronouns):
                        score += 0.1
                        signals.append("代词邻近")

        # 信号3: 实体类型基分
        type_base = {"person": 0.2, "org": 0.15, "location": 0.15, "product": 0.1}
        base = type_base.get(ent_type, 0.1)
        score += base
        # 首次出现时不加 type 信号描述，避免信号过多

        # 信号4: 中文人名 — 姓氏+名字结构本身可信
        if ent_type == "person" and any(name.startswith(s) for s in _CHINESE_SURNAMES):
            score += 0.1
            signals.append("中文姓氏")

        # 信号5: 英文名 — 首字母大写的多词短语
        if ent_type == "person" and " " in name and name[0].isupper():
            score += 0.1
            signals.append("英文全名")

        # 信号6: 全大写缩写 — 组织名
        if ent_type == "org" and name.isupper() and len(name) >= 2:
            score += 0.1
            signals.append("大写缩写")

        # 信号7: 头衔后缀 — 老师/主任/局长等
        title_suffixes = ["医生", "大夫", "老师", "先生", "女士", "同志", "经理", "主任", "教授", "院长", "局长"]
        if any(name.endswith(s) for s in title_suffixes):
            score += 0.15
            signals.append("头衔后缀")

        # 截断并归一化到 0.3-0.99
        confidence = max(0.3, min(0.99, score))
        confidence = round(confidence, 2)

        return {
            "confidence": confidence,
            "signals": signals[:5],  # 最多保留 5 个信号
        }

    _ENGLISH_GENERIC_WORDS = frozenset({
        "error", "window", "windows", "install", "installer", "dll", "utf", "utf8",
        "user", "profile", "system", "file", "config", "api", "app", "tool", "tools",
        "model", "models", "framework", "platform", "software", "protocol", "key",
        "keys", "default", "true", "false", "none", "null", "test", "tests",
        "fix", "fixed", "bug", "bugs", "debug", "code", "codes", "data", "info",
        "message", "messages", "memory", "memories", "server", "client",
        "service", "services", "thread", "threads", "pool", "process", "processes",
        "exe", "py", "pyinstaller",
    })

    @staticmethod
    def _is_english_generic(name: str) -> bool:
        """判断纯英文候选是否为通用词（非专有名词）。

        Bug fix: 原实现把"任何不带空格的纯英文词"一律过滤，
        导致 GPT4、Qwen2、Claude3、Llama3 等模型名全被误杀，KG 的 product
        实体稀疏。现在放宽：
        - 含数字的模型名（GPT4/Qwen2/Llama3/DeepSeek-V3）→ 保留
        - 全大写多字母缩写（NASA/IBM）→ 保留
        - 仅字母且非通用词表、且首字母大写且 >3 字符的专有名词 → 保留
        - 小写/混合且不在词表的普通英文 → 过滤（多为噪声）
        """
        lowered = name.lower()
        # 通用词表里的 → 过滤
        if lowered in L1Extractor._ENGLISH_GENERIC_WORDS:
            return True
        # 含数字的（GPT4、Qwen2、DeepSeek-V3）→ 模型/版本名，保留
        if any(ch.isdigit() for ch in name):
            return False
        # 多词短语（John Smith）→ 英文人名，保留
        if " " in name:
            return False
        # 全大写缩写（NASA）→ 保留
        if name.isupper() and len(name) >= 3:
            return False
        # 已过滤纯符号/空
        if not name.strip("."):
            return True
        # 首字母大写且大于 4 字符的专有名词 → 保留（如 Claude、Arcee）
        if name[0].isupper() and len(name) > 4:
            return False
        # 其余小写普通英文 → 过滤
        return True

    # 实体名清洗：剥离常见非实体前缀，并按后缀词定位裁剪
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
    _PRODUCT_SAFE_PREFIXES = frozenset({
        "评估", "测试", "管理", "开发", "服务", "支持", "使用", "研究",
        "建设", "运营", "提供", "实现", "打造", "学习", "分析", "监控",
        "预测", "营销", "客户", "智能", "数字",
    })

    def _clean_entity_name(self, name: str, ent_type: str = "org") -> str:
        """剥离常见非实体前缀 + 按实体类型后缀词裁剪，保留紧凑实体名。"""
        # 1. 循环剥离非实体前缀
        if ent_type == "person" and name in ("方案", "系统", "平台", "模型", "框架", "引擎", "管线", "器具", "工具"):
            return ""
        changed = True
        guard = 0
        while changed and len(name) > 2 and guard < 6:
            guard += 1
            changed = False
            for p in self._NOISE_PREFIXES:
                if p in self._PRODUCT_SAFE_PREFIXES and ent_type == "product":
                    continue
                if name.startswith(p) and len(name) > len(p) + 1:
                    name = name[len(p):]
                    changed = True
                    break
        # 2. 按类型后缀定位
        for v in sorted(_CHINESE_PERSON_VERBS, key=len, reverse=True):
            if v in name and len(name) > len(v):
                idx = name.find(v)
                if idx > 0:
                    name = name[:idx]
                break
        if ent_type == "person" and name.endswith("也"):
            name = name[:-1]
        suffixes = self._TYPE_SUFFIXES.get(ent_type, self._SUFFIX_WORDS)
        for suffix in sorted(suffixes, key=len, reverse=True):
            idx = name.rfind(suffix)
            if idx > 0:
                head = name[max(0, idx - 8):idx]
                for sep in ("的", "在", "跟", "到", "去", "来"):
                    if sep in head:
                        head = head.split(sep)[-1]
                cleaned = head + suffix
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
        """评估记忆的权威等级（Ground Truth 层级）。"""
        first_person_patterns = ["我是", "我叫", "我的", "我姓", "我住在", "我工作", "我今年", "我来自", "我喜欢", "我住", "我在",
                                 "我平时", "我觉得", "我最近", "我每天", "我主要", "我一直", "我一般", "我经常", "我负责",
                                 "我养", "我有", "我需要", "我想", "我打算", "我在用", "我用", "我吃", "我去", "我来"]
        for p in first_person_patterns:
            if p in text:
                return "critical"

        experience_verbs = ["我做过", "我用过", "我吃过", "我去过", "我试过", "我用了", "我做了", "我完成了"]
        for v in experience_verbs:
            if v in text:
                return "critical"

        if fact_type == "world" and entity_count >= 1:
            return "high"
        if entity_count >= 3:
            return "high"
        if fact_type == "experience":
            return "high"

        if entity_count >= 1:
            return "medium"
        if fact_type == "observation":
            return "medium"

        if fact_type == "opinion":
            return "low"

        return "medium"

    def _compute_trust_score(self, fact_type: str, entity_count: int, text_length: int) -> float:
        """计算记忆的 trust score（0-1）。"""
        base = FACT_TYPE_CONFIDENCE.get(fact_type, 0.5)
        entity_bonus = min(0.15, entity_count * 0.03)
        if 50 <= text_length <= 500:
            length_bonus = 0.1
        elif text_length < 20:
            length_bonus = -0.15
        elif text_length > 2000:
            length_bonus = -0.05
        else:
            length_bonus = 0.0
        return round(max(0.1, min(1.0, base + entity_bonus + length_bonus)), 2)


def _build_llm_prompt(messages: list[str]) -> str:
    """构建 LLM 提取提示词。"""
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
   - "causation": 因果关系/原因结果
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