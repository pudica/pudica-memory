"""search/types.py — 检索结果类型定义。

参考文档 7.3 节的 ScoredResult 和 FusionResult 类型定义。
"""

from dataclasses import dataclass, field


@dataclass
class ScoredResult:
    """单策略检索结果条目。

    Attributes:
        id: 文档 ID
        text: 文本内容
        score: 该策略内的原始得分（0-1）
        source: 策略来源（semantic / bm25 / graph / temporal）
        rank: 该策略内的排名
        metadata: 附加元数据
    """
    id: str
    text: str
    score: float = 0.0
    source: str = "unknown"
    rank: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class FusionResult:
    """RRF 融合后的最终结果。

    Attributes:
        id: 文档 ID
        text: 文本内容
        score: RRF 融合得分
        sources: 来自哪些策略
        metadata: 附加元数据
    """
    id: str
    text: str
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)