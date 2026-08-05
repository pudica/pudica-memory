"""benchmark/eval.py — 记忆系统评估工具（对标 LongMemEval）。

评估指标：
  - Recall@k: 前 k 条结果中包含正确答案的比例
  - MRR (Mean Reciprocal Rank): 正确答案排名的倒数的均值
  - NDCG@k: 归一化折损累积增益
  - Latency: 检索延迟分布

使用方法：
  from unified_memory.benchmark.eval import BenchmarkRunner
  runner = BenchmarkRunner(app)
  results = await runner.run(dataset)
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """单条评估样本。

    Attributes:
        query: 查询文本
        relevant_ids: 相关记忆 ID 列表（ground truth）
        category: 评估类别（single_hop / multi_hop / temporal / entity）
    """
    query: str
    relevant_ids: list[str]
    category: str = "general"


@dataclass
class EvalResult:
    """单条评估结果。"""
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_ms: float
    category: str = "general"


@dataclass
class BenchmarkSummary:
    """评估汇总。"""
    total_samples: int = 0
    avg_recall_at_1: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_recall_at_10: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    by_category: dict[str, dict] = field(default_factory=dict)


class BenchmarkRunner:
    """记忆系统评估运行器。

    对标 Hindsight 的 LongMemEval 评估和 MemPalace 的 R@5 指标。
    """

    def __init__(self, app: Any):
        """
        Args:
            app: UnifiedMemoryApp 实例
        """
        self._app = app

    async def run(self, dataset: list[EvalSample]) -> BenchmarkSummary:
        """运行完整评估。

        Args:
            dataset: 评估数据集

        Returns:
            评估汇总结果
        """
        results: list[EvalResult] = []

        for sample in dataset:
            result = await self._evaluate_single(sample)
            results.append(result)

        return self._summarize(results)

    async def _evaluate_single(self, sample: EvalSample) -> EvalResult:
        """评估单条样本。"""
        t0 = time.time()
        # 执行检索
        search_results = await self._app.pipeline.search(sample.query, top_k=10)
        latency_ms = (time.time() - t0) * 1000

        retrieved_ids = [r.id for r in search_results]
        relevant_set = set(sample.relevant_ids)

        # Recall@k
        recall_at_1 = self._recall_at_k(retrieved_ids, relevant_set, 1)
        recall_at_5 = self._recall_at_k(retrieved_ids, relevant_set, 5)
        recall_at_10 = self._recall_at_k(retrieved_ids, relevant_set, 10)

        # MRR
        mrr = self._mrr(retrieved_ids, relevant_set)

        # NDCG@10
        ndcg = self._ndcg_at_k(retrieved_ids, relevant_set, 10)

        return EvalResult(
            query=sample.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=sample.relevant_ids,
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_10=ndcg,
            latency_ms=latency_ms,
            category=sample.category,
        )

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """计算 Recall@k。"""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        """计算 MRR (Mean Reciprocal Rank)。"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """计算 NDCG@k。"""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)

        # 理想 DCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    def _summarize(self, results: list[EvalResult]) -> BenchmarkSummary:
        """汇总评估结果。"""
        if not results:
            return BenchmarkSummary()

        n = len(results)
        summary = BenchmarkSummary(total_samples=n)

        # 全局指标
        summary.avg_recall_at_1 = sum(r.recall_at_1 for r in results) / n
        summary.avg_recall_at_5 = sum(r.recall_at_5 for r in results) / n
        summary.avg_recall_at_10 = sum(r.recall_at_10 for r in results) / n
        summary.avg_mrr = sum(r.mrr for r in results) / n
        summary.avg_ndcg_at_10 = sum(r.ndcg_at_10 for r in results) / n
        summary.avg_latency_ms = sum(r.latency_ms for r in results) / n

        # 延迟分位数
        latencies = sorted(r.latency_ms for r in results)
        summary.p50_latency_ms = latencies[n // 2]
        summary.p95_latency_ms = latencies[int(n * 0.95)] if n > 1 else latencies[0]

        # 按类别分组
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        for cat, cat_results in categories.items():
            cn = len(cat_results)
            summary.by_category[cat] = {
                "count": cn,
                "recall_at_5": sum(r.recall_at_5 for r in cat_results) / cn,
                "mrr": sum(r.mrr for r in cat_results) / cn,
                "ndcg_at_10": sum(r.ndcg_at_10 for r in cat_results) / cn,
            }

        return summary

    def format_report(self, summary: BenchmarkSummary) -> str:
        """格式化评估报告。"""
        lines = [
            "====== 记忆系统评估报告 ======",
            f"样本数: {summary.total_samples}",
            "",
            "--- 检索质量指标 ---",
            f"  Recall@1:  {summary.avg_recall_at_1:.1%}",
            f"  Recall@5:  {summary.avg_recall_at_5:.1%}",
            f"  Recall@10: {summary.avg_recall_at_10:.1%}",
            f"  MRR:       {summary.avg_mrr:.3f}",
            f"  NDCG@10:   {summary.avg_ndcg_at_10:.3f}",
            "",
            "--- 性能指标 ---",
            f"  平均延迟:  {summary.avg_latency_ms:.1f}ms",
            f"  P50 延迟:  {summary.p50_latency_ms:.1f}ms",
            f"  P95 延迟:  {summary.p95_latency_ms:.1f}ms",
        ]

        if summary.by_category:
            lines.append("")
            lines.append("--- 分类指标 ---")
            for cat, stats in summary.by_category.items():
                lines.append(
                    f"  {cat}: R@5={stats['recall_at_5']:.1%}, "
                    f"MRR={stats['mrr']:.3f}, NDCG={stats['ndcg_at_10']:.3f}"
                )

        return "\n".join(lines)


def load_dataset(path: str) -> list[EvalSample]:
    """从 JSON 文件加载评估数据集。

    格式：
    [
        {
            "query": "查询文本",
            "relevant_ids": ["id1", "id2"],
            "category": "single_hop"
        },
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvalSample(
            query=item["query"],
            relevant_ids=item["relevant_ids"],
            category=item.get("category", "general"),
        )
        for item in data
    ]
