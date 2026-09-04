"""search/reranker.py — Cross-Encoder 重排器（Hindsight 启发）。

在 RRF 融合后对 top-N 结果进行二次打分重排，提升检索精度。
支持三种策略：
  - "heuristic": 基于 TF-IDF 关键词加权、来源多样性、时间新鲜度的启发式打分
  - "llm": 使用 LLM 对 query-doc pair 进行相关性打分（精度更高但更慢）
  - "cross-encoder": 使用预训练 cross-encoder 模型进行语义相关性打分（精度最高）

参考 Hindsight 的 cross-encoder reranking 流程：
  1. RRF 融合产出候选集
  2. 对候选集逐条打分（query-doc relevance）
  3. 按新分数排序截断

v3.1 升级：启发式重排从简单子串匹配升级为 TF-IDF 加权打分，
利用候选集自身的文档频率计算 IDF，对查询词在文档中的出现做 TF-IDF 加权，
显著提升关键词匹配精度。
"""

import asyncio
import logging
import math
import time
from collections import Counter
from typing import Any, Optional

from unified_memory.search.types import FusionResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 重排器。

    在 RRF 融合结果之上进行二次精排，显著提升 top-k 精度。

    策略选择：
    - "heuristic": 零依赖启发式打分，适合离线/低延迟场景
    - "llm": LLM 逐条打分，精度最高但需要 LLM 可用
    """

    def __init__(
        self,
        strategy: str = "heuristic",
        top_n: int = 20,
        final_k: int = 10,
        llm: Any = None,
        max_concurrent: int = 4,
        cross_encoder_model: Optional[str] = None,
    ):
        """
        Args:
            strategy: 重排策略 — "heuristic", "llm" 或 "cross-encoder"
            top_n: 从 RRF 结果中取 top_n 条进行重排
            final_k: 重排后返回的最终条数
            llm: LLM 客户端（strategy="llm" 时必需）
            max_concurrent: LLM 重排时的最大并发数
            cross_encoder_model: cross-encoder 模型名称或路径（strategy="cross-encoder" 时必需）
        """
        self._strategy = strategy
        self._top_n = top_n
        self._final_k = final_k
        self._llm = llm
        self._sem = asyncio.Semaphore(max_concurrent)
        self._cross_encoder = None
        self._cross_encoder_tokenizer = None
        if strategy == "cross-encoder":
            self._load_cross_encoder(cross_encoder_model)

    async def rerank(
        self,
        query: str,
        results: list[FusionResult],
    ) -> list[FusionResult]:
        """对 RRF 融合结果进行重排。

        Args:
            query: 用户查询文本
            results: RRF 融合后的结果列表（按得分降序）

        Returns:
            重排后的结果列表（按新得分降序），最多 final_k 条
        """
        if not results:
            return results

        # 取 top_n 候选
        candidates = results[: self._top_n]

        if self._strategy == "llm" and self._llm is not None:
            scored = await self._llm_rerank(query, candidates)
        elif self._strategy == "cross-encoder" and self._cross_encoder is not None:
            scored = self._cross_encoder_rerank(query, candidates)
        else:
            scored = self._heuristic_rerank(query, candidates)

        # 按新分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 返回 final_k 条，用新分数更新 FusionResult
        reranked: list[FusionResult] = []
        for result, new_score in scored[: self._final_k]:
            reranked.append(FusionResult(
                id=result.id,
                text=result.text,
                score=new_score,
                sources=result.sources,
                metadata={**result.metadata, "_reranked": True, "_rrf_score": result.score},
            ))

        logger.debug(
            "重排完成: %d 候选 → %d 结果 (策略=%s)",
            len(candidates), len(reranked), self._strategy,
        )
        return reranked

    # ------------------------------------------------------------------
    # 启发式重排
    # ------------------------------------------------------------------

    def _heuristic_rerank(
        self,
        query: str,
        candidates: list[FusionResult],
    ) -> list[tuple[FusionResult, float]]:
        """启发式重排：TF-IDF 关键词加权 + 来源多样性 + RRF 得分 + 长度惩罚。

        v3.1 升级：从简单子串匹配改为 TF-IDF 加权打分。
        - 对候选集构建文档频率(DF)表，计算每个查询词的 IDF
        - 对每条候选计算查询词的 TF-IDF 加权覆盖率
        - 利用 BM25 公式中的饱和参数 k1=1.2, b=0.75 做长度归一化

        综合评分公式：
            score = 0.45 * tfidf_score + 0.15 * source_diversity
                  + 0.25 * rrf_score + 0.15 * length_penalty

        Args:
            query: 查询文本
            candidates: 候选结果列表

        Returns:
            [(FusionResult, new_score), ...]
        """
        query_terms = self._tokenize(query)

        # 构建 TF-IDF 模型：以候选集为语料库
        # 1. 分词每条候选文档
        doc_tokens_list: list[list[str]] = []
        for result in candidates:
            doc_tokens_list.append(self._tokenize(result.text))

        # 2. 计算文档频率 (DF)：每个词出现在多少篇文档中
        n_docs = len(candidates)
        df: Counter = Counter()
        for tokens in doc_tokens_list:
            for term in set(tokens):
                df[term] += 1

        # 3. 计算 IDF：idf(t) = log((N + 1) / (df(t) + 1)) + 1  (smooth idf)
        idf: dict[str, float] = {}
        for term in set(query_terms):
            idf[term] = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1

        # 4. 对每条候选计算 TF-IDF 加权得分
        max_possible = sum(idf.values()) if idf else 1.0

        # BM25 参数
        k1 = 1.2
        b = 0.75
        avgdl = sum(len(toks) for toks in doc_tokens_list) / n_docs if n_docs else 1.0
        if avgdl == 0:
            avgdl = 1.0

        scored: list[tuple[FusionResult, float]] = []
        for i, result in enumerate(candidates):
            doc_tokens = doc_tokens_list[i]
            doc_len = len(doc_tokens)
            doc_counter = Counter(doc_tokens)

            # TF-IDF 加权得分（BM25 式饱和 + 长度归一化）
            tfidf_sum = 0.0
            for term in set(query_terms):
                tf = doc_counter.get(term, 0)
                if tf == 0:
                    continue
                # BM25 式 TF 饱和 + 长度归一化
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                tfidf_sum += tf_norm * idf.get(term, 0)

            tfidf_score = tfidf_sum / max_possible if max_possible > 0 else 0.0

            # 来源多样性（来自更多检索策略 = 更可靠）
            source_diversity = min(1.0, len(result.sources) / 4.0)

            # RRF 原始得分
            rrf_score = result.score

            # 长度惩罚（过短的信息量不足，过长可能噪声多）
            text_len = len(result.text)
            if text_len < 20:
                length_penalty = 0.3
            elif text_len > 2000:
                length_penalty = 0.7
            else:
                length_penalty = 1.0

            # 综合评分（权重调整：TF-IDF 占比提高，RRF 降低）
            final_score = (
                0.45 * tfidf_score
                + 0.15 * source_diversity
                + 0.25 * rrf_score
                + 0.15 * length_penalty
            )

            scored.append((result, final_score))

        return scored

    # ------------------------------------------------------------------
    # LLM 重排
    # ------------------------------------------------------------------

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[FusionResult],
    ) -> list[tuple[FusionResult, float]]:
        """LLM 重排：让 LLM 对每个 query-doc pair 打分。

        Args:
            query: 查询文本
            candidates: 候选结果列表

        Returns:
            [(FusionResult, new_score), ...]
        """
        async def score_one(result: FusionResult) -> tuple[FusionResult, float]:
            async with self._sem:
                try:
                    prompt = (
                        f"请对以下查询和文档的相关性打分（0.0-1.0，保留两位小数）。\n"
                        f"只返回一个数字，不要其他内容。\n\n"
                        f"查询: {query[:500]}\n\n"
                        f"文档: {result.text[:1000]}"
                    )
                    response = await self._llm.call(prompt, max_tokens=10, temperature=0.0)
                    score = float(response.strip())
                    score = max(0.0, min(1.0, score))
                    return (result, score)
                except Exception as e:
                    logger.warning("LLM 重排打分失败，回退到 RRF 得分: %s", e)
                    return (result, result.score)

        tasks = [score_one(r) for r in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored: list[tuple[FusionResult, float]] = []
        for item in results:
            if isinstance(item, Exception):
                logger.warning("LLM 重排任务异常: %s", item)
                continue
            scored.append(item)

        # 如果 LLM 重排全部失败，回退到启发式
        if not scored:
            logger.warning("LLM 重排全部失败，回退到启发式重排")
            return self._heuristic_rerank(query, candidates)

        return scored

    # ------------------------------------------------------------------
    # Cross-Encoder 重排
    # ------------------------------------------------------------------

    def _load_cross_encoder(self, model_name_or_path: Optional[str]) -> None:
        """加载 cross-encoder 模型。

        Args:
            model_name_or_path: 模型名称或路径。默认使用本地缓存的
                cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        if not model_name_or_path:
            model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, local_files_only=True
            )
            self._cross_encoder = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, local_files_only=True
            )
            self._cross_encoder.eval()
            logger.info(
                "Cross-encoder 模型加载成功: %s", model_name_or_path
            )
        except Exception as e:
            logger.warning(
                "Cross-encoder 模型加载失败，回退到启发式重排: %s", e
            )
            self._strategy = "heuristic"

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[FusionResult],
    ) -> list[tuple[FusionResult, float]]:
        """Cross-Encoder 重排：用预训练模型对 query-doc pair 打分。

        Args:
            query: 查询文本
            candidates: 候选结果列表

        Returns:
            [(FusionResult, new_score), ...]
        """
        import torch

        # 批量构建 query-doc pairs
        pairs = [(query, r.text[:512]) for r in candidates]
        inputs = self._cross_encoder_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._cross_encoder(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze(-1).tolist()

        if not isinstance(scores, list):
            scores = [scores]

        return [(r, s) for r, s in zip(candidates, scores)]

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：中文按字，英文按空格。

        Args:
            text: 输入文本

        Returns:
            分词后的 token 列表
        """
        tokens: list[str] = []
        # 英文部分按空格分词
        for word in text.split():
            if word.strip():
                tokens.append(word.strip())
        # 中文部分按字分词（trigram 已在 FTS5 处理，这里用于覆盖度计算）
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
        return tokens
