"""tasks/compression.py — 记忆压缩任务（MemPalace AAAK 启发）。

将旧记忆合并为压缩摘要，减少存储占用并提升检索效率。
压缩策略：
  1. 按 wing/room 分组
  2. 在每个分组内，按时间窗口聚合同主题记忆
  3. 用 LLM 或启发式生成压缩摘要
  4. 原始记忆标记为已归档，压缩摘要写入 compressed_memories 表

参考 MemPalace AAAK 的压缩理念：30x 压缩比，保留关键信息。

v3.1 升级：LLM 不可用时的 fallback 从简单截断拼接升级为抽取式摘要。
  - 按句子分割所有记忆内容
  - 用 TF-IDF 对句子打分（以全量记忆为语料库）
  - 选取得分最高的 top-k 句子，按原始顺序重排
  - 自动提取关键事实（数字、日期、人名、因果关系）
"""

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# 压缩提示词
COMPRESSION_PROMPT = """# 记忆压缩任务

## 需要压缩的记忆片段

{memories_text}

## 任务
将以上记忆合并为一条压缩摘要，要求：
1. 保留所有关键事实（人名、地点、时间、因果关系）
2. 去除重复信息
3. 按主题组织，逻辑清晰
4. 压缩到 300 字以内
5. 保留原始记忆的时间范围

## 输出格式
```json
{{
  "summary": "压缩摘要文本",
  "key_facts": ["关键事实1", "关键事实2"],
  "time_range": {{"start": "ISO8601", "end": "ISO8601"}}
}}
```"""


class Compressor:
    """记忆压缩器。

    将旧记忆按分组压缩为摘要，减少存储和检索开销。
    支持 LLM 压缩和启发式压缩两种模式。

    压缩流程：
    1. 查询超过 min_age_seconds 的旧记忆
    2. 按 wing/room 分组
    3. 每组内按时间排序，窗口聚合
    4. LLM/启发式生成压缩摘要
    5. 写入 compressed_memories 表
    6. 原始记忆可以删除或保留（取决于 keep_ratio）
    """

    def __init__(
        self,
        pool: Any,
        chroma: Any,
        llm: Any = None,
        settings: Optional[dict] = None,
    ):
        """
        Args:
            pool: SQLitePool 实例
            chroma: ChromaStore 实例
            llm: LLM 客户端（可选，提供后使用 LLM 压缩）
            settings: 配置字典
                - min_age_seconds: 触发压缩的最小记忆年龄
                - batch_size: 单次压缩最大处理条数
                - keep_ratio: 压缩后保留的原始记忆比例
        """
        self._pool = pool
        self._chroma = chroma
        self._llm = llm
        self._settings = settings or {}
        self._min_age = self._settings.get("min_age_seconds", 604800)
        self._batch_size = self._settings.get("batch_size", 50)
        self._keep_ratio = self._settings.get("keep_ratio", 0.2)

    async def compress(self) -> dict:
        """执行一次完整的压缩流程。

        Returns:
            压缩统计字典
        """
        logger.info("开始记忆压缩任务")

        # 1. 查询旧记忆
        old_memories = await self._query_old_memories()
        if not old_memories:
            logger.info("无待压缩记忆")
            return {"compressed": 0, "archived": 0, "groups": 0}

        # 2. 按 wing/room 分组
        groups = self._group_memories(old_memories)
        logger.info("压缩分组: %d 组, 共 %d 条记忆", len(groups), len(old_memories))

        # 3. 逐组压缩
        total_compressed = 0
        total_archived = 0
        compression_logs: list[dict] = []

        for group_key, memories in groups.items():
            if len(memories) < 3:
                # 少于 3 条不压缩
                continue

            result = await self._compress_group(group_key, memories)
            if result:
                total_compressed += result["source_count"]
                total_archived += result["archived_count"]
                compression_logs.append(result)

        # 4. 清理已压缩的 ChromaDB 向量
        if total_archived > 0:
            await self._cleanup_chroma_vectors(compression_logs)

        stats = {
            "compressed": total_compressed,
            "archived": total_archived,
            "groups": len(compression_logs),
            "logs": compression_logs,
            "timestamp": time.time(),
        }

        # 5. 记录压缩日志
        await self._log_compression(stats)

        logger.info(
            "记忆压缩完成: %d 条压缩为 %d 条摘要, %d 条原始记忆归档",
            total_compressed, len(compression_logs), total_archived,
        )
        return stats

    # ------------------------------------------------------------------
    # 查询与分组
    # ------------------------------------------------------------------

    async def _query_old_memories(self) -> list[dict]:
        """查询超过年龄阈值的记忆。"""
        cutoff = time.time() - self._min_age
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                """SELECT id, content, wing, room, source, fact_type, metadata, created_at
                   FROM memories
                   WHERE created_at < ?
                   ORDER BY wing, room, created_at
                   LIMIT ?""",
                (cutoff, self._batch_size),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await self._pool.release(conn)

    def _group_memories(self, memories: list[dict]) -> dict[str, list[dict]]:
        """按 wing/room 分组记忆。

        Args:
            memories: 记忆列表

        Returns:
            {"wing:room": [memory, ...]}
        """
        groups: dict[str, list[dict]] = {}
        for mem in memories:
            key = f"{mem.get('wing', 'default')}:{mem.get('room', 'general')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(mem)
        return groups

    # ------------------------------------------------------------------
    # 压缩执行
    # ------------------------------------------------------------------

    async def _compress_group(self, group_key: str, memories: list[dict]) -> Optional[dict]:
        """压缩单个分组。

        Args:
            group_key: "wing:room" 格式的分组键
            memories: 该分组的记忆列表

        Returns:
            压缩结果字典，失败返回 None
        """
        wing, room = group_key.split(":", 1) if ":" in group_key else ("default", "general")

        # 构建记忆文本
        memories_text = "\n---\n".join(
            f"[{m['created_at']}] {m['content']}" for m in memories
        )

        # 生成压缩摘要
        if self._llm is not None:
            summary_result = await self._llm_compress(memories_text)
        else:
            summary_result = self._heuristic_compress(memories)

        if not summary_result:
            return None

        # 写入压缩记忆表
        compressed_id = str(uuid4())
        source_ids = [m["id"] for m in memories]
        now = time.time()

        conn = await self._pool.acquire()
        try:
            await conn.execute(
                """INSERT INTO compressed_memories (id, summary, source_memory_ids, source_count, wing, room, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (compressed_id, summary_result["summary"],
                 json.dumps(source_ids, ensure_ascii=False),
                 len(memories), wing, room,
                 json.dumps({
                     "key_facts": summary_result.get("key_facts", []),
                     "time_range": summary_result.get("time_range", {}),
                     "compression_method": "llm" if self._llm else "heuristic",
                 }, ensure_ascii=False),
                 now),
            )
            await conn.commit()
        except Exception as e:
            logger.error("写入压缩记忆失败: %s", e)
            return None
        finally:
            await self._pool.release(conn)

        # 决定哪些原始记忆需要归档
        keep_count = max(1, int(len(memories) * self._keep_ratio))
        # 保留最近的 keep_count 条，其余归档
        to_archive = memories[:-keep_count] if keep_count < len(memories) else []
        archived_ids = [m["id"] for m in to_archive]

        # 从 memories 表删除已归档的记忆
        if archived_ids:
            await self._archive_memories(archived_ids)

        return {
            "compressed_id": compressed_id,
            "wing": wing,
            "room": room,
            "source_count": len(memories),
            "archived_count": len(archived_ids),
            "archived_ids": archived_ids,
            "summary_preview": summary_result["summary"][:200],
        }

    async def _llm_compress(self, memories_text: str) -> Optional[dict]:
        """使用 LLM 生成压缩摘要。"""
        prompt = COMPRESSION_PROMPT.format(memories_text=memories_text[:4000])
        try:
            response = await self._llm.call(
                prompt,
                response_format={"type": "json_object"},
            )
            result = json.loads(response)
            result.setdefault("summary", "")
            result.setdefault("key_facts", [])
            result.setdefault("time_range", {})
            if not result["summary"]:
                return None
            return result
        except Exception as e:
            logger.warning("LLM 压缩失败，回退到启发式: %s", e)
            return self._heuristic_compress_from_text(memories_text)

    def _heuristic_compress(self, memories: list[dict]) -> Optional[dict]:
        """启发式压缩：拼接摘要 + 提取关键事实。"""
        memories_text = "\n---\n".join(
            f"[{m['created_at']}] {m['content']}" for m in memories
        )
        return self._heuristic_compress_from_text(memories_text, memories)

    def _heuristic_compress_from_text(
        self, memories_text: str, memories: Optional[list[dict]] = None
    ) -> Optional[dict]:
        """从文本生成启发式压缩摘要（抽取式摘要）。

        v3.1 升级：从简单截断拼接改为 TF-IDF 抽取式摘要。
        1. 将所有记忆内容按句子分割
        2. 以全量句子为语料库构建 TF-IDF 模型
        3. 对每句打分（TF-IDF 权重 + 位置加成 + 实体密度加成）
        4. 选取得分最高的 top-k 句子，按原始顺序重排
        5. 自动提取关键事实（含数字、日期、人名的句子）

        Args:
            memories_text: 拼接后的记忆文本
            memories: 原始记忆列表（可选，提供时用于时间范围提取）

        Returns:
            {"summary": str, "key_facts": list, "time_range": dict}
        """
        # 收集所有句子
        sentences: list[tuple[str, int]] = []  # (sentence, source_index)
        if memories:
            for idx, m in enumerate(memories):
                content = m.get("content", "")
                for sent in self._split_sentences(content):
                    sent = sent.strip()
                    if len(sent) >= 5:  # 过滤过短的碎片
                        sentences.append((sent, idx))
        else:
            # 从文本分割
            for sent in self._split_sentences(memories_text):
                sent = sent.strip()
                if len(sent) >= 5:
                    sentences.append((sent, 0))

        if not sentences:
            # fallback: 截断
            return {
                "summary": memories_text[:300],
                "key_facts": [],
                "time_range": {},
            }

        # 构建 TF-IDF 模型
        n_sents = len(sentences)
        doc_tokens_list = [self._tokenize_for_summary(s[0]) for s in sentences]

        # 文档频率
        df: Counter = Counter()
        for tokens in doc_tokens_list:
            for term in set(tokens):
                df[term] += 1

        # IDF
        idf: dict[str, float] = {}
        for term in set().union(*[set(t) for t in doc_tokens_list]):
            idf[term] = math.log((n_sents + 1) / (df.get(term, 0) + 1)) + 1

        # 对每句打分
        scored_sentences: list[tuple[float, int, str]] = []  # (score, orig_idx, sentence)
        for i, (sent, _) in enumerate(sentences):
            tokens = doc_tokens_list[i]
            token_counter = Counter(tokens)
            doc_len = len(tokens)

            # TF-IDF 总分
            tfidf_sum = 0.0
            for term, tf in token_counter.items():
                tfidf_sum += tf * idf.get(term, 0)

            # 归一化（按文档长度）
            tfidf_score = tfidf_sum / max(doc_len, 1)

            # 位置加成：靠前的句子更可能包含概述信息
            position_bonus = 1.0 - 0.3 * (i / n_sents)

            # 实体密度加成：含数字/日期/人名的句子更重要
            entity_density = self._entity_density(sent)

            # 综合得分
            final_score = (
                0.5 * tfidf_score
                + 0.2 * position_bonus
                + 0.3 * entity_density
            )
            scored_sentences.append((final_score, i, sent))

        # 选 top-k 句子（目标压缩到 300 字以内）
        target_count = max(3, min(10, 300 // 30))  # 估计每句约 30 字
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_sentences[:target_count]

        # 按原始顺序重排
        top_k.sort(key=lambda x: x[1])

        summary = "；".join(s[2] for s in top_k)
        if len(summary) > 300:
            summary = summary[:297] + "..."

        # 提取关键事实：得分最高的句子中含数字/日期的
        key_facts: list[str] = []
        for score, _, sent in scored_sentences[:5]:
            if self._entity_density(sent) > 0.3:
                key_facts.append(sent[:80])

        # 时间范围
        time_range = {}
        if memories:
            times = [m["created_at"] for m in memories if "created_at" in m]
            if times:
                time_range = {"start": min(times), "end": max(times)}

        return {
            "summary": summary,
            "key_facts": key_facts[:5],
            "time_range": time_range,
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """将文本分割为句子。

        支持中文标点（。！？；）和英文标点（.!?;）。
        """
        # 按中英文句号/问号/感叹号/分号分割
        pattern = r'[。！？；.!?;]\s*'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize_for_summary(text: str) -> list[str]:
        """摘要用分词：中文按字+bigram，英文按空格。

        比 reranker 的分词多了 bigram，能更好地捕捉中文短语。
        """
        tokens: list[str] = []
        # 英文部分
        for word in text.split():
            w = word.strip().lower()
            if w:
                tokens.append(w)
        # 中文部分：单字 + bigram
        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        tokens.extend(chinese_chars)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    @staticmethod
    def _entity_density(text: str) -> float:
        """计算句子的实体密度（0-1）。

        检测数字、日期、人名标记等高信息量元素。
        """
        if not text:
            return 0.0
        signals = 0
        # 数字
        signals += len(re.findall(r'\d+', text))
        # 日期相关词
        signals += len(re.findall(r'年|月|日|时|分|秒|周|昨天|今天|明天|上周|下周', text))
        # 人名标记（X先生/X女士/小X/老X）
        signals += len(re.findall(r'先生|女士|小.|老.|博士|教授|医生', text))
        # 因果关系词
        signals += len(re.findall(r'因为|所以|导致|引起|由于|因此|结果|原因', text))
        # 金额
        signals += len(re.findall(r'元|块|万|亿|￥|¥|\$', text))
        # 归一化到 0-1
        return min(1.0, signals / 5.0)

    # ------------------------------------------------------------------
    # 归档与清理
    # ------------------------------------------------------------------

    async def _archive_memories(self, memory_ids: list[str]) -> None:
        """归档原始记忆（从 memories 表删除，FTS 触发器自动同步）。"""
        if not memory_ids:
            return
        conn = await self._pool.acquire()
        try:
            placeholders = ", ".join("?" for _ in memory_ids)
            await conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            )
            await conn.commit()
            logger.debug("归档 %d 条原始记忆", len(memory_ids))
        except Exception as e:
            logger.error("归档记忆失败: %s", e)
        finally:
            await self._pool.release(conn)

    async def _cleanup_chroma_vectors(self, logs: list[dict]) -> None:
        """清理已归档记忆在 ChromaDB 中的向量。"""
        all_ids: list[str] = []
        for log in logs:
            all_ids.extend(log.get("archived_ids", []))
        if not all_ids:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: self._chroma.delete_batch(all_ids)
            )
        except AttributeError:
            # ChromaStore 可能没有 delete_batch 方法
            for mid in all_ids:
                try:
                    await loop.run_in_executor(
                        None, lambda m=mid: self._chroma._collection.delete(ids=[m])
                    )
                except Exception as e:
                    logger.warning("清理向量失败 %s: %s", mid, e)
        except Exception as e:
            logger.warning("批量清理向量失败: %s", e)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    async def _log_compression(self, stats: dict) -> None:
        """记录压缩日志。"""
        conn = await self._pool.acquire()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_logs (
                    id TEXT PRIMARY KEY,
                    result TEXT,
                    created_at REAL
                )
            """)
            await conn.execute(
                "INSERT INTO compression_logs (id, result, created_at) VALUES (?, ?, ?)",
                (str(uuid4()), json.dumps(stats, ensure_ascii=False), time.time()),
            )
            await conn.commit()
        except Exception as e:
            logger.warning("写入压缩日志失败: %s", e)
        finally:
            await self._pool.release(conn)
