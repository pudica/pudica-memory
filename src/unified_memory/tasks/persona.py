"""tasks/persona.py — L3 用户画像层（Persona Distillation）。

从高置信度记忆（authority >= high, trust_score >= 0.7）中提取稳定用户画像，
按主题聚合生成 persona 摘要，存入 persona 表。

核心设计：
- **Distillation 而非直接查询**：不是简单聚合，而是按主题 clustering 后压缩
- **置信度门槛**：只取 authority >= high 且 trust_score >= 0.7 的记忆
- **增量更新**：每次只处理上次蒸馏后新增的记忆
- **自动过期**：persona 条目有 expires_at，支持定时重建
- **降级可用**：LLM 不可用时用启发式（TF-IDF 抽取式摘要）

参考资料：
- TencentDB Memory v2.0 的 L3 Core/Persona 层设计
- Compressor 的 TF-IDF 抽取式摘要思路
"""

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Persona 蒸馏提示词
PERSONA_DISTILL_PROMPT = """# 用户画像蒸馏任务

## 需要蒸馏的记忆片段（主题：{topic}）

{memories_text}

## 任务
从以上记忆中提取关于用户的稳定特征，生成一条用户画像摘要。
要求：
1. 只保留经过多次验证的稳定特征（偏好、习惯、能力、知识领域）
2. 去除临时性、一次性事件
3. 用第三人称表述（"用户"）
4. 每条 100 字以内
5. 标注置信度（high/medium/low）

## 输出格式
```json
{{
  "summary": "用户画像摘要",
  "confidence": "high",
  "evidence_count": 3,
  "key_facts": ["关键事实1", "关键事实2"]
}}
```
"""


class PersonaDistiller:
    """L3 用户画像蒸馏器。

    从高置信度记忆中提取稳定用户画像特征。
    支持 LLM 蒸馏和启发式蒸馏两种模式。

    蒸馏流程：
    1. 查询 authority >= high 且 trust_score >= 0.7 的记忆
    2. 按主题聚类（基于 L1 实体和关键词）
    3. 每个聚类生成一条 persona 摘要
    4. 写入 persona 表
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
            llm: LLM 客户端（可选，提供后使用 LLM 蒸馏）
            settings: 配置字典
                - min_authority: 最小 authority 级别（默认 'high'）
                - min_trust_score: 最小信任分数（默认 0.7）
                - max_persona_age_days: persona 最大存活天数（默认 90）
                - min_evidence: 单条 persona 最少证据条数（默认 2）
        """
        self._pool = pool
        self._chroma = chroma
        self._llm = llm
        self._settings = settings or {}
        self._min_authority = self._settings.get("min_authority", "high")
        self._min_trust_score = self._settings.get("min_trust_score", 0.7)
        self._max_persona_age = self._settings.get("max_persona_age_days", 90) * 86400
        self._min_evidence = self._settings.get("min_evidence", 2)

    async def distill(self, force: bool = False) -> dict:
        """执行一次完整的画像蒸馏。

        Args:
            force: 强制全量蒸馏（忽略上次蒸馏时间）

        Returns:
            蒸馏统计字典
        """
        logger.info("开始 Persona 蒸馏（force=%s）", force)

        # 1. 查询高置信度记忆
        high_conf_memories = await self._query_high_confidence_memories()
        if not high_conf_memories:
            logger.info("无高置信度记忆，跳过蒸馏")
            return {"distilled": 0, "personas": 0, "total_memories": 0}

        # 2. 过滤已处理的记忆（增量模式）
        if not force:
            last_ts = await self._get_last_distill_timestamp()
            if last_ts:
                high_conf_memories = [
                    m for m in high_conf_memories
                    if m.get("created_at", 0) > last_ts
                ]

        if not high_conf_memories:
            logger.info("无新记忆需要蒸馏")
            return {"distilled": 0, "personas": 0, "total_memories": 0}

        # 3. 按主题聚类
        clusters = self._cluster_by_topic(high_conf_memories)
        logger.info("Persona 聚类: %d 组, 共 %d 条记忆", len(clusters), len(high_conf_memories))

        # 4. 逐聚类蒸馏
        personas = []
        for cluster_key, memories in clusters.items():
            if len(memories) < self._min_evidence:
                logger.debug("聚类 '%s' 证据不足（%d < %d），跳过",
                             cluster_key, len(memories), self._min_evidence)
                continue

            persona = await self._distill_cluster(cluster_key, memories)
            if persona:
                personas.append(persona)

        # 5. 写入 persona 表
        written = await self._write_personas(personas)

        # 6. 更新蒸馏时间戳
        await self._update_distill_timestamp()

        stats = {
            "distilled": len(high_conf_memories),
            "personas": written,
            "total_memories": len(high_conf_memories),
            "clusters": len(clusters),
            "timestamp": time.time(),
        }

        logger.info(
            "Persona 蒸馏完成: %d 条记忆 → %d 条画像（%d 个聚类）",
            len(high_conf_memories), written, len(clusters),
        )
        return stats

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    async def _query_high_confidence_memories(self) -> list[dict]:
        """查询高置信度记忆（authority >= high 且 trust_score >= 0.7）。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                """SELECT id, content, wing, room, source, fact_type,
                          authority, trust_score, summary, metadata, created_at
                   FROM memories
                   WHERE authority >= ? AND trust_score >= ?
                   ORDER BY created_at DESC
                   LIMIT 500""",
                (self._min_authority, self._min_trust_score),
            )
            rows = await cursor.fetchall()
            memories = []
            for row in rows:
                mem = dict(row)
                # 解析 metadata
                try:
                    mem["metadata"] = json.loads(mem.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    mem["metadata"] = {}
                memories.append(mem)
            return memories
        finally:
            await self._pool.release(conn)

    async def _get_last_distill_timestamp(self) -> float:
        """获取上次蒸馏时间戳。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT value FROM config WHERE key = 'persona_last_distill'"
            )
            row = await cursor.fetchone()
            if row:
                return float(row["value"])
            return 0.0
        except (ValueError, TypeError):
            return 0.0
        finally:
            await self._pool.release(conn)

    async def _update_distill_timestamp(self) -> None:
        """更新蒸馏时间戳。"""
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES ('persona_last_distill', ?)",
                (str(time.time()),),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # 主题聚类
    # ------------------------------------------------------------------

    def _cluster_by_topic(self, memories: list[dict]) -> dict[str, list[dict]]:
        """按主题聚类记忆。

        优先使用 L1 实体信息，兜底用关键词匹配。

        Returns:
            {topic_key: [memory, ...]}
        """
        clusters: dict[str, list[dict]] = {}

        for mem in memories:
            # 尝试从 metadata 中获取 L1 实体
            md = mem.get("metadata", {})
            entities = md.get("entities", []) if isinstance(md, dict) else []

            if entities and isinstance(entities, list):
                # 用第一个实体名作为主题
                main_entity = entities[0] if entities else "general"
                if isinstance(main_entity, dict):
                    main_entity = main_entity.get("name", "general")
                topic = str(main_entity)
            else:
                # 兜底：从内容提取主题关键词
                topic = self._extract_topic_from_content(mem.get("content", ""))

            if topic not in clusters:
                clusters[topic] = []
            clusters[topic].append(mem)

        return clusters

    @staticmethod
    def _extract_topic_from_content(content: str) -> str:
        """从内容提取主题关键词。

        取内容中第一个有意义的实体或关键词。
        """
        if not content:
            return "general"

        # 尝试匹配常见领域关键词
        domain_patterns = [
            (r"(中医|中药|针灸|脉诊|方剂|体质)", "中医"),
            (r"(编程|代码|Python|开发|架构|算法)", "编程"),
            (r"(投资|股票|基金|理财|A股|市场)", "投资"),
            (r"(林草|林业|草原|生态|碳汇|GEF)", "林草"),
            (r"(玄学|八字|紫微|六爻|奇门|风水)", "玄学"),
            (r"(AI|人工智能|大模型|Agent|LLM|模型)", "AI"),
            (r"(健身|运动|跑步|训练|营养)", "健身"),
            (r"(阅读|读书|书|文章|论文)", "阅读"),
        ]
        for pattern, domain in domain_patterns:
            if re.search(pattern, content):
                return domain

        # 取第一个有意义的词（4字以上中文词）
        words = re.findall(r"[\u4e00-\u9fff]{4,}", content)
        if words:
            return words[0][:10]

        return "general"

    # ------------------------------------------------------------------
    # 蒸馏执行
    # ------------------------------------------------------------------

    async def _distill_cluster(self, topic: str, memories: list[dict]) -> Optional[dict]:
        """蒸馏单个聚类。

        Args:
            topic: 主题标签
            memories: 该主题的记忆列表

        Returns:
            蒸馏结果字典，失败返回 None
        """
        # 构建记忆文本
        memories_text = "\n---\n".join(
            f"[{m['created_at']}] {m['content']}"
            for m in memories
        )

        # 生成 persona 摘要
        if self._llm is not None:
            persona_result = await self._llm_distill(topic, memories_text)
        else:
            persona_result = self._heuristic_distill(memories)

        if not persona_result:
            return None

        now = time.time()
        return {
            "id": str(uuid4()),
            "topic": topic,
            "summary": persona_result.get("summary", ""),
            "confidence": persona_result.get("confidence", "medium"),
            "evidence_count": len(memories),
            "source_memory_ids": json.dumps([m["id"] for m in memories], ensure_ascii=False),
            "key_facts": json.dumps(persona_result.get("key_facts", []), ensure_ascii=False),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + self._max_persona_age,
            "metadata": json.dumps({
                "method": "llm" if self._llm else "heuristic",
                "topic": topic,
                "memory_count": len(memories),
            }, ensure_ascii=False),
        }

    async def _llm_distill(self, topic: str, memories_text: str) -> Optional[dict]:
        """使用 LLM 生成 persona 摘要。"""
        prompt = PERSONA_DISTILL_PROMPT.format(
            topic=topic,
            memories_text=memories_text[:4000],
        )
        try:
            response = await self._llm.call(
                prompt,
                response_format={"type": "json_object"},
            )
            result = json.loads(response)
            result.setdefault("summary", "")
            result.setdefault("confidence", "medium")
            result.setdefault("evidence_count", 0)
            result.setdefault("key_facts", [])
            if not result["summary"]:
                return None
            return result
        except Exception as e:
            logger.warning("LLM 蒸馏失败，回退到启发式: %s", e)
            return self._heuristic_distill_from_text(memories_text)

    def _heuristic_distill(self, memories: list[dict]) -> Optional[dict]:
        """启发式蒸馏：抽取式摘要 + 置信度判定。"""
        memories_text = "\n---\n".join(
            f"[{m['created_at']}] {m['content']}" for m in memories
        )
        return self._heuristic_distill_from_text(memories_text, memories)

    def _heuristic_distill_from_text(
        self, memories_text: str, memories: Optional[list[dict]] = None
    ) -> Optional[dict]:
        """从文本生成启发式 persona 摘要。

        基于 Compressor 的 TF-IDF 抽取式摘要，但更注重：
        1. 稳定性标记（信号词如"总是"、"习惯"、"偏好"）
        2. 置信度来自证据数量和一致性
        """
        # 收集所有句子
        sentences: list[tuple[str, int]] = []  # (sentence, source_index)
        if memories:
            for idx, m in enumerate(memories):
                content = m.get("content", "")
                for sent in self._split_sentences(content):
                    sent = sent.strip()
                    if len(sent) >= 8:
                        sentences.append((sent, idx))
        else:
            for sent in self._split_sentences(memories_text):
                sent = sent.strip()
                if len(sent) >= 8:
                    sentences.append((sent, 0))

        if not sentences:
            return {
                "summary": memories_text[:200],
                "confidence": "low",
                "evidence_count": len(memories) if memories else 1,
                "key_facts": [],
            }

        # 构建 TF-IDF 模型
        n_sents = len(sentences)
        doc_tokens_list = [self._tokenize_for_persona(s[0]) for s in sentences]

        doc_freq: Counter = Counter()
        for tokens in doc_tokens_list:
            for term in set(tokens):
                doc_freq[term] += 1

        idf: dict[str, float] = {}
        for term in set().union(*[set(t) for t in doc_tokens_list]):
            idf[term] = math.log((n_sents + 1) / (doc_freq.get(term, 0) + 1)) + 1

        # 打分：TF-IDF + 稳定性信号 + 位置
        scored_sentences: list[tuple[float, int, str]] = []
        for i, (sent, _) in enumerate(sentences):
            tokens = doc_tokens_list[i]
            token_counter = Counter(tokens)
            doc_len = len(tokens)

            tfidf_sum = sum(tf * idf.get(term, 0) for term, tf in token_counter.items())
            tfidf_score = tfidf_sum / max(doc_len, 1)

            # 稳定性信号加成
            stability_bonus = self._stability_signal(sent)

            # 位置加成
            position_bonus = 1.0 - 0.3 * (i / n_sents)

            final_score = 0.4 * tfidf_score + 0.4 * stability_bonus + 0.2 * position_bonus
            scored_sentences.append((final_score, i, sent))

        # 选 top-k 句子
        target_count = max(2, min(5, 200 // 30))
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_sentences[:target_count]

        top_k.sort(key=lambda x: x[1])
        summary = "；".join(s[2] for s in top_k)
        if len(summary) > 200:
            summary = summary[:197] + "..."

        # 置信度：基于证据数量和稳定性信号密度
        avg_stability = sum(
            self._stability_signal(s[2]) for s in scored_sentences[:5]
        ) / min(5, len(scored_sentences))
        evidence_count = len(memories) if memories else 1
        if evidence_count >= 5 and avg_stability > 0.5:
            confidence = "high"
        elif evidence_count >= 3 and avg_stability > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        # 关键事实
        key_facts = []
        for score, _, sent in scored_sentences[:5]:
            if self._stability_signal(sent) > 0.3:
                key_facts.append(sent[:60])

        return {
            "summary": summary,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "key_facts": key_facts[:5],
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    async def _write_personas(self, personas: list[dict]) -> int:
        """写入 persona 到数据库。

        使用 INSERT OR REPLACE 按 topic 去重。
        """
        if not personas:
            return 0

        conn = await self._pool.acquire()
        try:
            # 先建表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence TEXT DEFAULT 'medium',
                    evidence_count INTEGER DEFAULT 1,
                    source_memory_ids TEXT DEFAULT '[]',
                    key_facts TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    UNIQUE(topic)
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_personas_topic ON personas(topic)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_personas_confidence ON personas(confidence)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_personas_expires ON personas(expires_at)
            """)

            written = 0
            for p in personas:
                try:
                    await conn.execute(
                        """INSERT OR REPLACE INTO personas
                           (id, topic, summary, confidence, evidence_count,
                            source_memory_ids, key_facts, metadata,
                            created_at, updated_at, expires_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            p["id"], p["topic"], p["summary"], p["confidence"],
                            p["evidence_count"], p["source_memory_ids"],
                            p["key_facts"], p["metadata"],
                            p["created_at"], p["updated_at"], p["expires_at"],
                        ),
                    )
                    written += 1
                except Exception as e:
                    logger.warning("写入 persona '%s' 失败: %s", p.get("topic", "?"), e)

            await conn.commit()
            logger.info("写入 %d 条 persona 到数据库", written)
            return written
        finally:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # 过期清理
    # ------------------------------------------------------------------

    async def clean_expired(self) -> int:
        """清理过期的 persona 条目。

        Returns:
            删除的条目数
        """
        now = time.time()
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "DELETE FROM personas WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted = cursor.rowcount
            await conn.commit()
            if deleted:
                logger.info("清理 %d 条过期 persona", deleted)
            return deleted
        finally:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    async def get_personas(
        self, topic: Optional[str] = None, min_confidence: str = "low"
    ) -> list[dict]:
        """获取用户画像。

        Args:
            topic: 可选主题过滤
            min_confidence: 最低置信度过滤

        Returns:
            画像列表
        """
        confidence_levels = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_levels.get(min_confidence, 0)

        conn = await self._pool.acquire()
        try:
            if topic:
                cursor = await conn.execute(
                    "SELECT * FROM personas WHERE topic = ? ORDER BY confidence DESC, evidence_count DESC",
                    (topic,),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM personas ORDER BY confidence DESC, evidence_count DESC"
                )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                conf = row.get("confidence", "low")
                if confidence_levels.get(conf, 0) >= min_level:
                    results.append({
                        "id": row["id"],
                        "topic": row["topic"],
                        "summary": row["summary"],
                        "confidence": conf,
                        "evidence_count": row["evidence_count"],
                        "key_facts": json.loads(row.get("key_facts", "[]")),
                        "created_at": row["created_at"],
                        "expires_at": row.get("expires_at"),
                    })
            return results
        finally:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """分割句子。"""
        pattern = r"[。！？；.!?;]\s*"
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize_for_persona(text: str) -> list[str]:
        """画像用分词：中文按字+bigram，英文按空格。"""
        tokens: list[str] = []
        for word in text.split():
            w = word.strip().lower()
            if w:
                tokens.append(w)
        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        tokens.extend(chinese_chars)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    @staticmethod
    def _stability_signal(text: str) -> float:
        """检测稳定性信号密度（0-1）。

        稳定性信号表示该句描述的是稳定特征而非临时事件：
        - 习惯性表达（总是、经常、习惯、喜欢、偏好）
        - 能力描述（擅长、精通、熟悉、能、会）
        - 长期状态（一直、长期、多年来、持续）
        - 知识领域（研究领域、专业方向、关注）
        """
        if not text:
            return 0.0
        signals = 0
        patterns = [
            r"(总是|经常|通常|习惯|喜欢|偏好|倾向于|爱)",
            r"(擅长|精通|熟悉|熟练掌握|能|会|懂得)",
            r"(一直|长期|多年来|持续|始终|从未)",
            r"(研究|专业|领域|方向|关注|感兴趣)",
            r"(不[吃喝用]|避免|忌|戒|排斥)",
            r"(信任|信赖|认可|推荐|首选)",
        ]
        for pattern in patterns:
            signals += len(re.findall(pattern, text))
        return min(1.0, signals / 3.0)