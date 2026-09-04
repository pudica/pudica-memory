"""tasks/reflect.py — Reflect 任务：单次 LLM 调用，三级检索，批量更新。

参考文档 7.4 节 Reflector 实现 + hindsight reflect/agent.py 的完整流程。
v3.4.0 新增：disposition 语气推断输出。
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Reflect 提示词
# 语言保持规则参考（Hindsight #3181）：当源记忆为中文时，洞察/描述/建议
# 必须以中文输出。本 prompt 本身为中文，此规则作为防御性约束，防止 prompt
# 未来被改写成英文时多语言模型将中文源事实翻译成英文观察。
_LANGUAGE_RULE = """
## 输出语言

用源记忆的语言写出每个洞察、缺口、跨场景关联和心智模型更新的描述——永远不要翻译它们。当同一批源事实混用多种语言时，以多数派语言为准。专有名词、标识符、单位保持原样不译。
"""

REFLECT_PROMPT = """# 反思任务

## 上下文
你正在回顾最近的记忆片段。以下是相关的检索结果：

{context}

## 当前心智模型
以下是已知的用户偏好和信念：

{mental_models}

{language_rule}
## 任务
根据以上信息，执行以下分析：

### 1. 关键洞察
提取最重要的发现、模式或趋势。

### 2. 知识缺口
识别信息缺失或需要进一步调查的领域。

### 3. 跨场景关联
识别不同场景之间的隐含联系。

### 4. 更新建议
对知识图谱的具体更新建议（实体、关系）。

### 5. 心智模型更新
基于记忆片段，更新用户偏好/信念/行为模式。
格式: {{"category": "preference|belief|behavior|knowledge_level", "key": "...", "value": "...", "confidence_delta": 0.1}}

### 6. 语气/倾向推断（Disposition）
基于记忆片段，推断用户的当前倾向（语气、情绪、交流风格）。
格式: {{"disposition": "positive|neutral|negative|analytical|urgent", "confidence": 0.0~1.0, "reason": "..."}}

## 输出格式
```json
{{
  "insights": [
    {{"title": "...", "description": "..."}}
  ],
  "gaps": [
    {{"area": "...", "question": "..."}}
  ],
  "cross_scene_links": [
    {{"entity_a": "...", "relation": "...", "entity_b": "..."}}
  ],
  "kg_updates": [
    {{"action": "add_entity|add_relation", "subject": "...", "predicate": "...", "object": "..."}}
  ],
  "mental_model_updates": [
    {{"category": "preference", "key": "...", "value": "...", "confidence_delta": 0.1}}
  ],
  "disposition": {{
    "disposition": "positive|neutral|negative|analytical|urgent",
    "confidence": 0.0~1.0,
    "reason": "..."
  }}
}}"""


class Reflector:
    """Reflect 任务：单次 LLM 调用，三级检索，批量更新。

    参考文档 7.4 节和 hindsight reflect/agent.py 的完整流程：
    1. 优先从最近场景检索
    2. 三级检索：entity → time → random（类似 hindsight 的 pockets 检索）
    3. 单次 LLM 调用生成洞察
    4. 批量更新知识图谱
    v3.4.0：新增 disposition 输出
    """

    def __init__(
        self,
        llm: Any,
        kg: Any,
        chroma: Any,
        pool: Any,
        mental_models: Any = None,
        settings: Optional[dict] = None,
    ):
        self._llm = llm
        self._kg = kg
        self._chroma = chroma
        self._pool = pool
        self._mental_models = mental_models
        self._settings = settings or {}
        self._extract_window = self._settings.get("extract_window", 3600 * 24 * 7)
        self._max_context_chars = self._settings.get("max_context_chars", 4000)

    async def reflect(self) -> dict:
        """执行一次完整的 Reflect 流程。

        Returns:
            {"insights": [...], "gaps": [...], "cross_scene_links": [...],
             "kg_updates": [...], "disposition": {...}, "stats": {...}}
        """
        logger.info("开始 Reflect 任务")

        # 1. 三级检索
        context = await self._retrieve_context()

        if not context:
            logger.warning("Reflect 检索无上下文，跳过")
            return {"insights": [], "gaps": [], "cross_scene_links": [], "kg_updates": [], "disposition": {}, "stats": {}}

        # 2. 单次 LLM 调用
        result = await self._call_llm(context)

        # 3. 批量更新知识图谱
        stats = await self._apply_kg_updates(result.get("kg_updates", []))

        # 4. 更新心智模型（Hindsight: Mental Models）
        mm_stats = await self._apply_mental_model_updates(
            result.get("mental_model_updates", [])
        )

        # 5. 信念衰减（定期降低旧信念置信度）
        if self._mental_models:
            await self._mental_models.decay_beliefs(decay_factor=0.98)

        # 6. 提取 disposition
        disposition = result.get("disposition", {})
        if not isinstance(disposition, dict):
            disposition = {}

        result["stats"] = {
            "context_size": len(context),
            "kg_updates_applied": stats,
            "mental_model_updates": mm_stats,
        }
        result["disposition"] = disposition

        # 7. 写入 reflect 记录
        await self._log_reflect(result)

        logger.info("Reflect 完成: %d 洞察, %d KG 更新, disposition=%s",
                     len(result.get("insights", [])), stats,
                     disposition.get("disposition", "none"))
        return result

    async def _retrieve_context(self) -> str:
        """三级检索：entity → time → random。"""
        parts: list[str] = []

        # 一级：活跃实体
        hot_entities = await self._kg.get_hot_entities(min_relations=2, hours=24)
        for entity in hot_entities[:5]:
            context = await self._kg.get_entity_context(entity.name)
            if context["relations"]:
                rel_str = "; ".join(
                    f"{r.subject} {r.predicate} {r.object}"
                    for r in context["relations"][:3]
                )
                parts.append(f"实体: {entity.name} ({entity.entity_type}) [{rel_str}]")

        # 二级 + 三级：合并到单次连接中获取
        now = time.time()
        week_ago = now - self._extract_window
        conn = await self._pool.acquire()
        try:
            # 二级：最近场景
            cursor = await conn.execute(
                "SELECT name, summary, created_at FROM scenes WHERE created_at > ? ORDER BY updated_at DESC LIMIT 10",
                (week_ago,),
            )
            rows = await cursor.fetchall()
            for row in rows:
                summary = row["summary"] or ""
                parts.append(f"场景: {row['name']} ({row['created_at']}) [{summary[:200]}]")

            # 三级：随机采样（同一连接内完成）
            cursor = await conn.execute(
                "SELECT content FROM memories ORDER BY RANDOM() LIMIT 5"
            )
            rows = await cursor.fetchall()
            for row in rows:
                parts.append(f"记忆: {row['content'][:200]}")
        finally:
            await self._pool.release(conn)

        return "\n".join(parts)[:self._max_context_chars]

    async def _call_llm(self, context: str) -> dict:
        """单次 LLM 调用。"""
        mental_models_text = ""
        if self._mental_models:
            mental_models_text = await self._mental_models.format_for_context(max_items=15)
        if not mental_models_text:
            mental_models_text = "（暂无已知心智模型）"

        prompt = REFLECT_PROMPT.format(context=context, mental_models=mental_models_text, language_rule=_LANGUAGE_RULE)

        for attempt in range(self._max_retries):
            try:
                response = await self._llm.call(
                    prompt,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response)
                result.setdefault("insights", [])
                result.setdefault("gaps", [])
                result.setdefault("cross_scene_links", [])
                result.setdefault("kg_updates", [])
                result.setdefault("mental_model_updates", [])
                result.setdefault("disposition", {})
                return result
            except Exception as e:
                logger.warning("Reflect LLM 调用失败 (尝试 %d/%d): %s",
                               attempt + 1, self._max_retries, e)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        return {"insights": [], "gaps": [], "cross_scene_links": [], "kg_updates": [], "mental_model_updates": [], "disposition": {}}

    async def _apply_kg_updates(self, updates: list[dict]) -> int:
        """批量更新知识图谱。"""
        count = 0
        for update in updates:
            try:
                action = update.get("action", "")
                if action == "add_entity":
                    await self._kg.add_entity(
                        update["subject"],
                        update.get("type", "unknown"),
                        update.get("metadata", {}),
                    )
                    count += 1
                elif action == "add_relation":
                    await self._kg.add_relation(
                        update["subject"],
                        update["predicate"],
                        update["object"],
                        update.get("weight", 1.0),
                        source="reflect",
                    )
                    count += 1
            except Exception as e:
                logger.warning("KG 更新失败: %s", e)
        return count

    async def _apply_mental_model_updates(self, updates: list[dict]) -> int:
        """批量更新心智模型。"""
        if not self._mental_models or not updates:
            return 0
        count = 0
        for update in updates:
            try:
                await self._mental_models.upsert_belief(
                    category=update.get("category", "belief"),
                    key=update.get("key", ""),
                    value=update.get("value", ""),
                    confidence_delta=update.get("confidence_delta", 0.1),
                )
                count += 1
            except Exception as e:
                logger.warning("心智模型更新失败: %s", e)
        return count

    async def _log_reflect(self, result: dict) -> None:
        """记录 reflect 结果到日志表。"""
        conn = await self._pool.acquire()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS reflect_logs (
                    id TEXT PRIMARY KEY,
                    result TEXT,
                    created_at REAL
                )
            """)
            await conn.execute(
                "INSERT INTO reflect_logs (id, result, created_at) VALUES (?, ?, ?)",
                (str(uuid4()), json.dumps(result, ensure_ascii=False), time.time()),
            )
            await conn.commit()
        except Exception as e:
            logger.warning("写入 reflect 日志失败: %s", e)
        finally:
            await self._pool.release(conn)