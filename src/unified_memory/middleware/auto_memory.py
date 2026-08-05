"""middleware/auto_memory.py — 自动存取中间件。

在 LLM/Agent 和 pudica-memory 之间加一层透明中间件，实现：
  1. 自动存储：拦截每条用户消息和 LLM 回复，自动喂给 pipeline.ingest()
  2. 自动检索：从用户消息提取查询意图，自动调用 4 路并行检索
  3. 上下文注入：把检索到的记忆 + 心智模型信念注入 system prompt

LLM 完全不需要关心记忆的存取 —— 中间件全权代理。
仍保留手动 MCP 工具，高级场景可精确控制。

使用示例：
    app = UnifiedMemoryApp(config)
    await app.initialize()

    middleware = AutoMemoryMiddleware(app)
    enriched = await middleware.pre_process("帮我推荐一款适合气阴两虚的方剂")
    # enriched 包含了相关记忆和用户信念，直接喂给 LLM
    llm_response = await llm.chat(enriched)
    await middleware.post_process(llm_response)

设计原则：
  - 零侵入：不修改 pipeline / search / store 的任何接口
  - 可配置：auto_store / auto_search / auto_inject 可独立开关
  - 可降级：任何中间件步骤失败都不阻塞主流程
  - 可过滤：短消息、系统消息、重复消息自动跳过
"""

import asyncio
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AutoMemoryMiddleware:
    """自动存取中间件：包装 UnifiedMemoryApp，提供透明记忆存取。

    核心方法：
        pre_process(user_message) → enriched_prompt
            自动存储消息 → 自动检索记忆 → 注入上下文
        post_process(llm_response, user_message)
            自动存储 LLM 回复

    过滤规则：
        - 太短的消息（< min_message_length）不存储，不检索
        - 纯系统指令（以 / 开头）不存储
        - 重复消息依赖 L0 Dedup 自动去重
        - LLM 回复中纯工具调用部分不存储
    """

    # 无意义短消息模式（不存储、不检索）
    _SKIP_PATTERNS = [
        re.compile(r'^[\s]*$'),                     # 纯空白
        re.compile(r'^[好的嗯哦啊哈呵]+[\s!！。.]*$'),  # 纯语气词
        re.compile(r'^[/\\].*'),                     # 系统指令
        re.compile(r'^[\d\s]+$'),                    # 纯数字
    ]

    # LLM 回复中的工具调用标记（不存储这些行）
    _TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>|<function_call>|```tool|```function', re.IGNORECASE
    )

    def __init__(
        self,
        app: Any,
        *,
        auto_store: bool = True,
        auto_search: bool = True,
        auto_inject: bool = True,
        min_message_length: int = 5,
        search_top_k: int = 10,
        max_context_items: int = 5,
        max_context_chars: int = 2000,
        store_llm_response: bool = True,
    ):
        """初始化自动存取中间件。

        Args:
            app: UnifiedMemoryApp 实例（已 initialize）
            auto_store: 是否自动存储用户消息
            auto_search: 是否自动检索相关记忆
            auto_inject: 是否将记忆注入 system prompt
            min_message_length: 消息最小长度（字符），短于此值跳过
            search_top_k: 自动检索返回条数
            max_context_items: 注入上下文的最大记忆条数
            max_context_chars: 注入上下文的最大字符数
            store_llm_response: 是否自动存储 LLM 回复
        """
        self._app = app
        self._pipeline = app.pipeline
        self._temp_engine = app.registry._temp_engine if hasattr(app.registry, '_temp_engine') else None
        self._mental_models = app.mental_models
        self._reranker = app.reranker

        self._auto_store = auto_store
        self._auto_search = auto_search
        self._auto_inject = auto_inject
        self._min_message_length = min_message_length
        self._search_top_k = search_top_k
        self._max_context_items = max_context_items
        self._max_context_chars = max_context_chars
        self._store_llm_response = store_llm_response

        # 统计
        self._store_count = 0
        self._search_count = 0
        self._inject_count = 0
        self._skip_count = 0

        logger.info(
            "AutoMemoryMiddleware 已初始化 (store=%s, search=%s, inject=%s)",
            auto_store, auto_search, auto_inject,
        )

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    async def pre_process(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        source: str = "auto",
    ) -> str:
        """消息预处理：自动存储 + 自动检索 + 上下文注入。

        调用时机：在将用户消息发给 LLM 之前。
        流程：
            1. 过滤检查（太短/无意义 → 跳过）
            2. 自动存储消息到记忆管线
            3. 自动检索相关记忆（4路并行 + RRF融合）
            4. 将记忆 + 心智模型注入 system prompt
            5. 返回增强后的完整 prompt

        Args:
            user_message: 用户消息文本
            system_prompt: 原始 system prompt（可选）
            source: 消息来源标记

        Returns:
            增强后的 prompt（包含记忆上下文），可直接传给 LLM。
            如果 auto_inject=False，返回原始 system_prompt。
        """
        # 过滤检查
        if self._should_skip(user_message):
            self._skip_count += 1
            logger.debug("跳过消息（太短/无意义）: %s", user_message[:30])
            return system_prompt or ""

        # 并行执行：存储 + 检索
        tasks = []
        store_task = None
        search_task = None

        if self._auto_store:
            store_task = asyncio.create_task(
                self._safe_store(user_message, source)
            )

        if self._auto_search:
            search_task = asyncio.create_task(
                self._safe_search(user_message)
            )

        # 等待检索完成（存储是异步缓冲，不阻塞）
        search_results = []
        if search_task:
            try:
                search_results = await search_task
            except Exception as e:
                logger.warning("自动检索失败: %s", e)

        # 等待存储完成（确保消息已入缓冲区）
        if store_task:
            try:
                await store_task
            except Exception as e:
                logger.warning("自动存储失败: %s", e)

        # 上下文注入
        if not self._auto_inject:
            return system_prompt or ""

        context = await self._build_context(search_results, user_message)
        if not context:
            return system_prompt or ""

        self._inject_count += 1

        # 将记忆上下文注入 system prompt
        if system_prompt:
            return f"{system_prompt}\n\n{context}"
        else:
            return context

    async def post_process(
        self,
        llm_response: str,
        user_message: Optional[str] = None,
        source: str = "llm",
    ) -> None:
        """回复后处理：自动存储 LLM 回复。

        调用时机：在 LLM 回复生成之后。
        将 LLM 回复存入记忆，使后续对话能回忆 LLM 说过的内容。

        Args:
            llm_response: LLM 的回复文本
            user_message: 对应的用户消息（作为上下文元数据）
            source: 来源标记
        """
        if not self._store_llm_response:
            return

        if not llm_response or self._should_skip(llm_response):
            return

        # 去除工具调用部分，只存储有意义的文本
        cleaned = self._clean_llm_response(llm_response)
        if not cleaned or len(cleaned) < self._min_message_length:
            return

        metadata = {}
        if user_message:
            metadata["reply_to"] = user_message[:200]

        await self._safe_store(cleaned, source, metadata)

    # ------------------------------------------------------------------
    # 快捷方法：一次性处理完整对话轮次
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        user_message: str,
        llm_call: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        """处理完整对话轮次：pre_process → LLM 调用 → post_process。

        Args:
            user_message: 用户消息
            llm_call: 异步可调用对象，接收 prompt 返回 LLM 回复
            system_prompt: 原始 system prompt

        Returns:
            LLM 回复文本
        """
        enriched = await self.pre_process(user_message, system_prompt)
        response = await llm_call(enriched)
        await self.post_process(response, user_message)
        return response

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _should_skip(self, text: str) -> bool:
        """检查消息是否应该被跳过（太短或无意义）。"""
        if not text or len(text.strip()) < self._min_message_length:
            return True
        for pattern in self._SKIP_PATTERNS:
            if pattern.match(text.strip()):
                return True
        return False

    async def _safe_store(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """安全存储消息（异常不抛出）。"""
        try:
            msg_id = await self._pipeline.ingest(
                content, source=source, metadata=metadata
            )
            if msg_id:
                self._store_count += 1
                logger.debug("自动存储: %s... (id=%s)", content[:30], msg_id[:8])
        except Exception as e:
            logger.warning("自动存储异常: %s", e)

    async def _safe_search(self, query: str) -> list:
        """安全检索记忆（异常返回空列表）。"""
        try:
            self._search_count += 1
            # 直接调用 pipeline.search（内含 RRF 融合 + 重排）
            results = await self._pipeline.search(query, top_k=self._search_top_k)
            logger.debug("自动检索: query='%s...' → %d 条结果", query[:20], len(results))
            return results
        except Exception as e:
            logger.warning("自动检索异常: %s", e)
            return []

    async def _build_context(
        self,
        search_results: list,
        user_message: str,
    ) -> str:
        """构建注入 LLM 的记忆上下文。

        格式：
            ## 相关记忆
            1. [来源: semantic, bm25] 记忆内容...
            2. ...

            ## 用户心智模型
            - [preference] 气阴两虚: 倾向于经方...

        Args:
            search_results: 检索结果列表
            user_message: 用户消息（用于信念查询）

        Returns:
            格式化的上下文文本
        """
        parts: list[str] = []

        # 1. 记忆检索结果
        if search_results:
            memory_lines: list[str] = []
            total_chars = 0
            for i, result in enumerate(search_results[:self._max_context_items], 1):
                # 提取记忆文本
                text = self._extract_text(result)
                if not text:
                    continue

                # 截断过长记忆
                max_item_chars = self._max_context_chars // self._max_context_items
                if len(text) > max_item_chars:
                    text = text[:max_item_chars] + "..."

                sources = self._extract_sources(result)
                source_tag = f"[来源: {', '.join(sources)}]" if sources else ""
                memory_lines.append(f"{i}. {source_tag} {text}")

                total_chars += len(text)
                if total_chars >= self._max_context_chars:
                    break

            if memory_lines:
                parts.append("## 相关记忆\n" + "\n".join(memory_lines))

        # 2. 心智模型信念
        if self._mental_models:
            try:
                belief_text = await self._mental_models.format_for_context(
                    max_items=8
                )
                if belief_text:
                    parts.append(belief_text)
            except Exception as e:
                logger.debug("心智模型上下文构建失败: %s", e)

        if not parts:
            return ""

        header = "--- 以下为 pudica-Memory 自动注入的记忆上下文 ---"
        footer = "--- 记忆上下文结束 ---"
        return f"{header}\n\n" + "\n\n".join(parts) + f"\n\n{footer}"

    def _extract_text(self, result: Any) -> str:
        """从检索结果中提取文本内容。"""
        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, dict):
            return result.get("text", result.get("content", ""))
        return str(result)

    def _extract_sources(self, result: Any) -> list[str]:
        """从检索结果中提取来源策略。"""
        if hasattr(result, "sources"):
            return result.sources
        if isinstance(result, dict):
            return result.get("sources", [])
        return []

    def _clean_llm_response(self, response: str) -> str:
        """清理 LLM 回复：去除工具调用标记，保留有意义的文本。

        Args:
            response: LLM 原始回复

        Returns:
            清理后的文本
        """
        lines = response.split("\n")
        cleaned_lines: list[str] = []
        in_tool_block = False

        for line in lines:
            # 检测工具调用块开始
            if self._TOOL_CALL_PATTERN.search(line):
                in_tool_block = True
                continue
            # 检测代码块结束
            if in_tool_block and line.strip() == "```":
                in_tool_block = False
                continue
            if not in_tool_block:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    # ------------------------------------------------------------------
    # 统计与状态
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """获取中间件统计信息。

        Returns:
            {
                "auto_store": bool,
                "auto_search": bool,
                "auto_inject": bool,
                "store_count": int,
                "search_count": int,
                "inject_count": int,
                "skip_count": int,
            }
        """
        return {
            "auto_store": self._auto_store,
            "auto_search": self._auto_search,
            "auto_inject": self._auto_inject,
            "store_count": self._store_count,
            "search_count": self._search_count,
            "inject_count": self._inject_count,
            "skip_count": self._skip_count,
        }

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        self._store_count = 0
        self._search_count = 0
        self._inject_count = 0
        self._skip_count = 0
