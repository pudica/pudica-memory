"""pudica-memory MemoryProvider — 为 Hermes Agent 提供自动记忆存取。

安装到 Hermes 的 plugins/memory/ 目录下，通过 memory.provider: pudica_memory 激活。
在每轮对话前后自动存取记忆，同时暴露 MCP 工具供 LLM 手动调用。

工作原理：
- prefetch(query) → 每轮对话前检索相关记忆，注入 system prompt
- sync_turn(user, asst) → 每轮对话后自动存储到管线
- queue_prefetch(query) → 后台预取下一轮记忆
- get_tool_schemas() → 暴露 pudica-memory 的 MCP 工具
- handle_tool_call() → 转发工具调用到 pudica-memory 引擎
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hermes MemoryProvider 接口
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider, RecallStatus
except ImportError:
    # 在 Hermes 环境外测试时使用桩
    class MemoryProvider:
        name = "pudica_memory"
        def is_available(self) -> bool: return True
        def unavailable_reason(self) -> str: return ""
        def initialize(self, **kwargs): pass
        def shutdown(self): pass
        def prefetch(self, query: str, *, session_id: str = "") -> str: return ""
        def queue_prefetch(self, query: str, *, session_id: str = ""): pass
        def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = ""): pass
        def on_turn_start(self, turn, message): pass
        def get_tool_schemas(self): return []
        def handle_tool_call(self, name, args): return {}
        def recall_status(self) -> Optional[Any]: return None

    class RecallStatus:
        def __init__(self, provider_label="", count=0, glyph=""):
            self.provider_label = provider_label
            self.count = count
            self.glyph = glyph


# ---------------------------------------------------------------------------
# pudica-memory 引擎包装
# ---------------------------------------------------------------------------

class PudicaMemoryEngine:
    """pudica-memory 引擎的线程安全包装。

    在 Hermes 进程内加载 UnifiedMemoryApp，通过同步包装调用异步方法。
    """

    def __init__(self):
        self._app: Any = None
        self._config: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._run_in_loop = None  # 由 _init_loop 设置

    def _init_loop(self):
        """初始化事件循环（在后台线程中运行）。"""
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._run_in_loop = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result()

        # 启动循环线程
        def _run_loop():
            loop.run_forever()
        t = threading.Thread(target=_run_loop, daemon=True, name="pudica-memory-loop")
        t.start()

    def _ensure_initialized(self):
        """确保引擎已初始化。"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._init_loop()

            # 加载配置
            from unified_memory.config import Config
            config = Config.load()

            # 覆盖 LLM 配置（优先使用环境变量）
            env_api_base = os.environ.get("UNIFIED_MEMORY_LLM_API_BASE")
            env_api_key = os.environ.get("UNIFIED_MEMORY_LLM_API_KEY")
            env_model = os.environ.get("UNIFIED_MEMORY_LLM_MODEL")
            if env_api_base:
                config.llm.api_base = env_api_base
            if env_api_key:
                config.llm.api_key = env_api_key
            if env_model:
                config.llm.model = env_model

            # 启用中间件标志（让 app 初始化时创建 middleware，但 auto_memory 的 HTTP 中间件不用）
            config.middleware.enabled = True

            # 初始化应用
            from unified_memory.main import UnifiedMemoryApp
            app = UnifiedMemoryApp(config)
            self._run_in_loop(app.initialize())
            self._app = app
            self._config = config
            self._initialized = True
            logger.info("pudica-memory 引擎初始化完成: chroma=%s, sqlite=%s",
                        config.chroma.persist_dir, config.sqlite.db_path)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """搜索记忆。"""
        self._ensure_initialized()
        try:
            results = self._run_in_loop(
                self._app.pipeline.search(query, top_k=top_k)
            )
            if results and hasattr(results, "get"):  # 可能是 dict 格式
                items = results.get("results", results.get("items", []))
                if isinstance(items, list):
                    return items
            if isinstance(results, list):
                return results
            return []
        except Exception as e:
            logger.error("pudica-memory search 失败: %s", e)
            return []

    def ingest(self, content: str, source: str = "hermes_turn", metadata: dict | None = None) -> bool:
        """存储内容到记忆管线。"""
        self._ensure_initialized()
        try:
            self._run_in_loop(
                self._app.pipeline.ingest(content, source=source, metadata=metadata or {})
            )
            return True
        except Exception as e:
            logger.error("pudica-memory ingest 失败: %s", e)
            return False

    def get_tool_list(self) -> list[dict]:
        """获取工具列表。"""
        self._ensure_initialized()
        if self._app and self._app.registry:
            return self._app.registry.list_tools()
        return []

    def handle_tool(self, name: str, args: dict) -> Any:
        """处理工具调用。"""
        self._ensure_initialized()
        if self._app and self._app.registry:
            handler_info = self._app.registry.get_tool(name)
            if handler_info:
                handler = handler_info["handler"]
                try:
                    result = self._run_in_loop(handler(**args))
                    return result
                except Exception as e:
                    logger.error("pudica-memory tool '%s' 调用失败: %s", name, e)
                    return {"error": str(e)}
        return {"error": f"tool {name} not found"}

    def get_stats(self) -> dict:
        """获取系统统计。"""
        self._ensure_initialized()
        try:
            if self._app and self._app.pipeline:
                buffer = self._app.pipeline.get_buffer_size()
                return {
                    "ingest_count": 0,
                    "l1_count": 0,
                    "l2_count": 0,
                    "buffer_size": buffer,
                    "initialized": True,
                }
        except Exception:
            pass
        return {"initialized": False}

    def shutdown(self):
        """关闭引擎。"""
        if self._app and self._initialized:
            try:
                self._run_in_loop(self._app.shutdown())
            except Exception as e:
                logger.warning("pudica-memory shutdown 异常: %s", e)
            self._initialized = False


# ---------------------------------------------------------------------------
# 全局引擎实例
# ---------------------------------------------------------------------------

_engine: Optional[PudicaMemoryEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> PudicaMemoryEngine:
    """获取全局 pudica-memory 引擎实例。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PudicaMemoryEngine()
    return _engine


# ---------------------------------------------------------------------------
# MemoryProvider 实现
# ---------------------------------------------------------------------------

_PUDICA_GLYPH = "🧠"


class PudicaMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider — pudica-memory 自动存取适配器。

    每个对话轮次自动：
    1. prefetch — 从记忆库检索相关上下文，注入到 system prompt
    2. sync_turn — 将当前轮次对话存储到管线
    3. queue_prefetch — 后台预取下一轮记忆
    """

    name = "pudica_memory"

    def __init__(self):
        self._engine: Optional[PudicaMemoryEngine] = None
        self._initialized = False

        # 配置
        self._auto_retain = True
        self._auto_recall = True
        self._retain_every_n_turns = 1
        self._recall_sync = False
        self._recall_max_tokens = 4096
        self._recall_max_items = 10
        self._recall_max_chars = 2000
        self._recall_indicator = True
        self._retain_indicator = True
        self._recall_min_query_chars = 5

        # 状态
        self._turn_counter = 0
        self._session_turns: list[str] = []
        self._session_id = ""
        self._platform = "cli"
        self._user_id = ""
        self._user_name = ""
        self._chat_id = ""
        self._chat_name = ""
        self._agent_identity = ""

        # 预取缓存
        self._prefetch_result = ""
        self._prefetch_count = 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._last_recall_returned = False
        self._last_recall_count = 0

    # -- 生命周期 -----------------------------------------------------------

    def is_available(self) -> bool:
        """检查 pudica-memory 是否可用。"""
        try:
            import unified_memory  # noqa
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        """返回不可用的原因。"""
        try:
            import unified_memory  # noqa
            return ""
        except ImportError as e:
            return f"unified_memory 包未安装: {e}"

    def initialize(self, **kwargs) -> None:
        """初始化 provider。"""
        self._session_id = kwargs.get("session_id", "")
        self._platform = kwargs.get("platform", "cli")
        self._user_id = kwargs.get("user_id", "")
        self._user_name = kwargs.get("user_name", "")
        self._chat_id = kwargs.get("chat_id", "")
        self._chat_name = kwargs.get("chat_name", "")
        self._agent_identity = kwargs.get("agent_identity", "")

        # 从环境变量读取配置
        self._auto_retain = os.environ.get("PUDICA_MEMORY_AUTO_RETAIN", "true").lower() == "true"
        self._auto_recall = os.environ.get("PUDICA_MEMORY_AUTO_RECALL", "true").lower() == "true"
        self._retain_every_n_turns = int(os.environ.get("PUDICA_MEMORY_RETAIN_EVERY_N", "1"))

        # 初始化引擎
        self._engine = _get_engine()
        self._initialized = True
        logger.info("pudica-memory provider 初始化: session=%s, platform=%s, user=%s",
                     self._session_id, self._platform, self._user_id)

    def shutdown(self) -> None:
        """关闭 provider。"""
        self._initialized = False
        logger.info("pudica-memory provider 已关闭")

    # -- 自动检索（prefetch — 每轮对话前调用）--------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """从记忆库检索相关上下文。

        Hermes 在每轮对话前调用此方法，返回值注入到 system prompt 中。
        """
        if not self._auto_recall:
            return ""
        if not self._engine:
            return ""
        if len(query.strip()) < self._recall_min_query_chars:
            return ""

        if session_id:
            self._session_id = session_id

        # 检查后台预取结果
        with self._prefetch_lock:
            result = self._prefetch_result
            count = self._prefetch_count
            self._prefetch_result = ""
            self._prefetch_count = 0

        if result:
            self._last_recall_returned = True
            self._last_recall_count = count
            return self._format_recall(result)

        # 同步模式：实时检索
        try:
            items = self._engine.search(query, top_k=self._recall_max_items)
            if items:
                formatted = self._format_items(items)
                self._last_recall_returned = True
                self._last_recall_count = len(items)
                return formatted
        except Exception as e:
            logger.debug("pudica-memory prefetch 失败: %s", e)

        self._last_recall_returned = False
        self._last_recall_count = 0
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """后台预取下一轮记忆。"""
        if self._recall_sync:
            return
        if not self._auto_recall:
            return
        if not self._engine:
            return
        if len(query.strip()) < self._recall_min_query_chars:
            return

        def _run():
            try:
                items = self._engine.search(query, top_k=self._recall_max_items)
                if items:
                    with self._prefetch_lock:
                        self._prefetch_result = self._format_items(items)
                        self._prefetch_count = len(items)
            except Exception as e:
                logger.debug("pudica-memory queue_prefetch 失败: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="pudica-memory-prefetch"
        )
        self._prefetch_thread.start()

    # -- 自动存储（sync_turn — 每轮对话后调用）-------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """将当前轮次对话存储到记忆管线。"""
        if not self._auto_retain:
            return
        if not self._engine:
            return

        if session_id:
            self._session_id = session_id

        # 构建对话记忆
        now = datetime.now(timezone.utc).isoformat()
        turn = json.dumps([
            {"role": "user", "content": user_content, "timestamp": now},
            {"role": "assistant", "content": assistant_content, "timestamp": now},
        ], ensure_ascii=False)
        self._session_turns.append(turn)
        self._turn_counter += 1

        # 按配置的频率存储
        if self._turn_counter % self._retain_every_n_turns != 0:
            return

        # 组装本次要存储的内容
        content = "[" + ",".join(self._session_turns) + "]"
        metadata = {
            "source": "hermes_turn",
            "session_id": self._session_id,
            "platform": self._platform,
            "user_id": self._user_id,
            "user_name": self._user_name,
            "agent_identity": self._agent_identity,
            "turn_count": str(self._turn_counter),
            "retained_at": now,
        }

        # 异步存储（不阻塞对话）
        def _do_ingest():
            self._engine.ingest(content, source="hermes_turn", metadata=metadata)

        t = threading.Thread(target=_do_ingest, daemon=True, name="pudica-memory-retain")
        t.start()

    # -- 轮次开始钩子 --------------------------------------------------------

    def on_turn_start(self, turn, message) -> None:
        """每轮对话开始时触发的钩子。"""
        pass

    # -- 工具接口（暴露 MCP 工具给 LLM）--------------------------------------

    def get_tool_schemas(self) -> list[dict]:
        """返回 pudica-memory 工具的工具签名。"""
        if not self._engine:
            return []
        try:
            self._engine._ensure_initialized()
            return self._engine.get_tool_list()
        except Exception:
            return []

    def handle_tool_call(self, name: str, args: dict) -> Any:
        """处理工具调用。"""
        if not self._engine:
            return {"error": "pudica-memory not initialized"}
        return self._engine.handle_tool(name, args)

    def recall_status(self) -> Optional[RecallStatus]:
        """返回当前轮次注入的记忆数（用于界面指示器）。"""
        if not self._recall_indicator or not self._last_recall_returned:
            return None
        return RecallStatus(
            provider_label="pudica-memory",
            count=self._last_recall_count,
            glyph=_PUDICA_GLYPH,
        )

    # -- 辅助方法 -----------------------------------------------------------

    def _format_recall(self, text: str) -> str:
        """格式化召回结果。"""
        return text

    def _format_items(self, items: list) -> str:
        """将记忆条目格式化为提示文本。"""
        parts = []
        char_budget = self._recall_max_chars
        for item in items:
            if item is None:
                continue
            # 支持 dataclass (FusionResult) 和 dict 两种格式
            if isinstance(item, dict):
                content = item.get("content", item.get("text", ""))
                score = item.get("score", item.get("relevance", ""))
            else:
                content = getattr(item, "text", getattr(item, "content", ""))
                score = getattr(item, "score", getattr(item, "relevance", ""))
            if not content:
                continue
            line = content
            if score:
                line = f"[{score:.2f}] {content}"
            if len(line) > char_budget:
                line = line[:char_budget] + "..."
            parts.append(line)
            char_budget -= len(line)
            if char_budget <= 0:
                break
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 插件注册（Hermes plugin register 模式）
# ---------------------------------------------------------------------------

def register(ctx):
    """Hermes 插件注册接口。"""
    ctx.register_memory_provider(PudicaMemoryProvider())