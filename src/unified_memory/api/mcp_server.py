"""api/mcp_server.py — FastMCP 服务器实现。

提供 create_mcp_server(app: UnifiedMemoryApp) 工厂函数，
以及 MCPServer 类封装，兼容 mempalace 工具名。
"""

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _make_tool_handler(handler):
    """生成一个 FastMCP 兼容的、具有固定参数签名的工具函数。

    FastMCP 基于 inspect.signature 生成 JSON schema，**kwargs 不被支持。
    我们提取 handler 的显式参数名并生成一个**只有这些参数**的闭包。

    返回一个带有正确 __signature__ 的 async 函数。
    """
    sig = inspect.signature(handler)
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    explicit_params = [
        p.name
        for p in params.values()
        if p.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        and p.name != "self"
    ]

    async def tool_fn(**kwargs):
        if has_var_kw:
            return await handler(**kwargs)
        filtered = {k: kwargs[k] for k in explicit_params if k in kwargs}
        return await handler(**filtered)

    tool_fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                name=pname,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=params[pname].default,
                annotation=params[pname].annotation,
            )
            for pname in explicit_params
        ]
    )
    tool_fn.__name__ = handler.__name__
    # FastMCP 的 from_function 用 typing.get_type_hints() 查 __annotations__，
    # 不从 __signature__ 拿。这里直接复制类型标注过去。
    h = handler
    if hasattr(h, "__annotations__"):
        tool_fn.__annotations__ = {
            k: v for k, v in h.__annotations__.items() if k not in ("return",)
        }
    else:
        # 对于 bound method，__annotations__ 可能在 __func__ 上
        func = getattr(h, "__func__", h)
        if hasattr(func, "__annotations__"):
            tool_fn.__annotations__ = {
                k: v for k, v in func.__annotations__.items() if k not in ("return",)
            }
    return tool_fn


def create_mcp_server(app: Any) -> Any:
    """从 UnifiedMemoryApp 创建 FastMCP 应用。"""
    try:
        from fastmcp import FastMCP
    except ImportError:
        logger.error("fastmcp 未安装，请运行: pip install fastmcp")
        raise

    mcp = FastMCP("unified-memory")
    registry = app.registry

    for tool_def in registry.list_tools():
        name = tool_def["name"]
        description = tool_def["description"]
        handler_info = registry.get_tool(name)
        if handler_info is None:
            continue
        handler = handler_info["handler"]
        tool_fn = _make_tool_handler(handler)
        mcp.tool(name=name, description=description)(tool_fn)

    logger.info("MCP 应用已创建，注册 %d 个工具", len(registry.list_tools()))
    return mcp


class MCPServer:
    """FastMCP 服务器封装。"""

    def __init__(self, registry: Any):
        self._registry = registry
        self._mcp_app = None

    def create_app(self) -> Any:
        """创建 FastMCP 应用。"""
        try:
            from fastmcp import FastMCP
        except ImportError:
            logger.error("fastmcp 未安装，请运行: pip install fastmcp")
            raise

        app = FastMCP("unified-memory")
        for tool_def in self._registry.list_tools():
            name = tool_def["name"]
            description = tool_def["description"]
            handler_info = self._registry.get_tool(name)
            if handler_info is None:
                continue
            handler = handler_info["handler"]
            tool_fn = _make_tool_handler(handler)
            app.tool(name=name, description=description)(tool_fn)

        self._mcp_app = app
        logger.info("MCP 应用已创建，注册 %d 个工具", len(self._registry.list_tools()))
        return app

    async def run_stdio(self) -> None:
        """以 stdio 模式运行 MCP 服务器。"""
        app = self.create_app()
        logger.info("MCP 服务器 (stdio 模式) 启动")
        await app.run_stdio_async()

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """以 SSE 模式运行 MCP 服务器。"""
        app = self.create_app()
        logger.info("MCP 服务器 (SSE 模式) 启动: %s:%d", host, port)
        await app.run_sse_async(host=host, port=port)