"""api/mcp_server.py — FastMCP 服务器实现。

提供 create_mcp_server(app: UnifiedMemoryApp) 工厂函数，
以及 MCPServer 类封装，兼容 mempalace 工具名。
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_mcp_server(app: Any) -> Any:
    """从 UnifiedMemoryApp 创建 FastMCP 应用。

    这是 main.py 的入口点，直接生成 MCP 服务器实例。

    Args:
        app: UnifiedMemoryApp 实例

    Returns:
        FastMCP 应用实例，已注册所有工具
    """
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
        mcp.tool(name=name, description=description)(handler)

    logger.info("MCP 应用已创建，注册 %d 个工具", len(registry.list_tools()))
    return mcp


class MCPServer:
    """FastMCP 服务器封装（可选，直接使用 create_mcp_server 更简洁）。

    通过 ToolRegistry 暴露工具，兼容 mempalace 工具名。
    """

    def __init__(self, registry: Any):
        """
        Args:
            registry: ToolRegistry 实例
        """
        self._registry = registry
        self._mcp_app = None

    def create_app(self) -> Any:
        """创建 FastMCP 应用。

        Returns:
            FastMCP 应用实例
        """
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
            app.tool(name=name, description=description)(handler)

        self._mcp_app = app
        logger.info("MCP 应用已创建，注册 %d 个工具", len(self._registry.list_tools()))
        return app

    async def run_stdio(self) -> None:
        """以 stdio 模式运行 MCP 服务器。"""
        app = self.create_app()
        logger.info("MCP 服务器 (stdio 模式) 启动")
        await app.run_stdio_async()

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """以 SSE 模式运行 MCP 服务器。

        Args:
            host: 监听地址（默认 127.0.0.1，仅本地访问）
            port: 监听端口
        """
        app = self.create_app()
        logger.info("MCP 服务器 (SSE 模式) 启动: %s:%d", host, port)
        await app.run_sse_async(host=host, port=port)