"""pudica-Memory MCP 启动入口 — 供 Hermes Gateway / Codex marketplace 调用（stdio 模式）。"""
import sys
import os

# 在 fastmcp 导入前打补丁，跳过版本检查
import fastmcp.utilities.version_check as _vc
_vc.check_for_newer_version = lambda: None

import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

os.environ["PYTHONUTF8"] = "1"
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(PROJECT_ROOT, "data")

# 启动 MCP stdio 模式（供 Hermes Gateway 的 mcp_servers 通过 command 拉起）
# Gateway 通过 stdin/stdout 与 MCP 通信，不走 HTTP/SSE
from unified_memory.main import run_mcp
from unified_memory.config import Config

config = Config.load()
import asyncio
asyncio.run(run_mcp(config))