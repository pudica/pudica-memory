"""pudica-Memory MCP 启动入口 — 供 Codex marketplace 调用。"""
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

# 插入启动参数：用 HTTP SSE 模式启动（Hermes MCP client 走 SSE 连接）
# --http 启动 REST API + SSE 端点，供 Hermes Gateway 的 mcp_servers 连接
sys.argv = ["run_mcp.py", "--http", "--port", os.environ.get("UNIFIED_MEMORY_HTTP_PORT", "8420"), "--host", "127.0.0.1"]

from unified_memory.main import main

main()
