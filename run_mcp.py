"""pudica-Memory MCP 启动入口 — 供 Codex marketplace 调用。"""
import sys
import os

# 在 fastmcp 导入前打补丁，跳过版本检查
import fastmcp.utilities.version_check as _vc
_vc.check_for_newer_version = lambda: None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

os.environ["PYTHONUTF8"] = "1"
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(PROJECT_ROOT, "data")

# 插入 --mcp 参数
sys.argv = [sys.argv[0], "--mcp", "--auto"]

from unified_memory.main import main

main()
