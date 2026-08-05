"""pudica-Memory HTTP 启动入口 — 启动 REST API 服务。"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

os.environ["PYTHONUTF8"] = "1"
os.environ["UNIFIED_MEMORY_STORAGE_DIR"] = os.path.join(PROJECT_ROOT, "data")

# 插入 --http 参数，使 main() 进入 HTTP 模式
sys.argv = [sys.argv[0], "--http", "--host", "127.0.0.1", "--port", "8000"]
if "--log-level" in sys.argv:
    pass  # 保留用户指定的 log level
else:
    sys.argv.extend(["--log-level", "INFO"])

from unified_memory.main import main

main()