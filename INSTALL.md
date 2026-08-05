# pudica-Memory 安装指南

## 一、解压放置

解压到任意目录，例如：`~/pudica-memory/`。

## 二、安装依赖

```bash
cd pudica-memory
pip install -r requirements.txt
# 或安装为 Python 包：
pip install -e .
```

需要 Python >= 3.11。中文语义搜索使用**随包提供的本地 ONNX 嵌入模型**
（`models/bge-small-zh-v1.5/`，24MB int8 量化版），**无需联网下载、无需安装 torch**。

## 三、配置 LLM（可选）

编辑 `config.json` 里的 `llm` 段，指向你实际可用的 LLM 端点（用于 reflect 洞察等增强功能；
LLM 不可用时自动降级为本地规则模式，核心功能不受影响）：

```json
{
  "llm": {
    "api_base": "http://你的LLM服务器:端口/v1",
    "api_key": "你的key",
    "model": "你的模型名"
  }
}
```

推荐用环境变量覆盖，无需修改 config.json：

```bash
export UNIFIED_MEMORY_LLM_API_BASE="http://你的LLM:端口/v1"
export UNIFIED_MEMORY_LLM_API_KEY="你的key"
export UNIFIED_MEMORY_LLM_MODEL="你的模型名"
```

## 四、启动

**HTTP 模式（推荐）：**
```bash
python run_http.py
# 访问 http://127.0.0.1:8000/api/v1/health
```

**MCP 模式（供 AI Agent）：**
```bash
python run_mcp.py
```

**Windows 快捷方式：**
双击 `start_http.bat`

## 五、CLI 模式

```bash
# 查看帮助
python -m unified_memory.main --help

# 启动 HTTP 服务器
python -m unified_memory.main --http --host 127.0.0.1 --port 8000

# 启动 MCP 服务器
python -m unified_memory.main --mcp

# 运行集成测试
python -m unified_memory.main --test
```

## 六、验证

```bash
python seed.py
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"测试","top_k":5}'
```

## 常见问题

- **搜索返回空**：确认嵌入模型 `bge-small-zh-v1.5` 已正确加载。
- **L1 提取失败**：检查 LLM 配置是否正确。
- **数据目录**：默认 `data/`，可用 `UNIFIED_MEMORY_STORAGE_DIR` 环境变量覆盖。
- **编码问题**：设置 `PYTHONUTF8=1` 环境变量确保中文正确处理。