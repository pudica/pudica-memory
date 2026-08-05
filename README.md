# pudica-Memory — 统一记忆系统 v3.2.0

基于 SQLite + ChromaDB 双存储的轻量级 AI 记忆系统，融合 TencentDB Agent Memory、MemPalace、Hindsight 三大开源记忆体的设计精华，提供 4 层管线编排、4 路并行检索（TEMPR）、知识图谱、心智模型、自动存取中间件。

## 特性

### 核心能力
- **轻量部署**：纯 Python，零外部服务依赖，`pip install` 即可运行
- **双存储架构**：SQLite（结构数据 + FTS5 全文搜索）+ ChromaDB（向量存储 + 语义搜索）
- **4 层管线**：L0 去重 → L1 LLM 提取 → L2 场景组织 → L3 检索压缩
- **4 路并行检索**：语义检索 + BM25 全文 + 图链接扩展 + 时间检索 → RRF 融合
- **知识图谱**：LRU 缓存 + SQLite 按需加载，支持 10 万+ 实体而不 OOM
- **后台任务**：Reflect（自动摘要）+ Consolidation（去重合并）+ Compression（记忆压缩）

### v3.0 新增（融合三大开源记忆体）
- **Verbatim 逐字存储**（MemPalace）：管线处理前保存原始输入，支持精确回溯
- **Cross-Encoder 重排器**（Hindsight）：RRF 融合后二次精排，提升 top-k 精度
- **心智模型/信念系统**（Hindsight）：追踪用户偏好/信念/行为模式，贝叶斯后验更新
- **记忆压缩**（MemPalace AAAK）：旧记忆按分组压缩为摘要，30x 压缩比

### v3.1 升级（5 项精度提升）
- **TF-IDF 加权重排**：从子串匹配升级为 BM25 式饱和 + 长度归一化
- **抽取式摘要压缩**：从截断拼接升级为 TF-IDF 句子打分 + 实体密度检测
- **贝叶斯信念更新**：从简单累加升级为后验概率 + 冲突检测 + 冲突历史
- **自适应检索**：QueryClassifier 根据查询类型动态调整 4 路权重
- **LRU 缓存 + 按需加载**：KG 从全内存改为 LRU + SQLite 按需加载

### v3.2 新增（自动存取中间件）
- **AutoMemoryMiddleware**：零侵入自动存取，LLM 无需关心记忆管理
  - 自动存储：拦截用户消息和 LLM 回复，自动喂给管线
  - 自动检索：从用户消息提取查询意图，4 路并行检索
  - 上下文注入：把记忆 + 心智模型信念注入 system prompt
- **3 个 HTTP API 端点**：`/api/v1/auto/pre_process`、`/api/v1/auto/post_process`、`/api/v1/auto/stats`
- **`--auto` CLI 参数**：启动时自动启用中间件

## 快速开始

### 环境要求

- Python >= 3.11
- 中文语义搜索使用**本地 ONNX 嵌入模型**（`models/bge-small-zh-v1.5/`，随包提供，
  无需联网下载、无需安装 torch/sentence-transformers）

### 安装

```bash
cd pudica-memory
pip install -r requirements.txt
# 或安装为 Python 包：
pip install -e .
```

### 配置 LLM（可选）

`llm` 段用于 reflect 洞察、L1 LLM 增强提取等**增强功能**；LLM 不可用时系统自动降级为
纯本地规则模式（实体提取、场景组织、检索全部可用，只是没有 LLM 生成的洞察）：

```json
{
  "llm": {
    "api_base": "http://你的LLM服务器:端口/v1",
    "api_key": "你的key",
    "model": "你的模型名"
  }
}
```

或用环境变量覆盖（推荐，免改 config.json）：

```bash
export UNIFIED_MEMORY_LLM_API_BASE="http://你的LLM:端口/v1"
export UNIFIED_MEMORY_LLM_API_KEY="你的key"
export UNIFIED_MEMORY_LLM_MODEL="你的模型名"
```

支持的变量：`UNIFIED_MEMORY_LLM_API_BASE`、`UNIFIED_MEMORY_LLM_API_KEY`、`UNIFIED_MEMORY_LLM_MODEL`、`UNIFIED_MEMORY_LLM_MAX_TOKENS`、`UNIFIED_MEMORY_LLM_TEMPERATURE`、`UNIFIED_MEMORY_STORAGE_DIR`。

### 集成到 Hermes Agent（MCP）

```bash
hermes mcp add pudica-memory \
  --command "C:/你的路径/python.exe" \
  --args "C:/你的路径/pudica-memory/run_mcp.py" \
  --env "PYTHONUTF8=1" \
  --env "UNIFIED_MEMORY_STORAGE_DIR=C:/你的路径/pudica-memory/data"
```

> **Windows 注意**：MCP stdio 子进程的环境必须显式包含
> `SYSTEMROOT`/`WINDIR`/`COMSPEC`/`USERPROFILE`/`HOMEDRIVE`/`HOMEPATH`，
> 否则会触发 WinError 10106（无法加载 DLL）和 `Path.home()` 失败。
> 修改源码后无需重启 Hermes：`taskkill /PID <mcp进程> /F`，Hermes 会自动重连加载新代码。

### 启动

**HTTP 模式（推荐用于调试和监控）：**

```bash
python run_http.py
# 或直接使用 CLI：
python -m unified_memory.main --http --host 127.0.0.1 --port 8000
```

**MCP 模式（供 AI Agent 调用）：**

```bash
python run_mcp.py
# 或直接使用 CLI：
python -m unified_memory.main --mcp
```

**Windows 用户：**
双击 `start_http.bat` 即可启动 HTTP 服务。

### 验证

```bash
# HTTP 模式
curl http://127.0.0.1:8000/api/v1/health

# 搜索记忆
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"关键词","top_k":10}'

# 写入记忆
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"content":"要记住的内容","source":"chat"}'
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/stats` | 系统统计 |
| POST | `/api/v1/search` | 多策略检索 |
| POST | `/api/v1/ingest` | 管线摄取 |
| GET | `/api/v1/kg/query?entity=xxx` | 知识图谱查询 |
| POST | `/api/v1/reflect` | 手动触发 reflect |
| POST | `/api/v1/consolidate` | 手动触发 consolidation |
| GET | `/api/v1/mempalace/wings` | 列出 wing |
| GET | `/api/v1/mempalace/rooms` | 列出 room |
| GET | `/api/v1/mempalace/drawers` | 列出 drawer |
| GET | `/api/v1/mempalace/status` | 状态 |
| GET | `/api/v1/mempalace/taxonomy` | 分类树 |
| POST | `/api/v1/auto/pre_process` | 自动存取预处理（存储+检索+注入） |
| POST | `/api/v1/auto/post_process` | 自动存储 LLM 回复 |
| GET | `/api/v1/auto/stats` | 中间件统计 |

## MCP 工具

兼容 mempalace 工具名（8 个）：
- `mempalace_status` — 系统总览
- `mempalace_search` — 语义搜索
- `mempalace_add_drawer` — 写入内容
- `mempalace_list_wings` — 列出所有 wing
- `mempalace_list_rooms` — 列出 room
- `mempalace_list_drawers` — 列出 drawer
- `mempalace_get_drawer` — 获取单个 drawer
- `mempalace_get_taxonomy` — wing→room→count 树

unified-memory 工具（8 个）：
- `memory_ingest` — 写入消息，自动触发管线
- `memory_search` — 多策略检索
- `kg_query` — 知识图谱查询
- `pipeline_run` — 手动触发管线
- `reflect_trigger` — 手动触发 reflect
- `consolidate_trigger` — 手动触发 consolidation
- `system_health` — 健康检查
- `system_stats` — 系统统计

v3.0 新增工具（5 个）：
- `mental_models_query` — 查询用户心智模型/信念
- `mental_models_strong` — 获取高置信度信念
- `memory_compress` — 手动触发记忆压缩
- `verbatim_recall` — 逐字回溯原始记忆
- `memory_context` — 获取完整上下文（记忆+心智模型）

## 项目结构

```
pudica-memory/
├── config.json              # 配置文件
├── requirements.txt         # 依赖清单
├── pyproject.toml           # Python 包定义
├── README.md
├── run_mcp.py               # MCP 启动入口
├── run_http.py              # HTTP 启动入口
├── start_http.bat           # Windows 快捷启动
├── seed.py                  # 种子数据脚本
├── data/                    # 数据目录（自动创建）
│   ├── unified_memory.db    # SQLite 数据库
│   └── chroma/              # ChromaDB 持久化
└── src/unified_memory/      # 全部源码
    ├── main.py              # 入口（支持 --mcp/--http/--test/--auto）
    ├── config.py            # 配置管理
    ├── api/                 # MCP + HTTP 接口
    ├── middleware/          # 自动存取中间件（v3.2）
    ├── store/               # 存储层（SQLite + ChromaDB + KG + 心智模型）
    ├── pipeline/            # 4 层管线
    ├── search/              # 多策略检索（含重排器 + 自适应分类）
    └── tasks/               # 后台任务（reflect + consolidation + compression）
```

## 架构

```
API 层 (MCP + HTTP)
    │
管线层  L0 → L1 → L2 → L3
    │
存储层  SQLite + ChromaDB + 知识图谱
    │
搜索层  语义 | BM25 | 图链接 | 时间 → RRF 融合
```

## CLI 参数

```bash
python -m unified_memory.main --help

usage: main.py [-h] [--mcp] [--http] [--host HOST] [--port PORT]
               [--test] [--config CONFIG] [--auto]
               [--log-level {DEBUG,INFO,WARNING,ERROR}]

pudica-Memory — 统一记忆系统

options:
  -h, --help            显示帮助信息
  --mcp                 启动 MCP 服务器 (stdio 模式)
  --http                启动 HTTP 服务器 (REST API)
  --host HOST           HTTP 监听地址 (默认 127.0.0.1)
  --port PORT           HTTP 监听端口 (默认 8000)
  --test                运行集成测试
  --config CONFIG       配置文件路径
  --auto                启用自动存取中间件（消息自动存储+记忆自动检索+上下文自动注入）
  --log-level {DEBUG,INFO,WARNING,ERROR}
                         日志级别 (默认 INFO)
```

## 常见问题

- **搜索返回空 / 中文匹配差**：确认嵌入器加载的是本地 ONNX `bge-small-zh-v1.5`（512 维），
  日志中出现 `bge-small-zh ONNX 嵌入器已加载` 即正常。若向量库为空，运行
  `python rebuild_vectors.py` 从 SQLite 重建向量索引。
- **BM25 检索无结果**：v2.5.0 起 FTS5 使用 trigram 分词器 + 同步触发器，
  启动时自动迁移旧表结构；查询词少于 3 字符时 trigram 无法匹配（属正常限制）。
- **L1 提取失败**：检查 `config.json` 的 `llm` 段是否指向你真实可用的模型；
  未配置时自动使用本地规则提取，不影响写入与检索。
- **数据存储位置**：默认在 `data/` 目录，可用环境变量 `UNIFIED_MEMORY_STORAGE_DIR` 覆盖。
- **Python 找不到**：确保 Python >= 3.11 已安装，或在命令前设置完整 Python 路径。
- **MCP 工具报 WinError 10106**：MCP 子进程 env 缺少 Windows 系统变量，见上方 MCP 集成说明。
- **修改源码后工具仍是旧行为**：kill MCP 子进程让 Hermes 自动重连（无需重启 Hermes）。