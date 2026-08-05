# Changelog

## v3.2.0 (2026-08-04)

### 自动存取中间件（AutoMemoryMiddleware）

在 LLM/Agent 和 pudica-memory 之间加一层透明中间件，实现零侵入自动存取：
- **pre_process()**：自动存储用户消息 → 4 路并行检索 → 上下文注入 system prompt
- **post_process()**：自动存储 LLM 回复（过滤工具调用标记）
- **process_turn()**：完整对话轮次快捷方法
- 过滤规则：短消息/语气词/系统指令自动跳过
- 默认关闭，通过 `--auto` CLI 参数或 `config.middleware.enabled` 启用

新增文件：
- `middleware/auto_memory.py` — 中间件实现（~280 行）
- `middleware/__init__.py` — 包导出

新增配置（MiddlewareConfig）：
- `auto_store`/`auto_search`/`auto_inject` 三个独立开关
- `min_message_length`/`search_top_k`/`max_context_items`/`max_context_chars`
- `store_llm_response` — 是否自动存储 LLM 回复

新增 HTTP API：
- `POST /api/v1/auto/pre_process` — 自动存取预处理
- `POST /api/v1/auto/post_process` — 自动存储 LLM 回复
- `GET /api/v1/auto/stats` — 中间件统计

## v3.1.0 (2026-08-04)

### 5 项精度提升

1. **TF-IDF 加权重排**（`search/reranker.py`）
   - 从子串匹配升级为 BM25 式 TF 饱和（k1=1.2, b=0.75）+ IDF 加权
   - 综合评分：0.45×TF-IDF + 0.15×来源多样性 + 0.25×RRF + 0.15×长度惩罚

2. **抽取式摘要压缩**（`tasks/compression.py`）
   - 从截断拼接升级为 TF-IDF 句子打分 + 位置加成 + 实体密度检测
   - 自动提取含数字/日期/人名/因果关系的句子作为关键事实

3. **贝叶斯信念更新**（`store/mental_models.py`）
   - 从简单 confidence += delta 升级为贝叶斯后验更新
   - 信念冲突检测（Jaccard 相似度 < 0.4 视为矛盾）
   - 冲突历史追踪（metadata 中保留最近 20 条冲突记录）

4. **自适应检索**（`search/fusion.py`）
   - 新增 QueryClassifier：4 类查询分类（temporal/entity/keyword/semantic）
   - 根据查询类型动态调整 RRF 权重（如时间查询提权 temporal ×1.6）

5. **LRU 缓存 + 按需加载**（`store/kg.py`）
   - 从全内存 dict 改为 OrderedDict LRU 缓存（默认 10000 上限）
   - 轻量级 _all_entity_names 集合做存在性检查
   - _relation_index 索引实现 O(1) 邻居查找
   - load_from_db() 只加载名称和关系索引，不加载完整 Entity 对象

### 适配修改
- `tasks/consolidation.py`：_reassign_relations 重建 _relation_index
- `tasks/consolidation.py`：_upgrade 使用 _all_entity_names 快照
- `pipeline/engine.py`：KG 统计适配 LRU（len(_all_entity_names)）

## v3.0.0 (2026-08-04)

### 融合三大开源记忆体

综合 TencentDB Agent Memory、MemPalace、Hindsight 的设计精华，新增 4 个模块 + 5 个 MCP 工具。

#### 新增模块

1. **Verbatim 逐字存储**（MemPalace 启发）
   - `pipeline/engine.py` 中新增 `_store_verbatim()`
   - 管线处理前保存原始输入到 verbatim 表
   - 支持精确回溯（`verbatim_recall` 工具）

2. **Cross-Encoder 重排器**（Hindsight 启发）
   - `search/reranker.py` — 新建
   - 支持 heuristic（TF-IDF 加权）和 llm 两种策略
   - 在 RRF 融合结果之上进行二次精排

3. **心智模型/信念系统**（Hindsight 启发）
   - `store/mental_models.py` — 新建
   - 4 类信念：preference / belief / behavior / knowledge_level
   - 贝叶斯后验更新 + 冲突检测
   - format_for_context() — 将信念格式化为 Agent 上下文

4. **记忆压缩**（MemPalace AAAK 启发）
   - `tasks/compression.py` — 新建
   - 按 wing/room 分组压缩旧记忆
   - LLM 压缩 + 启发式抽取式摘要 fallback
   - 压缩摘要写入 compressed_memories 表

#### 新增 MCP 工具（5 个）
- `mental_models_query` — 查询用户心智模型/信念
- `mental_models_strong` — 获取高置信度信念
- `memory_compress` — 手动触发记忆压缩
- `verbatim_recall` — 逐字回溯原始记忆
- `memory_context` — 获取完整上下文（记忆+心智模型）

#### 配置新增
- `RerankerConfig` — 重排器配置
- `MentalModelConfig` — 心智模型配置
- `CompressionConfig` — 压缩任务配置

#### 集成修改
- `pipeline/engine.py`：集成 Verbatim 存储、Reranker、心智模型自动更新
- `api/tools.py`：注册 5 个新工具，存储 _temp_engine 供中间件访问
- `tasks/scheduler.py`：新增 Compressor 定时任务
- `store/sqlite_store.py`：新增 mental_models 和 compressed_memories 表

## v2.5.0 (2026-08-03)

深度代码审查 + 修复版。修复了 4 个崩溃级/静默失效 bug 与若干隐患，并引入本地 ONNX 中文嵌入器。

### 🚨 崩溃级修复

- **reflect_trigger 100% 崩溃**：`REFLECT_PROMPT.format()` 中 JSON 示例的 `{}` 未转义，
  触发 `KeyError: '"insights"'`，Reflect 任务从未成功执行。
  → `tasks/reflect.py`：JSON 示例花括号全部转义为 `{{}}`（保留 `{context}` 占位符）。
- **consolidate_trigger 崩溃**：`from datetime import time` 遮蔽标准库 `time`，
  `trigger_consolidate()`/`trigger_reflect()` 调用 `time.time()` 报
  `datetime.time has no attribute 'time'`。
  → `tasks/scheduler.py`：改为 `import time as time_mod` + `from datetime import time as dt_time`。

### 📉 静默失效修复

- **BM25 全文检索 100% 失效**（三重问题）：
  1. 数据库中 `memories_fts` 是旧版单列结构（`fts5(content, content_rowid=id)`），
     与代码假设的 `(id, content, wing, room)` 五列不符，BM25 SQL 报 `no such column`，
     被 `TEMPREngine` 的 `return_exceptions=True` 静默吞掉；
  2. FTS5 外部内容表没有任何同步触发器（`memories_fts` 表 0 行）；
  3. 默认 unicode61 分词器对连续中文按整段分词，中文子串匹配失效。
  → `store/sqlite_store.py`：启动时自动检测旧结构并重建为 **trigram 分词器** 五列结构，
    新增 `memories_ai/ad/au` 三个同步触发器 + `rebuild` 回填；
  → `search/sparse.py`：`SELECT` 用 `bm25(memories_fts)` 函数替代非法 `rank` 列，
    过滤 <3 字符词条（trigram 要求）。
- **知识图谱永不更新**：`engine._flush_buffer` 传给 L2 的 `merged_extracted` 只有
  `entities`/`relations`，缺 `summary`/`time_range`，`L2SceneOrganizer.organize()`
  第一行 `if not extracted.get("summary")` 直接 `return None`，L2 与 KG 全部失效。
  → `pipeline/engine.py`：合并各消息的 summary 与时间范围后传给 L2。
- **memories.metadata 列永远为空**：engine 插入 SQLite 时未写 `metadata` 列。
  → `pipeline/engine.py`：插入时序列化 fact_type/source/created_at 等 JSON。

### ⚠️ 隐患修复

- **shutdown 可能清空向量库**：`UnifiedMemoryApp.shutdown()` 调用
  `chroma._client.reset()`（chromadb 1.5.9 默认禁用但若配置允许会清空全部向量）。
  → `main.py`：改为 `client.close()` 安全释放。
- **RRF 融合结果文本重复**：同一文档被多策略命中时 `join("\n")` 产生重复文本。
  → `search/fusion.py`：文本去重。
- **L2 场景创建逻辑残留**：`_find_or_create_scene` 中 `or True` 调试残留。
  → `pipeline/l2_scene.py`：清理。
- **实体名不精确**：L1 正则贪婪匹配把整句当实体名（"用户参与了云南天海科技有限公司"）。
  → `pipeline/l1_extractor.py`：新增实体名清洗（噪声前缀剥离 + 按实体类型后缀定位裁剪
    + "的/在"分隔截断 + 类型后缀隔离防串扰）。

### ✨ 新能力

- **本地 ONNX 中文嵌入器**（v2.4.x 引入，随本版正式文档化）：
  - 模型：`Xenova/bge-small-zh-v1.5`（int8 量化，24MB）from ModelScope，
    存于 `models/bge-small-zh-v1.5/`；
  - 实现：`store/onnx_zh_embedder.py`，纯 onnxruntime，**无需 torch/sentence-transformers**；
  - `store/chroma_store.py` 的 `get_embedder()` 优先加载本地 ONNX 嵌入器；
  - 彻底解决此前"语义搜索返回空"问题（原回退链依赖 HuggingFace 下载，网络慢时写入无向量）。
- **运维脚本**：
  - `rebuild_vectors.py` — 从 SQLite 重建 ChromaDB 向量索引；
  - `cleanup_test_data.py` — 清理验证产生的测试记忆/场景/实体。

### 🔧 修复验证

- 独立进程综合验证 11/11 通过（FTS 触发器/回填、BM25 中文检索、L2 场景、KG 实体、
  metadata 落库、reflect/consolidate 不崩溃、多策略融合、文本去重、数据保留）。
- MCP 端到端：写入 → L1 清洗 → KG 提取（`kg_query("云岭集团")` 命中干净实体）→
  语义/BM25/时间三路融合检索。

### 已知边界

- 实体提取为启发式正则，个别长句仍可能带少量噪声（如 location 丢失"云南"前缀）；
- `WriteBuffer`/`SQLiteStore`/`SQLiteStore.search_fts` 为未接入的死代码（保留待后续清理）；
- Reflect 的"洞察/知识缺口"等 LLM 输出依赖外部 LLM（config.json `llm` 段），
  LLM 不可用时自动降级为本地规则模式，不崩溃。
