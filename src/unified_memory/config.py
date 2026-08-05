"""Unified Memory — 配置管理模块。"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM 调用配置。"""
    api_base: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:7b"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: float = 30.0


@dataclass
class SQLiteConfig:
    """SQLite 存储配置。"""
    db_path: str = ""  # 由 data_dir 自动计算
    pool_size: int = 5
    timeout: float = 30.0
    flush_interval: float = 5.0
    batch_size: int = 100


@dataclass
class ChromaConfig:
    """ChromaDB 存储配置。"""
    persist_dir: str = ""  # 由 data_dir 自动计算
    collection_name: str = "memories"
    hnsw_ef_construction: int = 100
    hnsw_m: int = 16
    hnsw_search_ef: int = 50


@dataclass
class PipelineConfig:
    """管线配置。"""
    l0_maxsize: int = 1000
    l1_batch_size: int = 5
    l1_idle_timeout: float = 60.0
    l2_null_threshold: int = 4
    l2_timeout: float = 300.0
    l3_default_budget: int = 4000
    verbatim_enabled: bool = True  # MemPalace: 逐字存储原始输入


@dataclass
class RerankerConfig:
    """重排器配置（Hindsight Cross-Encoder）。"""
    enabled: bool = True
    # 重排策略: "llm" 使用 LLM 打分, "heuristic" 使用启发式打分
    strategy: str = "heuristic"
    # 重排候选数量（从 RRF 结果中取 top_n 重排）
    top_n: int = 20
    # 重排后返回的最终数量
    final_k: int = 10
    # LLM 重排时的最大并发数
    max_concurrent: int = 4


@dataclass
class MentalModelConfig:
    """心智模型配置（Hindsight Mental Models）。"""
    enabled: bool = True
    # 信念更新阈值：新证据超过此值才更新信念
    belief_update_threshold: float = 0.6
    # 最大信念数量
    max_beliefs: int = 100
    # 信念过期时间（秒），0 = 永不过期
    belief_ttl: float = 0


@dataclass
class CompressionConfig:
    """记忆压缩配置（MemPalace AAAK-inspired）。"""
    enabled: bool = True
    # 触发压缩的记忆年龄阈值（秒），默认 7 天
    min_age_seconds: float = 604800
    # 单次压缩最大处理条数
    batch_size: int = 50
    # 压缩后保留的原始记忆比例（0-1），其余归档
    keep_ratio: float = 0.2
    # 压缩任务执行间隔（秒）
    interval: int = 86400  # 每天一次


@dataclass
class MiddlewareConfig:
    """自动存取中间件配置（AutoMemoryMiddleware）。"""
    enabled: bool = False  # 默认关闭，需显式启用
    # 自动存储用户消息
    auto_store: bool = True
    # 自动检索相关记忆
    auto_search: bool = True
    # 自动注入记忆到 system prompt
    auto_inject: bool = True
    # 消息最小长度（字符），短于此值跳过存储和检索
    min_message_length: int = 5
    # 自动检索返回条数
    search_top_k: int = 10
    # 注入上下文的最大记忆条数
    max_context_items: int = 5
    # 注入上下文的最大字符数
    max_context_chars: int = 2000
    # 是否自动存储 LLM 回复
    store_llm_response: bool = True


@dataclass
class TaskConfig:
    """后台任务配置。"""
    reflect_cron: str = "0 2 * * *"  # 每天凌晨 2 点
    consolidation_interval: int = 14400  # 每 4 小时
    dedup_threshold: float = 0.85
    compression_interval: int = 86400  # 每天一次（MemPalace 压缩）


@dataclass
class Config:
    """统一记忆系统全局配置。"""
    data_dir: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    mental_model: MentalModelConfig = field(default_factory=MentalModelConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    log_level: str = "INFO"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8080
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    _loaded: bool = False

    def __post_init__(self) -> None:
        """初始化后自动计算路径。"""
        if not self.data_dir:
            # 优先使用环境变量，其次用户主目录下的 .unified_memory/data，
            # 最后回退到源码目录旁的 data/（仅源码运行时正确）
            env_dir = os.environ.get("UNIFIED_MEMORY_STORAGE_DIR")
            if env_dir:
                self.data_dir = env_dir
            else:
                home_data = os.path.join(os.path.expanduser("~"), ".unified_memory", "data")
                # 检查 __file__ 是否在 site-packages 中（pip install 场景）
                file_dir = os.path.dirname(__file__)
                is_installed = "site-packages" in file_dir or "dist-packages" in file_dir
                if is_installed:
                    self.data_dir = home_data
                else:
                    # 源码运行：项目根目录下的 data/
                    self.data_dir = os.path.join(
                        os.path.dirname(os.path.dirname(file_dir)),
                        "data",
                    )
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """解析所有路径，确保 Windows 兼容。"""
        os.makedirs(self.data_dir, exist_ok=True)
        if not self.sqlite.db_path:
            self.sqlite.db_path = os.path.join(self.data_dir, "unified_memory.db")
        os.makedirs(os.path.dirname(self.sqlite.db_path), exist_ok=True)
        if not self.chroma.persist_dir:
            self.chroma.persist_dir = os.path.join(self.data_dir, "chroma")
        os.makedirs(self.chroma.persist_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 配置加载/保存
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """从 JSON 文件加载配置，不存在则使用默认值。

        默认查找路径（按优先级）：
        1. 显式传入的 path
        2. 环境变量 UNIFIED_MEMORY_CONFIG
        3. 项目根目录下的 config.json

        环境变量 UNIFIED_MEMORY_STORAGE_DIR 可覆盖 data_dir。
        """
        if path is None:
            path = os.environ.get("UNIFIED_MEMORY_CONFIG")
        if path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config.json",
            )
            if os.path.exists(default_path):
                path = default_path
        cfg = cls()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply_dict(cfg, data)
            cfg._resolve_paths()
        # 环境变量 UNIFIED_MEMORY_STORAGE_DIR 可以覆盖 JSON 中的 data_dir
        storage_dir = os.environ.get("UNIFIED_MEMORY_STORAGE_DIR")
        if storage_dir:
            cfg.data_dir = storage_dir
            cfg._resolve_paths()
        # 环境变量覆盖 LLM 端点（免改 config.json，便于多智能体共用同一份代码）
        _override_env(cfg)
        cfg._loaded = True
        return cfg

    def save(self, path: str) -> None:
        """保存配置到 JSON 文件。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _to_dict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("配置已保存到 %s", path)


def _apply_dict(obj: Any, data: dict) -> None:
    """递归地将字典应用到 dataclass 对象。"""
    for key, value in data.items():
        if hasattr(obj, key):
            field_val = getattr(obj, key)
            if hasattr(field_val, "__dataclass_fields__") and isinstance(value, dict):
                _apply_dict(field_val, value)
            else:
                setattr(obj, key, value)


LLM_ENV_MAP = {
    "UNIFIED_MEMORY_LLM_API_BASE": ("llm", "api_base"),
    "UNIFIED_MEMORY_LLM_API_KEY": ("llm", "api_key"),
    "UNIFIED_MEMORY_LLM_MODEL": ("llm", "model"),
    "UNIFIED_MEMORY_LLM_MAX_TOKENS": ("llm", "max_tokens"),
    "UNIFIED_MEMORY_LLM_TEMPERATURE": ("llm", "temperature"),
}


def _override_env(cfg: "Config") -> None:
    """用环境变量覆盖 LLM 配置，免改 config.json。

    支持：
      UNIFIED_MEMORY_LLM_API_BASE   → llm.api_base
      UNIFIED_MEMORY_LLM_API_KEY    → llm.api_key
      UNIFIED_MEMORY_LLM_MODEL      → llm.model
      UNIFIED_MEMORY_LLM_MAX_TOKENS → llm.max_tokens
      UNIFIED_MEMORY_LLM_TEMPERATURE→ llm.temperature
    """
    for env_name, (section, field) in LLM_ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is None:
            continue
        sub = getattr(cfg, section, None)
        if sub is None:
            continue
        if field in ("max_tokens",):
            try:
                val = int(val)
            except ValueError:
                logger.warning("%s 非整数，忽略: %s", env_name, val)
                continue
        elif field == "temperature":
            try:
                val = float(val)
            except ValueError:
                logger.warning("%s 非数字，忽略: %s", env_name, val)
                continue
        setattr(sub, field, val)


def _to_dict(obj: Any) -> dict:
    """递归地将 dataclass 转换为字典。"""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if not field_name.startswith("_") and not callable(value):
                result[field_name] = _to_dict(value)
        return result
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj