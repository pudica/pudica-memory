"""pudica-memory 错误码体系 — 借鉴 DSH 的 LlmError 设计。

每个错误有唯一代码、HTTP 状态码映射、用户可读消息。

错误码分类：
    AUTH_*            认证相关（401/403）
    MISSING_*         缺少必要配置（500）
    TRANSPORT_*       网络/通信错误（502/503）
    TIMEOUT_*         超时（504）
    STORAGE_*         存储错误（500）
    PIPELINE_*        管线处理错误（500）
    VALIDATION_*      参数校验错误（400）
    EMPTY_*           空结果（404）
    RATE_LIMITED_*    限流（429）

用法:
    from unified_memory.errors import PipelineError, ErrorCode, raise_with_code

    # 抛异常
    raise PipelineError("ChromaDB 写入失败", code=ErrorCode.STORAGE_WRITE_FAILED)

    # 或
    raise_with_code(ErrorCode.AUTH_INVALID_KEY, "API key 无效")

    # 在 try/except 中判断
    try:
        ...
    except PipelineError as e:
        if e.code == ErrorCode.STORAGE_WRITE_FAILED:
            handle_retry()
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """错误码枚举。

    命名规则：{分类}_{具体错误}
    分类：AUTH, MISSING, TRANSPORT, TIMEOUT, STORAGE, PIPELINE, VALIDATION, EMPTY, RATE_LIMITED
    """

    # --- 认证错误 ---
    AUTH_INVALID_KEY = "AUTH_INVALID_KEY"
    """提供的 API key 无效。"""
    AUTH_MISSING_KEY = "AUTH_MISSING_KEY"
    """需要 API key 但未提供。"""
    AUTH_INSUFFICIENT = "AUTH_INSUFFICIENT"
    """API key 有效但权限不足（如 user 级 key 尝试调用 system 级工具）。"""

    # --- 缺少必要配置 ---
    MISSING_CONFIG = "MISSING_CONFIG"
    """缺少必要的配置项。"""
    MISSING_ENV_VAR = "MISSING_ENV_VAR"
    """环境变量未设置（如 UNIFIED_MEMORY_LLM_API_KEY）。"""
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    """缺少依赖模块（如 chromadb/onnxruntime）。"""

    # --- 网络/通信错误 ---
    TRANSPORT_CONNECTION = "TRANSPORT_CONNECTION"
    """连接失败（如数据库连接、HTTP 请求）。"""
    TRANSPORT_TIMEOUT = "TRANSPORT_TIMEOUT"
    """连接超时。"""
    TRANSPORT_REFUSED = "TRANSPORT_REFUSED"
    """连接被拒绝。"""

    # --- 超时 ---
    TIMEOUT_LLM = "TIMEOUT_LLM"
    """LLM 调用超时。"""
    TIMEOUT_CHROMA = "TIMEOUT_CHROMA"
    """ChromaDB 操作超时。"""
    TIMEOUT_PIPELINE = "TIMEOUT_PIPELINE"
    """管线处理超时。"""

    # --- 存储错误 ---
    STORAGE_WRITE_FAILED = "STORAGE_WRITE_FAILED"
    """写入存储失败（SQLite 或 ChromaDB）。"""
    STORAGE_READ_FAILED = "STORAGE_READ_FAILED"
    """读取存储失败。"""
    STORAGE_DUPLICATE = "STORAGE_DUPLICATE"
    """数据重复（如 SHA256 hash 冲突）。"""
    STORAGE_CONSISTENCY = "STORAGE_CONSISTENCY"
    """存储一致性问题（如 SQLite 写入成功但 ChromaDB 失败，产生孤儿记录）。"""

    # --- 管线错误 ---
    PIPELINE_STAGE_FAILED = "PIPELINE_STAGE_FAILED"
    """管线步骤执行失败。"""
    PIPELINE_STAGE_NOT_FOUND = "PIPELINE_STAGE_NOT_FOUND"
    """引用的管线步骤未注册。"""
    PIPELINE_BUFFER_OVERFLOW = "PIPELINE_BUFFER_OVERFLOW"
    """缓冲区溢出（积累太多未处理消息）。"""

    # --- 参数校验错误 ---
    VALIDATION_INVALID_PARAM = "VALIDATION_INVALID_PARAM"
    """参数值无效。"""
    VALIDATION_MISSING_PARAM = "VALIDATION_MISSING_PARAM"
    """缺少必填参数。"""

    # --- 空结果 ---
    EMPTY_RESULTS = "EMPTY_RESULTS"
    """查询未返回结果。"""
    EMPTY_CONTENT = "EMPTY_CONTENT"
    """内容为空（无法处理）。"""

    # --- 限流 ---
    RATE_LIMITED = "RATE_LIMITED"
    """请求频率过高，被限流。"""


# HTTP 状态码映射
HTTP_STATUS_MAP: dict[ErrorCode, int] = {
    ErrorCode.AUTH_INVALID_KEY: 401,
    ErrorCode.AUTH_MISSING_KEY: 401,
    ErrorCode.AUTH_INSUFFICIENT: 403,
    ErrorCode.MISSING_CONFIG: 500,
    ErrorCode.MISSING_ENV_VAR: 500,
    ErrorCode.MISSING_DEPENDENCY: 500,
    ErrorCode.TRANSPORT_CONNECTION: 502,
    ErrorCode.TRANSPORT_TIMEOUT: 504,
    ErrorCode.TRANSPORT_REFUSED: 502,
    ErrorCode.TIMEOUT_LLM: 504,
    ErrorCode.TIMEOUT_CHROMA: 504,
    ErrorCode.TIMEOUT_PIPELINE: 504,
    ErrorCode.STORAGE_WRITE_FAILED: 500,
    ErrorCode.STORAGE_READ_FAILED: 500,
    ErrorCode.STORAGE_DUPLICATE: 409,
    ErrorCode.STORAGE_CONSISTENCY: 500,
    ErrorCode.PIPELINE_STAGE_FAILED: 500,
    ErrorCode.PIPELINE_STAGE_NOT_FOUND: 500,
    ErrorCode.PIPELINE_BUFFER_OVERFLOW: 503,
    ErrorCode.VALIDATION_INVALID_PARAM: 400,
    ErrorCode.VALIDATION_MISSING_PARAM: 400,
    ErrorCode.EMPTY_RESULTS: 404,
    ErrorCode.EMPTY_CONTENT: 400,
    ErrorCode.RATE_LIMITED: 429,
}


class PipelineError(Exception):
    """管线错误（带错误码）。

    用法:
        raise PipelineError("ChromaDB 写入失败", code=ErrorCode.STORAGE_WRITE_FAILED)
        raise PipelineError("连接超时", code=ErrorCode.TRANSPORT_TIMEOUT, details={"host": "..."})
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.PIPELINE_STAGE_FAILED,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.code = code
        self.details = details or {}
        self.cause = cause
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "error": True,
            "code": self.code.value,
            "message": str(self),
            "details": self.details,
            "http_status": HTTP_STATUS_MAP.get(self.code, 500),
        }

    def __str__(self) -> str:
        base = f"[{self.code.value}] {super().__str__()}"
        if self.cause:
            return f"{base} (cause: {self.cause})"
        return base


class AuthError(PipelineError):
    """认证错误。"""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.AUTH_INVALID_KEY,
                 details: Optional[dict] = None, cause: Optional[Exception] = None):
        super().__init__(message, code=code, details=details, cause=cause)


class StorageError(PipelineError):
    """存储错误。"""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_WRITE_FAILED,
                 details: Optional[dict] = None, cause: Optional[Exception] = None):
        super().__init__(message, code=code, details=details, cause=cause)


class TransportError(PipelineError):
    """网络/通信错误。"""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.TRANSPORT_CONNECTION,
                 details: Optional[dict] = None, cause: Optional[Exception] = None):
        super().__init__(message, code=code, details=details, cause=cause)


class ValidationError(PipelineError):
    """参数校验错误。"""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.VALIDATION_INVALID_PARAM,
                 details: Optional[dict] = None, cause: Optional[Exception] = None):
        super().__init__(message, code=code, details=details, cause=cause)


def raise_with_code(code: ErrorCode, message: str, **details) -> None:
    """用错误码抛异常。

    自动选择异常类型：
        AUTH_* → AuthError
        STORAGE_* → StorageError
        TRANSPORT_* → TransportError
        VALIDATION_* → ValidationError
        其他 → PipelineError

    用法:
        raise_with_code(ErrorCode.AUTH_INVALID_KEY, "API key 无效")
        raise_with_code(ErrorCode.STORAGE_WRITE_FAILED, "写入失败", table="memories")
    """
    code_str = code.value
    if code_str.startswith("AUTH_"):
        raise AuthError(message, code=code, details=details)
    elif code_str.startswith("STORAGE_"):
        raise StorageError(message, code=code, details=details)
    elif code_str.startswith("TRANSPORT_"):
        raise TransportError(message, code=code, details=details)
    elif code_str.startswith("VALIDATION_"):
        raise ValidationError(message, code=code, details=details)
    else:
        raise PipelineError(message, code=code, details=details)