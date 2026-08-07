"""pipeline/encoding_repair.py — Windows 编码修复预处理层。

移植自 MemPalace v3.5.0 encoding_repair.py，专为 pudica-memory 精简。

解决 Windows 环境下常见的 UTF-8 mojibake 问题：
1. CP1252 延续字节解码（UTF-8 字节被误读为 Windows-1252）
2. NUL 字节清理（某些工具写入时残留的零字节）
3. SQLite 路径百分号编码（特殊字符路径）

参考 MemPalace encoding_repair.py 的 repair_mojibake 逻辑。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# CP1252 中未定义的字节值（保留为 C1 控制码点）
_UNDEFINED_CP1252_BYTES = frozenset({0x81, 0x8D, 0x8F, 0x90, 0x9D})

# UTF-8-as-CP1252 的可见前导字节（高频 mojibake 特征）
# C2/C3 → Â/Ã, E2 → â, F0 → ð, EF → ï
_CONTINUATION_CHARS = "".join(
    bytes([b]).decode("cp1252")
    for b in range(0x80, 0xC0)
    if b not in _UNDEFINED_CP1252_BYTES
)
_CONTINUATION_CLASS = re.escape(_CONTINUATION_CHARS)

# 高置信度 mojibake 运行匹配
# 排除 C4/C5 (Ä/Å) 避免误伤合法科学文本（如 Å²）
_HIGH_CONFIDENCE_RUN = re.compile(
    rf"(?:"
    rf"[ÂÃ][{_CONTINUATION_CLASS}]"
    rf"|â[{_CONTINUATION_CLASS}]{{2}}"
    rf"|ð[{_CONTINUATION_CLASS}]{{3}}"
    rf"|ï[{_CONTINUATION_CLASS}]{{2}}"
    rf")+"
)


def _cp1252_character(byte_value: int) -> str:
    """将遗留字节映射到 Windows-1252 字符。"""
    if byte_value in _UNDEFINED_CP1252_BYTES:
        return chr(byte_value)
    return bytes([byte_value]).decode("cp1252")


def _encode_mojibake_candidate(text: str) -> bytes:
    """恢复 mojibake 候选的原始字节（含未定义 CP1252 值）。"""
    raw = bytearray()
    for character in text:
        codepoint = ord(character)
        if codepoint in _UNDEFINED_CP1252_BYTES:
            raw.append(codepoint)
        else:
            raw.extend(character.encode("cp1252"))
    return bytes(raw)


def _decode_high_confidence_run(match: re.Match) -> str:
    """解码一个高置信度 mojibake 运行匹配。"""
    candidate = match.group(0)
    try:
        return _encode_mojibake_candidate(candidate).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return candidate


def repair_mojibake_once(text: str) -> str:
    """修复一层高置信度 UTF-8-as-CP1252 mojibake。"""
    return _HIGH_CONFIDENCE_RUN.sub(_decode_high_confidence_run, text)


def repair_mojibake(text: str, *, max_passes: int = 3) -> str:
    """修复重复的高置信度 mojibake 层，直到稳定。

    Args:
        text: 可能包含 mojibake 的文本
        max_passes: 最大修复轮次（默认 3）

    Returns:
        修复后的文本
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    current = text
    for _ in range(max_passes):
        repaired = repair_mojibake_once(current)
        if repaired == current:
            break
        current = repaired
    return current


# NUL 字节清理
_NUL_BYTE = re.compile(r"\x00+")


def clean_nul_bytes(text: str) -> str:
    """清理文本中的 NUL 字节（\x00）。

    Windows 环境下某些工具写入时会在文本中残留 NUL 字节，
    导致 SQLite 存储和 ChromaDB 向量化时报错。

    Args:
        text: 可能包含 NUL 字节的文本

    Returns:
        清理后的文本
    """
    return _NUL_BYTE.sub("", text)


def sqlite_escape_path(path: str) -> str:
    """对特殊字符路径进行百分号编码，适配 SQLite 路径存储。

    Windows 路径中的特殊字符（如 #, %, &, 空格）在 SQLite
    查询中可能导致解析错误。本函数仅对路径中的特殊字符编码。

    Args:
        path: 原始路径

    Returns:
        编码后的路径
    """
    # 仅对路径中的特殊字符编码（保留 / 和 \）
    encoded = ""
    for ch in path:
        if ch in ('#', '%', '&', '?', '='):
            encoded += f"%{ord(ch):02X}"
        else:
            encoded += ch
    return encoded


def repair_document(text: str) -> str:
    """完整的文档编码修复管线：清理 NUL → 修复 mojibake。

    这是 ingest 管道的预处理入口。先清理 NUL 字节（防止下游
    向量化引擎报错），再修复 mojibake（恢复被误读的中文）。

    Args:
        text: 原始文档文本

    Returns:
        修复后的文档文本
    """
    if not text:
        return text
    step1 = clean_nul_bytes(text)
    step2 = repair_mojibake(step1)
    if step2 != text:
        logger.debug("Encoding repair: cleaned %d NUL bytes, repaired mojibake",
                     len(text) - len(step1))
    return step2