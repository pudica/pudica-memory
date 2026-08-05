"""pipeline/l0_dedup.py — L0：消息去重 + LRU 缓存。

参考 TencentDB offload/state-manager.ts 的 pendingToolPairs + processedToolCallIds。
文档 6.1 节 L0Dedup 实现。
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class L0Dedup:
    """L0：消息去重 + LRU 缓存。

    使用内容 SHA256 指纹做去重，OrderedDict 实现 LRU 淘汰。
    参考 TencentDB state-manager.ts 的 processedToolCallIds 模式。
    """

    def __init__(self, maxsize: int = 1000):
        """
        Args:
            maxsize: 缓存最大条目数
        """
        self.cache: OrderedDict[str, Any] = OrderedDict()  # fingerprint → content
        self.maxsize = maxsize
        self.processed_ids: set[str] = set()

    def is_duplicate(self, content: str) -> bool:
        """检查内容是否重复，同时更新缓存。

        Args:
            content: 要检查的内容

        Returns:
            True 如果已存在（重复），False 如果新内容
        """
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        if fingerprint in self.cache or fingerprint in self.processed_ids:
            return True
        self.cache[fingerprint] = content
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return False

    def add(self, content: str, content_id: str = "") -> None:
        """主动添加内容到去重缓存。

        Args:
            content: 内容文本
            content_id: 可选 ID，写入 processed_ids
        """
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        self.cache[fingerprint] = content
        if content_id:
            self.processed_ids.add(content_id)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """清空去重缓存。"""
        self.cache.clear()
        self.processed_ids.clear()

    def get_fingerprint(self, content: str) -> str:
        """计算内容指纹。

        Args:
            content: 内容文本

        Returns:
            SHA256 十六进制指纹
        """
        return hashlib.sha256(content.encode()).hexdigest()

    async def load_from_db(self, pool: Any) -> None:
        """从 SQLite 加载已有的 content_hash 指纹，防止重启后重复写入。

        进程重启后内存 OrderedDict 为空，已写入 SQLite 的消息会被重新 ingest。
        本方法从 memories 表加载所有 content_hash 填入缓存，实现重启后去重持久化。

        Args:
            pool: SQLitePool 实例
        """
        conn = await pool.acquire()
        try:
            cursor = await conn.execute("SELECT content_hash FROM memories WHERE content_hash IS NOT NULL AND content_hash != ''")
            rows = await cursor.fetchall()
            loaded = 0
            for row in rows:
                fp = row["content_hash"]
                if fp and fp not in self.cache:
                    self.cache[fp] = ""
                    loaded += 1
            # 如果加载量超过 maxsize，保留最近的条目
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
            if loaded:
                logger.info("L0 去重缓存已从 SQLite 加载 %d 条指纹", loaded)
        except Exception as e:
            logger.warning("从 SQLite 加载去重缓存失败（非致命）: %s", e)
        finally:
            await pool.release(conn)