"""pipeline/entity_registry.py — 实体注册表（白名单+自动发现机制）。

参考 MemPalace 的 whitelist 设计，用于：
1. 白名单确认：confirmed 实体写入 KG，rejected 实体不再检测
2. 自动发现：高频出现（≥3次）的实体自动升级为 candidate
3. 同步/异步双接口：tools.py 用 async，l1_extractor 的 _filter 用 get_state_sync

使用 SQLite 持久化，数据存储在 unified_memory 的 main DB 中。
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 自动升级阈值（出现次数 ≥ 此值 → candidate）
AUTO_CONFIRM_THRESHOLD = 3


class EntityRegistry:
    """实体注册表，管理实体的 confirmed/candidate/rejected 状态。

    Attributes:
        _pool: SQLitePool 实例
        _cache: 内存缓存 {name: {state, entity_type, count, ...}}
        _initialized: 是否已加载
    """

    def __init__(self, pool):
        self._pool = pool
        self._cache: dict[str, dict] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """从数据库加载注册表到内存缓存。"""
        if self._initialized:
            return
        conn = await self._pool.acquire()
        try:
            # 建表（幂等）
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_registry (
                    name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL DEFAULT 'org',
                    state TEXT NOT NULL DEFAULT 'candidate',
                    count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await conn.commit()

            cursor = await conn.execute(
                "SELECT name, entity_type, state, count FROM entity_registry ORDER BY name"
            )
            rows = await cursor.fetchall()
            for row in rows:
                self._cache[row["name"]] = {
                    "entity_type": row["entity_type"],
                    "state": row["state"],
                    "count": row["count"],
                }
            self._initialized = True
            logger.info("实体注册表加载完成: %d 条", len(self._cache))
        finally:
            await self._pool.release(conn)

    def get_state_sync(self, name: str) -> str:
        """同步获取实体状态（供 l1_extractor._filter 在非 async 上下文调用）。

        Returns:
            "confirmed" / "candidate" / "rejected" / "unregistered"
        """
        entry = self._cache.get(name)
        if entry is None:
            return "unregistered"
        return entry["state"]

    async def register(self, name: str, entity_type: str = "org",
                       status: str = "candidate") -> dict:
        """注册实体到注册表，或更新已有实体的出现次数。

        Args:
            name: 实体名
            entity_type: 实体类型 (person/org/location/product)
            status: 状态 (confirmed/candidate/rejected)

        Returns:
            {"state": str, "count": int}
        """
        entry = self._cache.get(name)
        now = time.time()

        if entry:
            # 已有记录，更新计数值
            count = entry["count"] + 1
            state = entry["state"]

            # 自动升级：高频出现且非 rejected → confirmed
            if count >= AUTO_CONFIRM_THRESHOLD and state == "candidate":
                state = "confirmed"
                logger.info("实体自动升级为 confirmed: %s (count=%d)", name, count)

            self._cache[name] = {
                "entity_type": entity_type,
                "state": state,
                "count": count,
            }
            await self._persist(name, entity_type, state, count)
            return {"state": state, "count": count}
        else:
            # 新实体
            self._cache[name] = {
                "entity_type": entity_type,
                "state": status,
                "count": 1,
            }
            await self._persist(name, entity_type, status, 1)
            logger.info("实体注册: %s (type=%s, state=%s)", name, entity_type, status)
            return {"state": status, "count": 1}

    async def confirm(self, name: str) -> bool:
        """确认实体为可信（state → confirmed）。

        Returns:
            True 如果实体存在且状态变更成功
        """
        entry = self._cache.get(name)
        if entry is None:
            return False
        entry["state"] = "confirmed"
        await self._persist(name, entry["entity_type"], "confirmed", entry["count"])
        logger.info("实体确认: %s", name)
        return True

    async def reject(self, name: str) -> bool:
        """拒绝实体（不再检测）。

        Returns:
            True 如果实体存在且状态变更成功
        """
        entry = self._cache.get(name)
        if entry is None:
            return False
        entry["state"] = "rejected"
        await self._persist(name, entry["entity_type"], "rejected", entry["count"])
        logger.info("实体拒绝: %s", name)
        return True

    async def list_candidates(self) -> list[dict]:
        """列出所有待确认的候选实体。"""
        return [
            {"name": name, "entity_type": data["entity_type"], "count": data["count"]}
            for name, data in self._cache.items()
            if data["state"] == "candidate"
        ]

    async def list_confirmed(self) -> list[dict]:
        """列出所有已确认的实体。"""
        return [
            {"name": name, "entity_type": data["entity_type"], "count": data["count"]}
            for name, data in self._cache.items()
            if data["state"] == "confirmed"
        ]

    def get_stats(self) -> dict:
        """获取注册表统计。"""
        confirmed = sum(1 for d in self._cache.values() if d["state"] == "confirmed")
        candidates = sum(1 for d in self._cache.values() if d["state"] == "candidate")
        rejected = sum(1 for d in self._cache.values() if d["state"] == "rejected")
        return {
            "total": len(self._cache),
            "confirmed": confirmed,
            "candidates": candidates,
            "rejected": rejected,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _persist(self, name: str, entity_type: str, state: str, count: int) -> None:
        """持久化实体记录到数据库。"""
        now = time.time()
        conn = await self._pool.acquire()
        try:
            await conn.execute("""
                INSERT INTO entity_registry (name, entity_type, state, count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    entity_type=excluded.entity_type,
                    state=excluded.state,
                    count=excluded.count,
                    updated_at=excluded.updated_at
            """, (name, entity_type, state, count, now, now))
            await conn.commit()
        except Exception as e:
            logger.error("持久化实体注册失败: %s", e)
        finally:
            await self._pool.release(conn)