"""pipeline/entity_registry.py — 实体注册表（白名单）。

参考 MemPalace entity_registry.py + entity_detector.py 设计：
- 实体分两种状态：confirmed（可信，写入 KG）/ candidate（待确认，不写入 KG）
- 自动从旧数据中挖掘候选实体，人工确认后升级
- L1 提取时只将 confirmed 实体写入 KG，candidate 只保留在 SQLite 文本中

v1.0: 2026-08-21 — 新增实体注册表模块
"""

import json
import logging
import time
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# 实体状态
ENTITY_CONFIRMED = "confirmed"   # 可信实体，写入 KG
ENTITY_CANDIDATE = "candidate"   # 待确认，不写入 KG
ENTITY_REJECTED = "rejected"     # 已拒绝的噪声，不再检测

# 默认实体类型
ENTITY_TYPES = frozenset({"person", "org", "location", "product"})


class EntityRegistry:
    """实体注册表 — 白名单模式。

    L1 提取的实体只有在此注册表中为 confirmed 状态时，才写入 KG。
    candidate 实体只保留在 SQLite 文本中，不污染 KG 关系图。

    用法：
        registry = EntityRegistry(pool)
        await registry.ensure_confirmed("王哥", "person")
        ok = await registry.should_write_to_kg("王哥")
    """

    def __init__(self, pool):
        """
        Args:
            pool: SQLitePool 实例
        """
        self._pool = pool
        self._cache: dict[str, str] = {}  # name -> status
        self._cache_time = 0.0
        self._cache_ttl = 300.0  # 5 分钟缓存

    async def _ensure_table(self) -> None:
        """确保实体注册表存在。"""
        conn = await self._pool.acquire()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'org',
                    status TEXT NOT NULL DEFAULT 'candidate',
                    proof_count INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            # 索引：名称快速查找 + 状态过滤
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_registry_name ON entity_registry(name)")
            except Exception:
                pass
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_registry_status ON entity_registry(status)")
            except Exception:
                pass
            await conn.commit()
        finally:
            await self._pool.release(conn)

    async def _refresh_cache(self) -> None:
        """从 DB 刷新全量缓存。"""
        now = time.time()
        if now - self._cache_time < self._cache_ttl and self._cache:
            return
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute("SELECT name, status FROM entity_registry")
            rows = await cursor.fetchall()
            self._cache = {row["name"]: row["status"] for row in rows}
            self._cache_time = now
            logger.debug("实体注册表缓存刷新: %d 条", len(self._cache))
        finally:
            await self._pool.release(conn)

    async def get_status(self, name: str) -> Optional[str]:
        """获取实体状态。

        Returns:
            "confirmed" | "candidate" | "rejected" | None（未注册）
        """
        await self._refresh_cache()
        return self._cache.get(name)

    async def get_state(self, name: str) -> Optional[str]:
        """get_state 是 get_status 的别名（供 l1_extractor 调用）。"""
        return await self.get_status(name)

    async def initialize(self) -> None:
        """initialize — 确保表和初始数据就绪。"""
        await self._ensure_table()

    async def register(self, name: str, entity_type: str,
                       metadata: Optional[dict] = None,
                       status: str = ENTITY_CANDIDATE) -> str:
        """注册一个实体到注册表。

        Args:
            name: 实体名称
            entity_type: 实体类型（person/org/location/product）
            metadata: 附加元数据
            status: 初始状态（默认 candidate）

        Returns:
            "created" | "updated" | "skipped"
        """
        if status not in (ENTITY_CONFIRMED, ENTITY_CANDIDATE, ENTITY_REJECTED):
            status = ENTITY_CANDIDATE

        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = time.time()

        # 检查是否已存在
        existing = await self.get_status(name)
        if existing is not None:
            if existing == status:
                return "skipped"
            # 更新状态
            conn = await self._pool.acquire()
            try:
                await conn.execute(
                    "UPDATE entity_registry SET status=?, updated_at=? WHERE name=?",
                    (status, now, name),
                )
                await conn.commit()
            finally:
                await self._pool.release(conn)
            self._cache[name] = status
            return "updated"

        # 新建
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                "INSERT INTO entity_registry (id, name, entity_type, status, proof_count, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
                (str(uuid4()), name, entity_type, status, meta_json, now, now),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)
        self._cache[name] = status
        return "created"

    async def confirm(self, name: str) -> bool:
        """将候选实体升级为 confirmed 状态。

        Returns:
            True 确认成功，False 实体不存在
        """
        existing = await self.get_status(name)
        if existing is None:
            return False
        if existing == ENTITY_CONFIRMED:
            return True
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                "UPDATE entity_registry SET status=?, updated_at=? WHERE name=?",
                (ENTITY_CONFIRMED, time.time(), name),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)
        self._cache[name] = ENTITY_CONFIRMED
        return True

    async def reject(self, name: str) -> bool:
        """将实体标记为 rejected（噪声不再检测）。

        Returns:
            True 成功，False 实体不存在
        """
        existing = await self.get_status(name)
        if existing is None:
            return False
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                "UPDATE entity_registry SET status=?, updated_at=? WHERE name=?",
                (ENTITY_REJECTED, time.time(), name),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)
        self._cache[name] = ENTITY_REJECTED
        return True

    async def should_write_to_kg(self, name: str) -> bool:
        """判断实体是否应写入 KG。

        Returns:
            True 当且仅当实体为 confirmed 状态
        """
        status = await self.get_status(name)
        return status == ENTITY_CONFIRMED

    async def list_confirmed(self) -> list[dict]:
        """列出所有已确认实体。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT name, entity_type, proof_count, created_at FROM entity_registry WHERE status=? ORDER BY name",
                (ENTITY_CONFIRMED,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await self._pool.release(conn)

    async def list_candidates(self) -> list[dict]:
        """列出所有候选实体（待人工确认）。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT name, entity_type, proof_count, created_at FROM entity_registry WHERE status=? ORDER BY proof_count DESC, name",
                (ENTITY_CANDIDATE,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await self._pool.release(conn)

    async def increment_proof(self, name: str) -> None:
        """增加实体的证明次数（用于候选实体排序）。"""
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                "UPDATE entity_registry SET proof_count = proof_count + 1, updated_at=? WHERE name=?",
                (time.time(), name),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)

    async def auto_discover(self, entities: list[dict]) -> dict:
        """自动发现并注册实体列表（批量）。

        L1 提取后调用此方法，将提取的实体注册为 candidate，
        高频出现的实体自动升级为 confirmed。

        Args:
            entities: [{"name": "...", "type": "..."}, ...]

        Returns:
            {"created": N, "confirmed": N, "skipped": N}
        """
        created = 0
        confirmed = 0
        skipped = 0
        conn = await self._pool.acquire()
        try:
            now = time.time()
            for ent in entities:
                name = ent.get("name", "").strip()
                ent_type = ent.get("type", "org")
                if not name or len(name) < 2:
                    continue

                # 检查是否已注册
                cursor = await conn.execute(
                    "SELECT status, proof_count FROM entity_registry WHERE name=?",
                    (name,),
                )
                row = await cursor.fetchone()

                if row:
                    # 已存在：增加证明次数
                    new_count = row["proof_count"] + 1
                    await conn.execute(
                        "UPDATE entity_registry SET proof_count=?, updated_at=? WHERE name=?",
                        (new_count, now, name),
                    )
                    # 高频出现（>=3 次）自动升级 confirmed
                    if row["status"] == ENTITY_CANDIDATE and new_count >= 3:
                        await conn.execute(
                            "UPDATE entity_registry SET status=?, updated_at=? WHERE name=?",
                            (ENTITY_CONFIRMED, now, name),
                        )
                        confirmed += 1
                    else:
                        skipped += 1
                else:
                    # 新增候选实体
                    await conn.execute(
                        "INSERT INTO entity_registry (id, name, entity_type, status, proof_count, metadata, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 1, '{}', ?, ?)",
                        (str(uuid4()), name, ent_type, ENTITY_CANDIDATE, now, now),
                    )
                    created += 1

                # 更新内存缓存
                self._cache[name] = ENTITY_CONFIRMED if confirmed > 0 else ENTITY_CANDIDATE

            await conn.commit()
        finally:
            await self._pool.release(conn)

        # 刷新缓存
        self._cache_time = 0  # 强制下次读取时刷新
        return {"created": created, "confirmed": confirmed, "skipped": skipped}

    async def batch_register(self, names: list[str], entity_type: str = "org",
                             status: str = ENTITY_CONFIRMED) -> int:
        """批量注册实体（用于初始化时从旧数据回填）。

        Args:
            names: 实体名称列表
            entity_type: 实体类型
            status: 状态（默认 confirmed）

        Returns:
            新增数量
        """
        count = 0
        for name in names:
            result = await self.register(name, entity_type, status=status)
            if result == "created":
                count += 1
        return count

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """搜索实体注册表（模糊匹配名称）。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT name, entity_type, status, proof_count FROM entity_registry WHERE name LIKE ? ORDER BY proof_count DESC LIMIT ?",
                (f"%{query}%", limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await self._pool.release(conn)