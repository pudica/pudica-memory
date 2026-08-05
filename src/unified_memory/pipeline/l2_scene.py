"""pipeline/l2_scene.py — L2：场景组织 + 知识图谱更新。

参考 TencentDB scene-extractor.ts 和 hindsight reflect/agent.py。
文档 6.1 节 L2SceneOrganizer 实现。
"""

import logging
import time
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class L2SceneOrganizer:
    """L2：场景组织 + 知识图谱更新。

    将提取结果组织到场景中，更新知识图谱的实体和关系。
    参考 TencentDB scene-extractor.ts 的场景 CRUD 操作。
    """

    def __init__(self, kg: Any, pool: Any):
        """
        Args:
            kg: KnowledgeGraph 实例
            pool: SQLitePool 实例
        """
        self._kg = kg
        self._pool = pool

    async def organize(self, extracted: dict) -> Optional[str]:
        """将提取结果组织到场景中，更新知识图谱。

        Args:
            extracted: L1 提取结果字典，包含 entities, relations, summary 等

        Returns:
            场景 ID，如果无法创建则返回 None
        """
        if not extracted or not extracted.get("summary"):
            return None

        # 1. 更新知识图谱实体和关系
        for entity in extracted.get("entities", []):
            await self._kg.add_entity(entity["name"], entity["type"])

        for rel in extracted.get("relations", []):
            await self._kg.add_relation(
                rel["subject"], rel["predicate"], rel["object"]
            )

        # 2. 场景归类（返回场景 ID 和是否新建的标志）
        scene_id, just_created = await self._find_or_create_scene(extracted)

        # 3. 更新场景摘要（仅当场景已存在时追加，新建时不重复）
        if not just_created:
            await self._update_scene_summary(scene_id, extracted["summary"])

        # 4. 更新时间线
        if extracted.get("time_range"):
            await self._update_timeline(scene_id, extracted["time_range"])

        return scene_id

    async def _find_or_create_scene(self, extracted: dict) -> tuple[str, bool]:
        """查找已有场景或创建新场景。

        使用单个连接 + INSERT OR IGNORE 避免并发创建重复场景。

        Args:
            extracted: 提取结果

        Returns:
            (场景 ID, 是否新建)
        """
        # 默认使用第一个实体的名称作为场景名
        scene_name = "general"
        if extracted.get("entities"):
            scene_name = extracted["entities"][0]["name"]

        scene_id = str(uuid4())
        now = time.time()
        summary = extracted.get("summary", "")

        conn = await self._pool.acquire()
        try:
            # 先尝试查找
            cursor = await conn.execute(
                "SELECT id FROM scenes WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (scene_name,),
            )
            row = await cursor.fetchone()
            if row:
                return row["id"], False  # 已存在，非新建

            # 不存在则创建（INSERT OR IGNORE 防并发重复）
            await conn.execute(
                """INSERT OR IGNORE INTO scenes (id, name, summary, wing, room, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (scene_id, scene_name, summary,
                 "default", "general", now, now),
            )
            await conn.commit()

            # INSERT 可能被并发忽略，重新查询获取实际 ID（幂等）
            cursor = await conn.execute(
                "SELECT id FROM scenes WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (scene_name,),
            )
            row = await cursor.fetchone()
            if row:
                if row["id"] != scene_id:
                    logger.debug("场景 '%s' 已被并发创建，使用已有 ID", scene_name)
                    return row["id"], False  # 被并发创建，非新建
                return row["id"], True  # 确实新建
            return scene_id, True  # 查不到说明就是新建的
        finally:
            await self._pool.release(conn)

    async def _update_scene_summary(self, scene_id: str, summary: str) -> None:
        """更新场景摘要（带截断保护，防止无限增长）。

        Args:
            scene_id: 场景 ID
            summary: 新的摘要文本
        """
        MAX_SUMMARY_LEN = 5000
        conn = await self._pool.acquire()
        try:
            await conn.execute(
                """UPDATE scenes
                   SET summary = substr(summary || '\n' || ?, 1, ?),
                       updated_at = ?
                   WHERE id = ?""",
                (summary, MAX_SUMMARY_LEN, time.time(), scene_id),
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)

    async def _update_timeline(
        self, scene_id: str, time_range: dict
    ) -> None:
        """更新场景的时间线。

        Args:
            scene_id: 场景 ID
            time_range: {"start": timestamp, "end": timestamp}
        """
        if not time_range:
            return
        # 统一转时间戳（支持 ISO8601 字符串和 float）
        from datetime import datetime
        start_val = time_range.get("start")
        end_val = time_range.get("end")
        if isinstance(start_val, str):
            try:
                start_val = datetime.fromisoformat(start_val).timestamp()
            except (ValueError, TypeError):
                start_val = None
        if isinstance(end_val, str):
            try:
                end_val = datetime.fromisoformat(end_val).timestamp()
            except (ValueError, TypeError):
                end_val = None

        # 仅当值有效时才更新，避免 0/None 被 MIN/MAX 选中导致时间线设为 epoch
        conn = await self._pool.acquire()
        try:
            set_clauses: list[str] = []
            params: list[Any] = []
            if start_val is not None:
                set_clauses.append("time_start = COALESCE(MIN(?, time_start), ?)")
                params.extend([start_val, start_val])
            if end_val is not None:
                set_clauses.append("time_end = COALESCE(MAX(?, time_end), ?)")
                params.extend([end_val, end_val])
            set_clauses.append("updated_at = ?")
            params.append(time.time())
            params.append(scene_id)

            set_sql = ", ".join(set_clauses)
            await conn.execute(
                f"""UPDATE scenes SET {set_sql} WHERE id = ?""",
                params,
            )
            await conn.commit()
        finally:
            await self._pool.release(conn)