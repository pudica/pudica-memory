"""tasks/scheduler.py — 任务调度器。

每天凌晨 2 点 reflect，每 4 小时 consolidation。
参考 hindsight reflect/agent.py 的 schedule 模式。
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
import time as time_mod
import zoneinfo
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskScheduler:
    """任务调度器。

    支持：
    - 每天凌晨 2 点执行 reflect 任务
    - 每 4 小时执行 consolidation 任务
    - 手动触发
    - 调度统计
    """

    def __init__(
        self,
        reflector: Any,
        consolidator: Any,
        reflect_interval_hours: int = 24,
        consolidate_interval_hours: int = 4,
        compressor: Any = None,
        compress_interval_hours: int = 24,
    ):
        """
        Args:
            reflector: Reflector 实例
            consolidator: Consolidator 实例
            reflect_interval_hours: Reflect 间隔（默认 24 小时）
            consolidate_interval_hours: Consolidation 间隔（默认 4 小时）
            compressor: Compressor 实例（可选，MemPalace 压缩）
            compress_interval_hours: 压缩任务间隔（默认 24 小时）
        """
        self._reflector = reflector
        self._consolidator = consolidator
        self._reflect_interval = reflect_interval_hours
        self._consolidate_interval = consolidate_interval_hours
        self._compressor = compressor
        self._compress_interval = compress_interval_hours

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # 统计
        self.reflect_count = 0
        self.consolidate_count = 0
        self.compress_count = 0
        self.last_reflect_time: Optional[float] = None
        self.last_consolidate_time: Optional[float] = None
        self.last_compress_time: Optional[float] = None

    async def start(self) -> None:
        """启动调度器后台循环。"""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("调度器已启动 (reflect=%dh, consolidate=%dh)",
                     self._reflect_interval, self._consolidate_interval)

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("调度器已停止")

    async def _run_loop(self) -> None:
        """调度主循环。"""
        # 计算下次执行时间（统一使用 Asia/Shanghai 本地时区，避免与循环内 now 时区混用）
        shanghai = zoneinfo.ZoneInfo("Asia/Shanghai")
        next_reflect = self._next_reflect_time()
        next_consolidate = datetime.now(shanghai) + timedelta(hours=self._consolidate_interval)
        next_compress = datetime.now(shanghai) + timedelta(hours=self._compress_interval) if self._compressor else None

        while self._running:
            now = datetime.now(shanghai)
            should_reflect = now >= next_reflect
            should_consolidate = now >= next_consolidate
            should_compress = self._compressor is not None and next_compress is not None and now >= next_compress

            if should_reflect:
                try:
                    await self._reflector.reflect()
                    self.reflect_count += 1
                    self.last_reflect_time = now.timestamp()
                    logger.info("Reflect 任务完成 (累计: %d)", self.reflect_count)
                except Exception as e:
                    logger.error("Reflect 任务失败: %s", e)
                next_reflect = self._next_reflect_time()

            if should_consolidate:
                try:
                    await self._consolidator.consolidate()
                    self.consolidate_count += 1
                    self.last_consolidate_time = now.timestamp()
                    logger.info("Consolidation 任务完成 (累计: %d)", self.consolidate_count)
                except Exception as e:
                    logger.error("Consolidation 任务失败: %s", e)
                next_consolidate = now + timedelta(hours=self._consolidate_interval)

            if should_compress:
                try:
                    await self._compressor.compress()
                    self.compress_count += 1
                    self.last_compress_time = now.timestamp()
                    logger.info("Compression 任务完成 (累计: %d)", self.compress_count)
                except Exception as e:
                    logger.error("Compression 任务失败: %s", e)
                next_compress = now + timedelta(hours=self._compress_interval)

            # 休眠 60 秒后再次检查
            await asyncio.sleep(60)

    def _next_reflect_time(self) -> datetime:
        """计算下次 Reflect 执行时间（每天 Asia/Shanghai 本地时区凌晨 2:00）。

        Returns:
            下次执行时间
        """
        now = datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai"))
        target = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return target

    async def trigger_reflect(self) -> dict:
        """手动触发 Reflect 任务。

        Returns:
            Reflect 结果
        """
        logger.info("手动触发 Reflect")
        result = await self._reflector.reflect()
        self.reflect_count += 1
        self.last_reflect_time = time_mod.time()
        return result

    async def trigger_consolidate(self) -> dict:
        """手动触发 Consolidation 任务。

        Returns:
            Consolidation 结果
        """
        logger.info("手动触发 Consolidation")
        result = await self._consolidator.consolidate()
        self.consolidate_count += 1
        self.last_consolidate_time = time_mod.time()
        return result

    def get_status(self) -> dict:
        """获取调度器状态。

        Returns:
            {"running": bool, "reflect_count": N, "consolidate_count": N,
             "last_reflect": float or None, "last_consolidate": float or None}
        """
        return {
            "running": self._running,
            "reflect_count": self.reflect_count,
            "consolidate_count": self.consolidate_count,
            "compress_count": self.compress_count,
            "last_reflect": self.last_reflect_time,
            "last_consolidate": self.last_consolidate_time,
            "last_compress": self.last_compress_time,
        }