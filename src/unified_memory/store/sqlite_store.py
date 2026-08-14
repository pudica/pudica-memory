"""store/sqlite_store.py — SQLite 连接池 + WAL 模式 + 写入缓冲队列。

参考文档 7.1 节的 SQLitePool + WriteBuffer 实现：
- WAL 模式支持读写并发
- 连接池复用，避免重复创建连接
- 异步写入缓冲队列，批量刷盘
- 内存双写缓冲，进程崩溃至多丢 5s 数据
"""

import json
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
import aiosqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 连接池
# ---------------------------------------------------------------------------

class SQLitePool:
    """轻量级 aiosqlite 连接池，支持 WAL 模式和连接复用。

    参考：文档 7.1 节 SQLitePool 实现。
    """

    def __init__(self, db_path: str, maxsize: int = 5, timeout: float = 30.0):
        """
        Args:
            db_path: 数据库文件路径
            maxsize: 连接池最大连接数
            timeout: 连接超时（秒）
        """
        self._db_path = db_path
        self._maxsize = maxsize
        self._timeout = timeout
        self._pool: list[aiosqlite.Connection] = []
        self._in_use: set[aiosqlite.Connection] = set()
        self._lock = asyncio.Lock()
        # 信号量控制并发连接数：池耗尽时 acquire 会阻塞等待，而非直接抛错
        self._sem = asyncio.Semaphore(maxsize)
        self._initialized = False

    async def _make_conn(self) -> aiosqlite.Connection:
        """创建一条新连接，配置 WAL 模式等优化参数。"""
        conn = await aiosqlite.connect(
            self._db_path,
            timeout=self._timeout,
            check_same_thread=False,
        )
        # WAL 模式 + 同步模式降级，允许读写并发
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        # 外键约束
        await conn.execute("PRAGMA foreign_keys=ON")
        # 缓存 8MB
        await conn.execute("PRAGMA cache_size=-8000")
        # 临时表放内存
        await conn.execute("PRAGMA temp_store=MEMORY")
        # 减少 mmap 大小（适配 Windows）
        await conn.execute("PRAGMA mmap_size=0")
        conn.row_factory = aiosqlite.Row
        return conn

    async def initialize(self) -> None:
        """初始化数据库，创建所有表。"""
        if self._initialized:
            return
        conn = await self._make_conn()
        try:
            await self._create_tables(conn)
            self._pool.append(conn)
            self._initialized = True
            logger.info("SQLite 数据库已初始化: %s", self._db_path)
        except Exception:
            await conn.close()
            raise

    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        """创建所有需要的表。"""
        # 记忆主表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT,
                wing TEXT DEFAULT 'default',
                room TEXT DEFAULT 'general',
                source TEXT DEFAULT '',
                fact_type TEXT DEFAULT 'observation',
                proof_count INTEGER DEFAULT 1,
                source_memory_ids TEXT DEFAULT '[]',
                authority TEXT DEFAULT 'medium',
                trust_score REAL DEFAULT 0.5,
                summary TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        # 记忆 FTS5 全文搜索表
        # 迁移：旧版表是 fts5(content, content_rowid=id) 单列结构或非 trigram 分词器，
        # 与代码假设不符（BM25 中文检索依赖 trigram）。检测到旧结构先重建。
        try:
            cur = await conn.execute("PRAGMA table_info(memories_fts)")
            fts_cols = [r[1] for r in await cur.fetchall()]
            cur2 = await conn.execute("SELECT sql FROM sqlite_master WHERE name='memories_fts'")
            row2 = await cur2.fetchone()
            fts_sql = row2["sql"] if row2 else ""
            if (fts_cols and fts_cols != ["id", "content", "wing", "room"]) or (
                fts_cols and "trigram" not in fts_sql
            ):
                logger.warning("memories_fts 结构/分词器不符，重建为 trigram 五列结构")
                await conn.execute("DROP TABLE IF EXISTS memories_fts")
        except Exception:
            pass  # 表不存在时 PRAGMA 返回空，正常
        # Bug fix: trigram tokenizer may be unavailable on some Python/SQLite builds
        # (e.g. default Windows Python). Fall back to unicode61 if trigram fails.
        try:
            await conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id UNINDEXED,
                    content,
                    wing,
                    room,
                    content='memories',
                    content_rowid='rowid',
                    tokenize='trigram'
                )
            """)
        except Exception:
            logger.warning(
                "FTS5 trigram tokenizer 不可用（可能是 SQLite 未编译 trigram），"
                "回退到 unicode61 分词器。中文检索效果会下降。"
            )
            await conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id UNINDEXED,
                    content,
                    wing,
                    room,
                    content='memories',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)
        # Bug fix (2026-08-13): 旧库（v3.2.1 以前）的 memories 表可能没有
        # authority / trust_score / summary 列。CREATE TABLE IF NOT EXISTS 不会
        # 修改已有表，导致 engine.py 的 INSERT 报 "no such column: authority"。
        # 这里检测缺失列并用 ALTER TABLE ADD COLUMN 逐个补齐。
        cur = await conn.execute("PRAGMA table_info(memories)")
        existing_cols = {r[1] for r in await cur.fetchall()}
        for col_sql in (
            "authority TEXT DEFAULT 'medium'",
            "trust_score REAL DEFAULT 0.5",
            "summary TEXT DEFAULT ''",
        ):
            col_name = col_sql.split()[0]
            if col_name not in existing_cols:
                logger.info("迁移：为 memories 表补充缺失列 %s", col_name)
                await conn.execute(f"ALTER TABLE memories ADD COLUMN {col_sql}")
        # 迁移后刷新已出现但为空的 authority/trust_score（旧数据默认 medium/0.5）
        # 实体表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        # 关系表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT DEFAULT 'auto',
                created_at REAL NOT NULL,
                FOREIGN KEY (subject) REFERENCES entities(name),
                FOREIGN KEY (object) REFERENCES entities(name)
            )
        """)
        # 场景表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                summary TEXT DEFAULT '',
                wing TEXT DEFAULT 'default',
                room TEXT DEFAULT 'general',
                time_start REAL,
                time_end REAL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        # 配置表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # Verbatim 逐字存储表（MemPalace: 存储原始输入）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS verbatim (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                raw_content TEXT NOT NULL,
                source TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        # 心智模型表（Hindsight: 用户偏好/信念/行为模式）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mental_models (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                source_memory_ids TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(category, key)
            )
        """)
        # 压缩记忆表（MemPalace AAAK: 压缩后的记忆摘要）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS compressed_memories (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                source_memory_ids TEXT NOT NULL,
                source_count INTEGER NOT NULL,
                wing TEXT DEFAULT 'default',
                room TEXT DEFAULT 'general',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        # 索引
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_wing ON memories(wing)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_room ON memories(room)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_verbatim_memory ON verbatim(memory_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_verbatim_created ON verbatim(created_at)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_category ON mental_models(category)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_compressed_created ON compressed_memories(created_at)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_compressed_wing ON compressed_memories(wing)")
        # 场景名唯一约束（防止并发 create 时重复插入）
        await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_scenes_name ON scenes(name)")
        # FTS5 外部内容表同步触发器（必须存在，否则 memories_fts 永远为空，BM25 检索失效）
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, id, content, wing, room)
                VALUES (new.rowid, new.id, new.content, new.wing, new.room);
            END
        """)
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content, wing, room)
                VALUES ('delete', old.rowid, old.id, old.content, old.wing, old.room);
            END
        """)
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content, wing, room)
                VALUES ('delete', old.rowid, old.id, old.content, old.wing, old.room);
                INSERT INTO memories_fts(rowid, id, content, wing, room)
                VALUES (new.rowid, new.id, new.content, new.wing, new.room);
            END
        """)
        # Bug fix: 仅在 FTS 表为空时才重建索引，避免每次 init 都全量 rebuild。
        # 外部内容表在初始化后由触发器自动同步，无需重复 rebuild。
        # Bug fix: 用 LIMIT 1 替代 COUNT(*) 做存在性检查，避免大表全表扫描。
        cursor = await conn.execute("SELECT 1 FROM memories_fts LIMIT 1")
        fts_empty = (await cursor.fetchone()) is None
        if fts_empty:
            await conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        await conn.commit()

    async def acquire(self) -> aiosqlite.Connection:
        """从连接池获取一条连接。

        池满时阻塞等待（通过信号量），直到有连接被释放，而非直接抛错。
        """
        await self._sem.acquire()
        try:
            async with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                else:
                    conn = await self._make_conn()
                self._in_use.add(conn)
            return conn
        except Exception:
            # 建连失败或取连接异常时，归还信号量额度，避免永久占用
            self._sem.release()
            raise

    async def release(self, conn: aiosqlite.Connection) -> None:
        """释放连接回连接池。"""
        async with self._lock:
            self._in_use.discard(conn)
            try:
                await conn.rollback()
                self._pool.append(conn)
            except Exception:
                await conn.close()
        # 无论释放成功与否，都归还信号量额度
        self._sem.release()

    async def close(self) -> None:
        """关闭所有连接。"""
        async with self._lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
            self._in_use.clear()
            self._initialized = False


# ---------------------------------------------------------------------------
# 写入缓冲队列
# ---------------------------------------------------------------------------

@dataclass
class WriteOp:
    """一条写入操作。

    Attributes:
        table: 目标表名
        data: 写入数据（字典）
        timestamp: 操作时间戳
        future: 异步 Future，写入完成后通知调用方
    """
    table: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)


class WriteBuffer:
    """写入缓冲队列。

    .. deprecated::
        WriteBuffer 当前未被 UnifiedMemoryApp 使用（PipelineEngine 实现了自己的
        缓冲+写入逻辑）。保留此类作为未来统一写入路径的基础设施，但请注意
        其内存双写保护机制当前未生效。如需启用，请在 main.py 中实例化并
        通过 WriteBuffer 路由所有写入。

    积累写入请求，每 5 秒或积压 100 条时批量 flush。
    参考 TencentDB 的 SerialQueue 模式（文档 7.1 节）。
    """

    def __init__(self, pool: SQLitePool, flush_interval: float = 5.0, batch_size: int = 100):
        """
        Args:
            pool: SQLite 连接池
            flush_interval: 自动刷盘间隔（秒）
            batch_size: 触发刷盘的队列长度
        """
        self._pool = pool
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._queue: asyncio.Queue[WriteOp] = asyncio.Queue()
        self._mem_buffer: list[WriteOp] = []  # 内存双写缓冲
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """启动后台 flush 协程。"""
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.debug("WriteBuffer 后台 flush 协程已启动")

    async def stop(self, flush_remaining: bool = True) -> None:
        """停止后台 flush 协程。

        Args:
            flush_remaining: 停止前是否刷盘剩余数据
        """
        self._running = False
        if flush_remaining:
            await self._flush_now()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("WriteBuffer 已停止")

    async def write(self, table: str, data: dict) -> asyncio.Future:
        """提交一条写入（缓冲后异步刷盘）。

        Args:
            table: 目标表名
            data: 写入数据字典

        Returns:
            写入完成后的 Future
        """
        op = WriteOp(table=table, data=data)
        await self._queue.put(op)
        self._mem_buffer.append(op)  # 内存双写保护
        return op.future

    async def flush_now(self) -> int:
        """紧急刷盘，立即将缓冲区所有内容写入 SQLite。

        Returns:
            写入的记录数
        """
        return await self._flush_now()

    async def _flush_loop(self) -> None:
        """后台协程：定时 flush。"""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            try:
                # 无条件刷盘，不论队列是否达到阈值（防止少量数据永远不持久化）
                await self._flush_now()
            except Exception:
                logger.exception("WriteBuffer flush 循环异常")

    async def _flush_now(self) -> int:
        """将当前队列中的写入批量刷盘。

        Returns:
            写入的记录数
        """
        ops: list[WriteOp] = []
        # 尽量从队列中取，但不阻塞
        while not self._queue.empty() and len(ops) < self._batch_size * 2:
            try:
                op = self._queue.get_nowait()
                ops.append(op)
            except asyncio.QueueEmpty:
                break

        if not ops:
            # 检查内存缓冲有没有积累
            if self._mem_buffer:
                ops = self._mem_buffer.copy()
                # 不在此处 clear，等 DB 写入成功后才清除
            else:
                return 0

        conn = await self._pool.acquire()
        try:
            for op in ops:
                cols = ", ".join(op.data.keys())
                placeholders = ", ".join("?" for _ in op.data)
                # 白名单校验表名和列名，防止 SQL 注入
                _VALID_TABLES = {"memories", "entities", "relations", "scenes", "config"}
                if op.table not in _VALID_TABLES:
                    raise ValueError(f"Invalid table name: {op.table}")
                for col_name in op.data.keys():
                    if not col_name.replace("_", "").isalnum():
                        raise ValueError(f"Invalid column name: {col_name}")
                sql = f"INSERT OR REPLACE INTO {op.table} ({cols}) VALUES ({placeholders})"
                await conn.execute(sql, list(op.data.values()))
            await conn.commit()
            # 所有操作成功后才清除内存缓冲（Bug fix: 仅清除本次刷盘的条目，
            # 保留刷盘期间新加入的条目，防止竞态导致内存双写保护失效）
            for op in ops:
                if not op.future.done():
                    op.future.set_result(True)
            logger.debug("Flushed %d writes to SQLite", len(ops))
            # 仅移除已刷盘的 ops，保留刷盘期间被 write() 新增的条目
            flushed_count = len(ops)
            if len(self._mem_buffer) > flushed_count:
                self._mem_buffer[:] = self._mem_buffer[flushed_count:]
            else:
                self._mem_buffer.clear()
            return len(ops)
        except Exception as e:
            await conn.rollback()
            for op in ops:
                if not op.future.done():
                    op.future.set_exception(e)
            raise
        finally:
            await self._pool.release(conn)


# ---------------------------------------------------------------------------
# 便捷查询方法
# ---------------------------------------------------------------------------

class SQLiteStore:
    """SQLite 存储封装，提供高层查询方法。"""

    def __init__(self, pool: SQLitePool, buffer: WriteBuffer):
        self._pool = pool
        self._buffer = buffer

    async def add_memory(self, memory_id: str, content: str, **kwargs: Any) -> None:
        """添加一条记忆记录。

        Args:
            memory_id: 记忆 ID
            content: 记忆内容
            **kwargs: 其他字段（wing, room, source, fact_type 等）
        """
        import hashlib
        now = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        data = {
            "id": memory_id,
            "content": content,
            "content_hash": content_hash,
            "wing": kwargs.get("wing", "default"),
            "room": kwargs.get("room", "general"),
            "source": kwargs.get("source", ""),
            "fact_type": kwargs.get("fact_type", "observation"),
            "proof_count": kwargs.get("proof_count", 1),
            "source_memory_ids": json.dumps(kwargs.get("source_memory_ids", [])),
            "metadata": json.dumps(kwargs.get("metadata", {})),
            "created_at": now,
            "updated_at": now,
        }
        await self._buffer.write("memories", data)

    async def get_memory(self, memory_id: str) -> Optional[dict]:
        """获取单条记忆。"""
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await self._pool.release(conn)

    async def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 全文搜索。

        Args:
            query: 搜索关键词（空格分隔，自动用 OR 连接）
            limit: 返回条数

        Returns:
            搜索结果列表
        """
        fts_query = " OR ".join(query.split())
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                """SELECT m.*, rank FROM memories_fts
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await self._pool.release(conn)

    async def get_recent(self, hours: int = 24, limit: int = 100) -> list[dict]:
        """获取最近 N 小时的记忆。"""
        cutoff = time.time() - hours * 3600
        conn = await self._pool.acquire()
        try:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (cutoff, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await self._pool.release(conn)

    async def get_by_wing_room(
        self, wing: str, room: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """按 wing/room 查询记忆。"""
        conn = await self._pool.acquire()
        try:
            if room:
                cursor = await conn.execute(
                    "SELECT * FROM memories WHERE wing = ? AND room = ? ORDER BY created_at DESC LIMIT ?",
                    (wing, room, limit),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM memories WHERE wing = ? ORDER BY created_at DESC LIMIT ?",
                    (wing, limit),
                )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await self._pool.release(conn)