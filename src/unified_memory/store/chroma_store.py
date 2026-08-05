"""store/chroma_store.py — ChromaDB 优化版存储后端。

参考文档 7.2 节实现：
- HNSW 自适应参数（按集合大小）
- 批量写入，一次 embedding 调用
- 嵌入器全局缓存
- 元数据索引（wing/room 两级过滤）
- 写入去重缓存
"""

import hashlib
import logging
import threading
import time
from typing import Any, Optional

import os
import chromadb
import concurrent.futures
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# 嵌入器全局缓存（复用 mempalace 模式）
_EF_CACHE: dict[str, Any] = {}
_EF_CACHE_LOCK = threading.Lock()
_EF_DIMENSION = 512  # BAAI/bge-small-zh-v1.5 输出维度
# Bug fix: ChromaDB 专用线程池，避免与默认 ThreadPoolExecutor 竞争导致死锁
_CHROMA_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_CHROMA_EXECUTOR_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


class _SentenceTransformerWrapper:
    """将 SentenceTransformer 包装为 chromadb EmbeddingFunction。"""

    def __init__(self, model):
        self._model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def name() -> str:
        return "BAAI/bge-small-zh-v1.5"


# ---------------------------------------------------------------------------
# HNSW 参数配置
# ---------------------------------------------------------------------------

def _hnsw_params(collection_size: int = 0) -> dict[str, Any]:
    """根据集合大小返回 HNSW 参数。

    - 小规模 (<10K)：快速构建，低内存
    - 中等规模 (<100K)：平衡精度和速度
    - 大规模 (>100K)：高精度，更多内存
    """
    if collection_size < 10_000:
        return {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:M": 16,
            "hnsw:search_ef": 50,
            "hnsw:num_threads": 1,
            "hnsw:batch_size": 100,
            "hnsw:sync_threshold": 100,
        }
    elif collection_size < 100_000:
        return {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:M": 32,
            "hnsw:search_ef": 100,
            "hnsw:num_threads": 1,
            "hnsw:batch_size": 1000,
            "hnsw:sync_threshold": 1000,
        }
    else:
        return {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 400,
            "hnsw:M": 48,
            "hnsw:search_ef": 200,
            "hnsw:num_threads": 1,
            "hnsw:batch_size": 10000,
            "hnsw:sync_threshold": 10000,
        }


# ---------------------------------------------------------------------------
# 嵌入器缓存
# ---------------------------------------------------------------------------

def get_embedder() -> Any:
    """获取缓存的 embedding function，全局只加载一次模型。

    使用 BAAI/bge-small-zh-v1.5 中文语义模型（512维）。
    参考 mempalace `embedding.py` 的 `_EF_CACHE` 全局缓存模式。
    """
    global _EF_DIMENSION
    cache_key = "default"
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _EF_CACHE_LOCK:
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # 首选：本地 bge-small-zh ONNX 中文嵌入器（免 torch、免网络下载、CPU 可跑）
        try:
            from unified_memory.store.onnx_zh_embedder import get_bge_onnx_embedding_function

            onnx_ef = get_bge_onnx_embedding_function()
            if onnx_ef is not None:
                _EF_DIMENSION = 512  # Bug fix: 现在在 _EF_CACHE_LOCK 保护内写入
                _EF_CACHE[cache_key] = onnx_ef
                return onnx_ef
        except Exception as e:  # noqa: BLE001
            logger.warning("bge ONNX 中文嵌入器加载失败，走原回退链: %s", e)
        # 原回退链：bge-small-zh (sentence-transformers) -> ONNXMiniLM_L6_V2
        # 优先使用本地缓存的 bge-small-zh-v1.5
        model_name = "BAAI/bge-small-zh-v1.5"
        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(model_name, local_files_only=True)
            _EF_DIMENSION = 512
            # 包装为 chromadb 可用的 EmbeddingFunction
            ef = _SentenceTransformerWrapper(st)
            _EF_CACHE[cache_key] = ef
            logger.info("嵌入器已加载: %s (dim=%d, local)", model_name, _EF_DIMENSION)
            return ef
        except Exception as e:
            logger.warning("bge-small-zh 本地加载失败 (%s), 尝试在线加载...", e)
            # 清除 OFFLINE 标志，让 fallback 能正常下载
            if "TRANSFORMERS_OFFLINE" in os.environ:
                del os.environ["TRANSFORMERS_OFFLINE"]
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
                _EF_DIMENSION = 512
                _EF_CACHE[cache_key] = ef
                logger.info("嵌入器已加载: %s (dim=%d, online)", model_name, _EF_DIMENSION)
                return ef
            except Exception as e2:
                logger.warning("bge-small-zh 在线加载也失败 (%s), 回退到 ONNXMiniLM_L6_V2", e2)
                ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
                fallback_dim = 384
                # 如果之前已用 512 维创建了集合，回退到 384 维会导致维度不匹配
                if _EF_DIMENSION != fallback_dim:
                    logger.error(
                        "嵌入器维度不匹配：已用 %d 维创建集合，但回退嵌入器为 %d 维。"
                        "请删除 ChromaDB 持久化目录后重启以重建集合。",
                        _EF_DIMENSION, fallback_dim,
                    )
                _EF_DIMENSION = fallback_dim
                _EF_CACHE[cache_key] = ef
                logger.info("嵌入器已加载: ONNXMiniLM_L6_V2 (fallback, dim=%d)", fallback_dim)
                return ef


# ---------------------------------------------------------------------------
# ChromaDB Store
# ---------------------------------------------------------------------------

def shutdown_chroma_executor() -> None:
    """关闭 ChromaDB 专用线程池（Bug fix: 防止 ResourceWarning）。

    在应用 shutdown 时调用，显式关闭线程池。
    """
    global _CHROMA_EXECUTOR
    if _CHROMA_EXECUTOR is not None:
        try:
            _CHROMA_EXECUTOR.shutdown(wait=False)
        except Exception:
            pass
        finally:
            _CHROMA_EXECUTOR = None


def get_chroma_executor() -> concurrent.futures.ThreadPoolExecutor:
    """获取 ChromaDB 专用线程池（Bug fix: 避免与 asyncio 默认线程池竞争）。

    ChromaDB 的同步操作在 run_in_executor 中执行，
    使用专用线程池防止所有线程被 ChromaDB I/O 占用后
    管线中其他 run_in_executor 调用全部阻塞。
    """
    global _CHROMA_EXECUTOR
    if _CHROMA_EXECUTOR is not None:
        return _CHROMA_EXECUTOR
    with _CHROMA_EXECUTOR_LOCK:
        if _CHROMA_EXECUTOR is None:
            _CHROMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="chroma_worker",
            )
        return _CHROMA_EXECUTOR


class ChromaStore:
    """优化版 ChromaDB 存储后端。

    支持 HNSW 自适应参数、批量写入、嵌入器缓存、元数据索引。
    """

    # Bug fix: _seen_hashes 上限，防止长期运行进程无限增长导致内存泄漏。
    # 50K 覆盖绝大多数使用场景（restart 后从空开始，持续运行数月也很难突破此值）。
    _MAX_SEEN_HASHES: int = 50000

    def __init__(self, persist_dir: str, collection_name: str = "memories"):
        """
        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: 集合名称
        """
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client: Optional[PersistentClient] = None
        self._collection: Any = None
        self._embedder = get_embedder()
        # 写入去重缓存（内容 hash → bool，有上限防泄漏）
        self._seen_hashes: set[str] = set()
        self._batch_lock = threading.Lock()
        # Bug fix: 保护 _ensure_collection 的线程安全，防止多线程同时创建 collection
        self._collection_lock = threading.Lock()

    def _ensure_client(self) -> None:
        """确保 PersistentClient 已创建。"""
        if self._client is None:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)

    def _add_safe(self, documents, ids, metadatas) -> None:
        """幂等地写入文档。

        内存去重缓存 `_seen_hashes` 进程重启即丢失，重启后再次写入相同内容会
        产生相同的 doc_id，ChromaDB 会抛 "ID already exists"。这里捕获该异常并
        降级为 upsert，保证写入幂等、不再崩溃。
        """
        try:
            self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
        except Exception as e:  # 兼容不同版本：DuplicateIDError / ValueError("ID already exists")
            if "already exists" in str(e) or "DuplicateID" in str(e):
                logger.debug("ChromaDB 文档 ID 已存在，降级为 upsert: %s", e)
                self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            else:
                raise

    def _distance_to_score(self, distance: float) -> float:
        """余弦距离（ChromaDB 返回，范围 [0, 2]）转相似度分数。

        相似度 = 1 - distance，但当两向量方向相反时 distance 可达 2，分数会变负。
        钳制到 [0, 1] 区间，避免下游融合/排序出现负分。
        """
        return max(0.0, 1.0 - distance)

    def _ensure_collection(self, create: bool = True) -> None:
        """确保 collection 已存在（线程安全）。

        Bug fix: 添加 _collection_lock 防止多线程同时创建/获取 collection
        导致 ChromaDB 内部状态不一致或重复创建错误。
        """
        self._ensure_client()
        if self._collection is not None:
            return
        with self._collection_lock:
            # 双重检查：锁内再确认一次
            if self._collection is not None:
                return
            try:
                self._collection = self._client.get_collection(
                    self._collection_name,
                    embedding_function=self._embedder,
                )
                logger.info("ChromaDB 集合已加载: %s", self._collection_name)
            except (ValueError, chromadb.errors.NotFoundError):
                if not create:
                    raise
                params = _hnsw_params(0)
                self._collection = self._client.create_collection(
                    self._collection_name,
                    metadata=params,
                    embedding_function=self._embedder,
                )
                logger.info(
                    "ChromaDB 集合已创建: %s (HNSW params: %s)",
                    self._collection_name, params,
                )

    # ------------------------------------------------------------------
    # 写入操作
    # ------------------------------------------------------------------

    def _trim_seen_hashes_if_needed(self) -> None:
        """防止 _seen_hashes 无限增长（Bug fix: 长期运行进程内存泄漏）。

        超过阈值时清除一半条目，后续写入重新建立去重缓存。
        使用 set-pop 采样（无顺序保证，但哈希值均匀分布）。
        """
        if len(self._seen_hashes) >= self._MAX_SEEN_HASHES:
            trim_count = len(self._seen_hashes) // 2
            for _ in range(trim_count):
                try:
                    self._seen_hashes.pop()
                except KeyError:
                    break
            logger.debug(
                "_seen_hashes 清理 %d 条（当前 %d）",
                trim_count, len(self._seen_hashes),
            )

    def add_drawer(
        self,
        content: str,
        wing: str = "default",
        room: str = "general",
        metadata: Optional[dict] = None,
    ) -> str:
        """写入一条记录，支持 wing/room 两级元数据索引。

        Args:
            content: 文本内容
            wing: 所属 wing
            room: 所属 room
            metadata: 附加元数据

        Returns:
            文档 ID
        """
        self._ensure_collection()

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            return content_hash
        self._trim_seen_hashes_if_needed()
        self._seen_hashes.add(content_hash)

        doc_id = f"{wing}/{room}/{content_hash[:16]}"
        meta: dict[str, Any] = {
            "wing": wing,
            "room": room,
            "content_hash": content_hash,
            "created_at": time.time(),
        }
        if metadata:
            meta.update(metadata)

        self._add_safe(
            documents=[content],
            ids=[doc_id],
            metadatas=[meta],
        )
        return doc_id

    def add_batch(self, items: list[dict]) -> list[str]:
        """批量写入多条记录，一次 embedding 调用。

        Args:
            items: 写入项列表，每项包含 content, wing, room, metadata 等

        Returns:
            写入的文档 ID 列表
        """
        if not items:
            return []
        self._ensure_collection()

        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        result_ids: list[str] = []

        for item in items:
            content = item["content"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash in self._seen_hashes:
                continue
            self._trim_seen_hashes_if_needed()
            self._seen_hashes.add(content_hash)

            wing = item.get("wing", "default")
            room = item.get("room", "general")
            doc_id = item.get("id", f"{wing}/{room}/{content_hash[:16]}")

            documents.append(content)
            ids.append(doc_id)
            metadatas.append({
                "wing": wing,
                "room": room,
                "content_hash": content_hash,
                "created_at": time.time(),
                **(item.get("metadata") or {}),
            })
            result_ids.append(doc_id)

        if not documents:
            return []

        with self._batch_lock:
            self._add_safe(documents=documents, ids=ids, metadatas=metadatas)
        logger.debug("批量写入 %d 条到 ChromaDB", len(documents))
        return result_ids

    def delete(self, doc_id: str) -> None:
        """删除一条记录。

        Args:
            doc_id: 文档 ID
        """
        self._ensure_collection()
        self._collection.delete(ids=[doc_id])

    def delete_batch(self, doc_ids: list[str]) -> None:
        """批量删除多条记录（Bug fix: 新增方法，避免压缩任务回退到逐条删除）。

        Args:
            doc_ids: 文档 ID 列表
        """
        if not doc_ids:
            return
        self._ensure_collection()
        self._collection.delete(ids=doc_ids)
        logger.debug("批量删除 %d 条 ChromaDB 记录", len(doc_ids))

    # ------------------------------------------------------------------
    # 搜索操作
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[dict]:
        """语义搜索，支持 wing/room 两级元数据过滤。

        Args:
            query: 查询文本
            n_results: 返回条数
            wing: 可选，按 wing 过滤
            room: 可选，按 room 过滤

        Returns:
            搜索结果列表，每项含 id, content, metadata, score
        """
        self._ensure_collection()

        # 空集合保护
        if self._collection.count() == 0:
            return []

        where: dict[str, Any] = {}
        if wing is not None:
            where["wing"] = wing
        if room is not None:
            where["room"] = room

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for i, doc_id in enumerate(results["ids"][0] if results["ids"] else []):
            output.append({
                "id": doc_id,
                "content": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "score": self._distance_to_score(results["distances"][0][i]) if results.get("distances") else 0,
            })
        logger.debug("语义搜索 '%s': %d 条结果", query[:50], len(output))
        return output

    def search_by_embedding(
        self,
        embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """按向量搜索。

        Args:
            embedding: 查询向量
            n_results: 返回条数
            where: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        self._ensure_collection()
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict] = []
        for i, doc_id in enumerate(results["ids"][0] if results["ids"] else []):
            output.append({
                "id": doc_id,
                "content": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "score": self._distance_to_score(results["distances"][0][i]) if results.get("distances") else 0,
            })
        return output

    def search_by_metadata(
        self,
        where: dict,
        limit: int = 100,
    ) -> list[dict]:
        """按元数据过滤查询（无需 embedding）。

        Args:
            where: 元数据过滤条件
            limit: 返回条数

        Returns:
            匹配的记录列表
        """
        self._ensure_collection()
        results = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        output: list[dict] = []
        for i, doc_id in enumerate(results["ids"]):
            output.append({
                "id": doc_id,
                "content": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            })
        return output

    def get_status(self) -> dict:
        """获取 ChromaDB 状态信息。

        Returns:
            {"collection": name, "count": N, "persist_dir": path}
        """
        self._ensure_collection()
        return {
            "collection": self._collection_name,
            "count": self._collection.count(),
            "persist_dir": self._persist_dir,
        }