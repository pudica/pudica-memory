"""Pudica Memory Client for BEAM Benchmark.

Async client that replaces Mem0Client with pudica-memory HTTP API
running on http://127.0.0.1:8420.

Implements the same interface as Mem0Client:
  - add(messages, user_id, ...) -> ingest conversation turns
  - search(query, user_id, top_k, ...) -> retrieve memories
  - delete_user(user_id) -> clear memories for a user
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class PudicaClient:
    """Async client for pudica-memory HTTP API.

    Connects to the local memory service at port 8420.
    """

    def __init__(
        self,
        host: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
    ):
        self.host = (host or os.getenv("PUDICA_MEMORY_HOST", "http://127.0.0.1:8420")).rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict | None = None,
    ) -> dict:
        """Make an HTTP request with retry logic."""
        url = f"{self.host}{path}"
        session = await self._get_session()

        for attempt in range(self.max_retries):
            try:
                async with session.request(method, url, json=json_data) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    text = await resp.text()
                    logger.warning(
                        "Pudica API %s %s returned %d: %s",
                        method, path, resp.status, text[:200],
                    )
                    if resp.status in (429, 502, 503, 504):
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return {"error": text, "status": resp.status}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "Pudica API %s %s attempt %d/%d failed: %s",
                    method, path, attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict],
        user_id: str = "default",
        **kwargs: Any,
    ) -> dict:
        """Ingest conversation turns into memory.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            user_id: User identifier (used as source tag).
            **kwargs: Additional options (ignored).

        Returns:
            Dict with ingestion result.
        """
        # Format conversation as text
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"[{role}]: {content}\n"

        # Ingest with metadata
        result = await self._request("POST", "/api/v1/ingest", {
            "content": conversation_text,
            "source": f"beam_{user_id}",
            "metadata": {
                "source": f"beam_{user_id}",
                "user_id": user_id,
                "timestamp": time.time(),
            },
        })

        memory_id = result.get("id", "")
        return {
            "id": memory_id,
            "messages": messages,
            "memory": conversation_text,
            "event": "added",
            "success": "error" not in result,
        }

    async def add_batch(
        self,
        memories: list[dict],
        user_id: str = "default",
        **kwargs: Any,
    ) -> list[dict]:
        """Ingest multiple memories in batch.

        Args:
            memories: List of dicts, each with 'messages' or 'content'.
            user_id: User identifier.
            **kwargs: Additional options.

        Returns:
            List of per-memory results.
        """
        results = []
        for memory in memories:
            messages = memory.get("messages", [])
            if messages:
                result = await self.add(messages, user_id=user_id, **kwargs)
            elif "content" in memory:
                result = await self._request("POST", "/api/v1/ingest", {
                    "content": memory["content"],
                    "source": f"beam_{user_id}",
                    "metadata": {
                        "source": f"beam_{user_id}",
                        "user_id": user_id,
                        "timestamp": time.time(),
                    },
                })
                result = {
                    "id": result.get("id", ""),
                    "memory": memory["content"],
                    "success": "error" not in result,
                }
            else:
                result = {"id": "", "success": False}
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 200,
        **kwargs: Any,
    ) -> list[dict]:
        """Search memories by query.

        Args:
            query: Search query string.
            user_id: User identifier (ignored, memory is shared).
            top_k: Max number of results to return.
            **kwargs: Additional options.

        Returns:
            List of search result dicts with 'memory', 'score', 'id'.
        """
        result = await self._request("POST", "/api/v1/search", {
            "query": query,
            "top_k": top_k,
        })

        raw_results = result.get("results", [])
        formatted = []
        for r in raw_results:
            formatted.append({
                "memory": r.get("text", ""),
                "score": r.get("score", 0.0),
                "id": r.get("id", ""),
                "created_at": r.get("metadata", {}).get("created_at", ""),
            })

        # Sort by score descending (BEAM expects this)
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted

    # -------------------------------------------------------------------------
    # User management
    # -------------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> dict:
        """Delete all memories for a user.

        Note: pudica-memory doesn't support per-user deletion natively.
        This is a no-op for BEAM since we use a fresh conversation per eval.
        """
        logger.info("delete_user(%s): no-op for pudica-memory", user_id)
        return {"success": True, "message": "no-op"}

    async def get_user(self, user_id: str) -> dict:
        """Get user info (not implemented for pudica-memory)."""
        return {"user_id": user_id, "memory_count": 0}