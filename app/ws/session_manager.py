from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class SessionWebSocketManager:
    """
    In-memory WS session manager (free-first).

    Production scaling note:
    - If you run multiple API instances, you must replace this with Redis pub/sub or similar.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._connections: dict[str, set[WebSocket]] = {}  # key: session_user_key

    @staticmethod
    def _key(session_id: str, user_id: str) -> str:
        return f"{session_id}:{user_id}"

    async def connect(self, *, session_id: str, user_id: str, websocket: WebSocket):
        key = self._key(session_id, user_id)
        async with self._lock:
            conns = self._connections.setdefault(key, set())
            conns.add(websocket)

    async def disconnect(self, *, session_id: str, user_id: str, websocket: WebSocket):
        key = self._key(session_id, user_id)
        async with self._lock:
            conns = self._connections.get(key)
            if not conns:
                return
            conns.discard(websocket)
            if not conns:
                self._connections.pop(key, None)

    async def send_json(self, *, session_id: str, user_id: str, message: dict[str, Any]):
        key = self._key(session_id, user_id)
        async with self._lock:
            conns = list(self._connections.get(key, set()))

        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                # Best-effort; connection may have dropped.
                continue


session_manager = SessionWebSocketManager()

