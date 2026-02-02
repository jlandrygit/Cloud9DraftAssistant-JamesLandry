"""DraftStateStore abstraction and in-memory implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from core.draft_state import DraftState


class DraftStateStore(Protocol):
    """Interface for storing live draft sessions.

    Thread safety depends on the concrete implementation. The in-memory store
    provided here uses a coarse lock and is safe for a single-process demo.
    """

    def get(self, session_id: str) -> DraftState | None:
        """Return the DraftState for a session, or None if missing/expired."""
        ...

    def set(self, session_id: str, state: DraftState) -> None:
        """Store the DraftState for a session."""
        ...

    def delete(self, session_id: str) -> None:
        """Delete a session if it exists."""
        ...

    def count(self) -> int:
        """Return the number of active sessions."""
        ...

    def ttl_seconds(self) -> int:
        """Return the configured TTL in seconds."""
        ...


@dataclass(frozen=True)
class _Entry:
    state: DraftState
    expires_at: float


class InMemoryDraftStateStore:
    """In-memory DraftState store with TTL eviction.

    Thread safety: this implementation uses a single lock to protect access.
    It is suitable for a single-process demo and is not shared across workers.
    """

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._entries: dict[str, _Entry] = {}

    def get(self, session_id: str) -> DraftState | None:
        """Return the DraftState for a session, or None if missing/expired."""
        with self._lock:
            self._evict_expired_locked()
            entry = self._entries.get(session_id)
            return entry.state if entry else None

    def set(self, session_id: str, state: DraftState) -> None:
        """Store the DraftState for a session with refreshed TTL."""
        with self._lock:
            expires_at = time.time() + self._ttl_seconds
            self._entries[session_id] = _Entry(state=state, expires_at=expires_at)

    def delete(self, session_id: str) -> None:
        """Delete a session if it exists."""
        with self._lock:
            self._entries.pop(session_id, None)

    def count(self) -> int:
        """Return the number of active sessions."""
        with self._lock:
            self._evict_expired_locked()
            return len(self._entries)

    def ttl_seconds(self) -> int:
        """Return the configured TTL in seconds."""
        return self._ttl_seconds

    def _evict_expired_locked(self) -> None:
        """Evict expired entries. Caller must hold the lock."""
        now = time.time()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
