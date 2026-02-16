"""JSON-file–based Chainlit data layer.

Stores users and conversation threads as JSON files under `.chat_data/`.
Zero external dependencies — uses only the Python standard library.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from chainlit.data import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

# ── Storage paths ────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), ".chat_data")
_USERS_FILE = os.path.join(_ROOT, "users.json")
_THREADS_DIR = os.path.join(_ROOT, "threads")

_lock = asyncio.Lock()


# ── Disk helpers ─────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    os.makedirs(_THREADS_DIR, exist_ok=True)


def _read_json(path: str) -> dict | list:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict | list) -> None:
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _thread_path(thread_id: str) -> str:
    return os.path.join(_THREADS_DIR, f"{thread_id}.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data layer ───────────────────────────────────────────────────────

class JsonDataLayer(BaseDataLayer):
    """Persist Chainlit threads to local JSON files."""

    # ── Users ────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        async with _lock:
            users = _read_json(_USERS_FILE)
        u = users.get(identifier)
        if not u:
            return None
        return PersistedUser(
            id=u["id"],
            identifier=identifier,
            metadata=u.get("metadata", {}),
            createdAt=u.get("createdAt", _now_iso()),
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        now = _now_iso()
        async with _lock:
            users = _read_json(_USERS_FILE)
            if user.identifier not in users:
                users[user.identifier] = {
                    "id": user.identifier,
                    "metadata": user.metadata or {},
                    "createdAt": now,
                }
                _write_json(_USERS_FILE, users)
            u = users[user.identifier]
        return PersistedUser(
            id=u["id"],
            identifier=user.identifier,
            metadata=u.get("metadata", {}),
            createdAt=u.get("createdAt", now),
        )

    # ── Threads ──────────────────────────────────────────────────────

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        path = _thread_path(thread_id)
        async with _lock:
            data = _read_json(path)
        if data:
            # Compat: ensure userIdentifier exists (Chainlit requires it)
            if "userIdentifier" not in data and "userId" in data:
                data["userIdentifier"] = data["userId"]
        return data or None

    async def get_thread_author(self, thread_id: str) -> str:
        thread = await self.get_thread(thread_id)
        if thread:
            return thread.get("userIdentifier") or thread.get("userId") or ""
        return ""

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        async with _lock:
            path = _thread_path(thread_id)
            thread = _read_json(path) or {
                "id": thread_id,
                "createdAt": _now_iso(),
                "steps": [],
            }
            if name is not None:
                thread["name"] = name
            if user_id is not None:
                thread["userId"] = user_id
                thread["userIdentifier"] = user_id  # Save both for compat
            if metadata is not None:
                thread["metadata"] = metadata
            if tags is not None:
                thread["tags"] = tags
            _write_json(path, thread)

    async def delete_thread(self, thread_id: str) -> None:
        path = _thread_path(thread_id)
        async with _lock:
            if os.path.exists(path):
                os.remove(path)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        _ensure_dirs()
        threads: list[ThreadDict] = []
        async with _lock:
            for fname in os.listdir(_THREADS_DIR):
                if not fname.endswith(".json"):
                    continue
                data = _read_json(os.path.join(_THREADS_DIR, fname))
                if not data:
                    continue
                
                # Compat: ensure userIdentifier exists
                if "userIdentifier" not in data and "userId" in data:
                    data["userIdentifier"] = data["userId"]

                # Apply user filter
                if filters.userId and data.get("userId") != filters.userId:
                    continue
                # Apply search filter
                if filters.search:
                    name = data.get("name", "")
                    if filters.search.lower() not in name.lower():
                        continue
                threads.append(data)

        # Sort newest-first
        threads.sort(key=lambda t: t.get("createdAt", ""), reverse=True)

        # Paginate with cursor-based pagination
        start = 0
        if pagination.cursor:
            for i, t in enumerate(threads):
                if t["id"] == pagination.cursor:
                    start = i + 1
                    break

        page = threads[start : start + pagination.first]
        has_next = (start + pagination.first) < len(threads)

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=page,
        )

    # ── Steps (messages & agent actions) ─────────────────────────────

    async def create_step(self, step_dict: dict) -> None:
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        async with _lock:
            path = _thread_path(thread_id)
            thread = _read_json(path) or {
                "id": thread_id,
                "createdAt": _now_iso(),
                "steps": [],
            }
            steps = thread.setdefault("steps", [])
            steps.append(step_dict)
            _write_json(path, thread)

    async def update_step(self, step_dict: dict) -> None:
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        step_id = step_dict.get("id")
        async with _lock:
            path = _thread_path(thread_id)
            thread = _read_json(path)
            if not thread:
                return
            for i, s in enumerate(thread.get("steps", [])):
                if s.get("id") == step_id:
                    thread["steps"][i] = step_dict
                    break
            _write_json(path, thread)

    async def delete_step(self, step_id: str) -> None:
        # Iterate all threads (rare operation)
        _ensure_dirs()
        async with _lock:
            for fname in os.listdir(_THREADS_DIR):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(_THREADS_DIR, fname)
                thread = _read_json(path)
                if not thread:
                    continue
                original_len = len(thread.get("steps", []))
                thread["steps"] = [
                    s for s in thread.get("steps", []) if s.get("id") != step_id
                ]
                if len(thread["steps"]) < original_len:
                    _write_json(path, thread)
                    return

    # ── Elements (files, images) ─────────────────────────────────────

    async def create_element(self, element: dict) -> None:
        pass  # Not persisting binary elements for now

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[dict]:
        return None

    async def delete_element(
        self, element_id: str, thread_id: Optional[str] = None
    ) -> None:
        pass

    # ── Feedback ─────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return feedback.id or ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    # ── Misc ─────────────────────────────────────────────────────────

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> List[dict]:
        return []
