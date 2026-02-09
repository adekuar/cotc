"""Checkpoint storage for DAG state persistence."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from .models import TaskDAG, TaskStatus

logger = logging.getLogger(__name__)


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    async def save(self, key: str, data: str) -> None:
        """Save data to storage."""
        pass

    @abstractmethod
    async def load(self, key: str) -> str | None:
        """Load data from storage. Returns None if not found."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data from storage."""
        pass


class InMemoryStorage(CheckpointStorage):
    """In-memory checkpoint storage for development/testing."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def save(self, key: str, data: str) -> None:
        """Save data to in-memory store."""
        self._store[key] = data

    async def load(self, key: str) -> str | None:
        """Load data from in-memory store."""
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        """Delete data from in-memory store."""
        self._store.pop(key, None)


class RedisStorage(CheckpointStorage):
    """Redis-backed checkpoint storage."""

    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "chatbot:checkpoint:"):
        self.prefix = prefix
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url)
        except ImportError:
            raise ImportError("redis package not installed. Run: pip install redis")

    async def save(self, key: str, data: str) -> None:
        """Save data to Redis."""
        await self._redis.set(self.prefix + key, data)

    async def load(self, key: str) -> str | None:
        """Load data from Redis."""
        result = await self._redis.get(self.prefix + key)
        if result is None:
            return None
        return result.decode() if isinstance(result, bytes) else result

    async def delete(self, key: str) -> None:
        """Delete data from Redis."""
        await self._redis.delete(self.prefix + key)


class CheckpointManager:
    """Manages saving and loading DAG checkpoints."""

    def __init__(self, storage: CheckpointStorage):
        self.storage = storage

    async def save_checkpoint(self, dag: TaskDAG) -> None:
        """Save the current DAG state."""
        # Support both pydantic v1 and v2
        if hasattr(dag, "model_dump_json"):
            data = dag.model_dump_json()
        else:
            data = dag.json()
        await self.storage.save(dag.id, data)
        logger.debug(f"Saved checkpoint for DAG {dag.id}")

    async def load_checkpoint(self, dag_id: str) -> TaskDAG | None:
        """Load a DAG from checkpoint."""
        data = await self.storage.load(dag_id)
        if data is None:
            return None

        # Support both pydantic v1 and v2
        if hasattr(TaskDAG, "model_validate_json"):
            dag = TaskDAG.model_validate_json(data)
        else:
            dag = TaskDAG.parse_raw(data)
        logger.debug(f"Loaded checkpoint for DAG {dag_id}")
        return dag

    async def delete_checkpoint(self, dag_id: str) -> None:
        """Delete a DAG checkpoint."""
        await self.storage.delete(dag_id)
        logger.debug(f"Deleted checkpoint for DAG {dag_id}")

    async def prepare_for_resume(self, dag: TaskDAG) -> None:
        """Prepare a DAG for resuming execution.

        Resets any RUNNING tasks back to PENDING so they can be re-executed.
        """
        for task in dag.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PENDING
                logger.info(f"Reset running task {task.id} to pending for resume")
