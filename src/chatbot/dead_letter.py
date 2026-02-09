"""Dead letter queue for failed tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .models import TaskNode


@dataclass
class DeadLetterEntry:
    """An entry in the dead letter queue."""

    task_id: str
    dag_id: str
    task_name: str
    prompt: str
    error: str
    retry_count: int
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] | None = None


class DeadLetterQueue:
    """Queue for tasks that have permanently failed."""

    def __init__(self) -> None:
        self._entries: list[DeadLetterEntry] = []

    def add(
        self,
        task: TaskNode,
        dag_id: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a failed task to the dead letter queue."""
        entry = DeadLetterEntry(
            task_id=task.id,
            dag_id=dag_id,
            task_name=task.name,
            prompt=task.prompt,
            error=error,
            retry_count=task.retry_count,
            metadata=metadata,
        )
        self._entries.append(entry)

    def get_entries(self, dag_id: str | None = None) -> list[DeadLetterEntry]:
        """Get entries, optionally filtered by DAG ID."""
        if dag_id is None:
            return list(self._entries)
        return [e for e in self._entries if e.dag_id == dag_id]

    def clear(self, dag_id: str | None = None) -> int:
        """Clear entries, optionally for a specific DAG. Returns count removed."""
        if dag_id is None:
            count = len(self._entries)
            self._entries.clear()
            return count

        original_len = len(self._entries)
        self._entries = [e for e in self._entries if e.dag_id != dag_id]
        return original_len - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
