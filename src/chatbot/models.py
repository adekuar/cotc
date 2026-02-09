"""Core data structures for the task DAG system."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a task in the DAG."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCallRecord(BaseModel):
    """Record of a tool call made during task execution."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None


class TaskNode(BaseModel):
    """A single task node in the DAG."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    name: str = ""
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    retry_count: int = 0
    is_complex: bool = False
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    def reset_for_retry(self) -> None:
        """Reset task state for retry."""
        self.status = TaskStatus.PENDING
        self.error = None


class TaskDAG(BaseModel):
    """Directed Acyclic Graph of tasks."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_question: str
    tasks: dict[str, TaskNode] = Field(default_factory=dict)
    final_prompt: str = ""

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the DAG."""
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> TaskNode | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks that are ready to execute (dependencies satisfied)."""
        ready = []
        completed_or_skipped = {
            tid
            for tid, t in self.tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        }

        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            deps_satisfied = all(
                dep_id in completed_or_skipped for dep_id in task.depends_on
            )
            if deps_satisfied:
                ready.append(task)

        return ready

    def all_tasks_done(self) -> bool:
        """Check if all tasks are in a terminal state."""
        terminal_states = {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }
        return all(t.status in terminal_states for t in self.tasks.values())

    def get_completion_stats(self) -> dict[str, Any]:
        """Get completion statistics for the DAG."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "completion_rate": completed / total if total > 0 else 0,
        }


class DecompositionResult(BaseModel):
    """Result from decomposing a question or task."""

    tasks: list[TaskNode]
    aggregation_instructions: str


class ComplexityCheckResult(BaseModel):
    """Result from checking task complexity."""

    is_simple: bool
    sub_tasks: list[TaskNode] = Field(default_factory=list)
