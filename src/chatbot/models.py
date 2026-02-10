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
    requires_data: bool = True  # Whether this task requires external data to be meaningful
    has_tool_failures: bool = False  # Set True if task completed but had critical tool failures

    def has_critical_tool_failures(self) -> bool:
        """Check if this task has critical tool failures that prevent meaningful completion."""
        if not self.tool_calls:
            return False

        # Count failed vs successful tool calls
        failed_calls = [tc for tc in self.tool_calls if tc.result and not tc.result.get("success", False)]
        successful_calls = [tc for tc in self.tool_calls if tc.result and tc.result.get("success", False)]

        # If all tool calls failed, this is a critical failure
        if failed_calls and not successful_calls:
            return True

        # If more than half of tool calls failed, consider it critical
        total_calls = len(self.tool_calls)
        if total_calls > 0 and len(failed_calls) / total_calls > 0.5:
            return True

        return False

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
        """Get all tasks that are ready to execute (dependencies satisfied).

        Tasks are NOT ready if any dependency has critical tool failures,
        preventing downstream hallucination when upstream data acquisition failed.
        """
        ready = []

        # Identify tasks with critical failures (completed but with tool failures)
        failed_completion_ids = {
            tid for tid, t in self.tasks.items()
            if t.status == TaskStatus.COMPLETED and t.has_tool_failures
        }

        completed_or_skipped = {
            tid
            for tid, t in self.tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        }

        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Block if any dependency has critical failures
            if any(dep_id in failed_completion_ids for dep_id in task.depends_on):
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

    def get_tasks_with_critical_failures(self) -> list[TaskNode]:
        """Get list of tasks that have critical tool failures."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED and task.has_critical_tool_failures()
        ]

    def needs_user_help(self) -> bool:
        """Check if the DAG execution needs user help due to critical failures."""
        # Check for explicitly failed tasks
        failed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        if failed_tasks:
            return True

        # Check for tasks with critical tool failures
        critical_failures = self.get_tasks_with_critical_failures()
        if critical_failures:
            return True

        return False


class DecompositionResult(BaseModel):
    """Result from decomposing a question or task."""

    tasks: list[TaskNode]
    aggregation_instructions: str


class ComplexityCheckResult(BaseModel):
    """Result from checking task complexity."""

    is_simple: bool
    sub_tasks: list[TaskNode] = Field(default_factory=list)
