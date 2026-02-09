"""High-level API for the chatbot system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .aggregator import ResultAggregator
from .checkpoint import CheckpointManager, CheckpointStorage, InMemoryStorage
from .decomposition import DecompositionEngine
from .executor import DAGExecutor, ExecutionConfig
from .llm import LLMProvider, OpenAIProvider
from .models import TaskDAG
from .tools import ToolRegistry, create_default_registry


@dataclass
class ChatResponse:
    """Response from the chatbot."""

    answer: str
    status: str  # "complete" or "partial"
    completion_rate: float
    failed_tasks: int
    skipped_tasks: int
    dag_id: str
    task_details: dict[str, Any] | None = None


class ChatbotAPI:
    """High-level API for the fault-tolerant chatbot."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        storage: CheckpointStorage | None = None,
        config: ExecutionConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm or OpenAIProvider()
        self.storage = storage or InMemoryStorage()
        self.config = config or ExecutionConfig()
        self.tool_registry = tool_registry or create_default_registry()

        self.checkpoint_manager = CheckpointManager(self.storage)
        self.decomposition_engine = DecompositionEngine(self.llm)
        self.executor = DAGExecutor(
            llm=self.llm,
            decomposition_engine=self.decomposition_engine,
            checkpoint_manager=self.checkpoint_manager,
            config=self.config,
            tool_registry=self.tool_registry,
        )
        self.aggregator = ResultAggregator(self.llm)

    async def ask(self, question: str, include_details: bool = False) -> ChatResponse:
        """Process a user question through the chatbot."""
        # Decompose the question into tasks
        dag = await self.decomposition_engine.decompose_question(question)

        # Execute the DAG
        execution_result = await self.executor.execute_dag(dag)

        # Aggregate results
        answer = await self.aggregator.aggregate_results(dag)

        # Get summary
        summary = self.aggregator.get_partial_result_summary(dag)

        # Build task details if requested
        task_details = None
        if include_details:
            task_details = self._build_task_details(dag)

        return ChatResponse(
            answer=answer,
            status="complete" if summary["is_complete"] else "partial",
            completion_rate=summary["completion_rate"],
            failed_tasks=summary["failed_tasks"],
            skipped_tasks=summary["skipped_tasks"],
            dag_id=dag.id,
            task_details=task_details,
        )

    async def resume(self, dag_id: str, include_details: bool = False) -> ChatResponse | None:
        """Resume a previously interrupted request."""
        # Load checkpoint
        dag = await self.checkpoint_manager.load_checkpoint(dag_id)
        if dag is None:
            return None

        # Prepare for resume
        await self.checkpoint_manager.prepare_for_resume(dag)

        # Execute remaining tasks
        execution_result = await self.executor.execute_dag(dag)

        # Aggregate results
        answer = await self.aggregator.aggregate_results(dag)

        # Get summary
        summary = self.aggregator.get_partial_result_summary(dag)

        # Build task details if requested
        task_details = None
        if include_details:
            task_details = self._build_task_details(dag)

        return ChatResponse(
            answer=answer,
            status="complete" if summary["is_complete"] else "partial",
            completion_rate=summary["completion_rate"],
            failed_tasks=summary["failed_tasks"],
            skipped_tasks=summary["skipped_tasks"],
            dag_id=dag.id,
            task_details=task_details,
        )

    def _build_task_details(self, dag: TaskDAG) -> dict[str, Any]:
        """Build full task details from a DAG."""
        return {
            "dag_id": dag.id,
            "user_question": dag.user_question,
            "tasks": [
                {
                    "id": task.id,
                    "parent_id": task.parent_id,
                    "name": task.name,
                    "prompt": task.prompt,
                    "depends_on": task.depends_on,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                    "retry_count": task.retry_count,
                    "is_complex": task.is_complex,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": tc.result,
                        }
                        for tc in task.tool_calls
                    ],
                }
                for task in dag.tasks.values()
            ],
            "dead_letter_queue": len(self.executor.dead_letter_queue),
            "stats": dag.get_completion_stats(),
        }

    async def get_dag_status(self, dag_id: str) -> dict[str, Any] | None:
        """Get the current status of a DAG."""
        dag = await self.checkpoint_manager.load_checkpoint(dag_id)
        if dag is None:
            return None

        stats = dag.get_completion_stats()

        return {
            "dag_id": dag.id,
            "user_question": dag.user_question,
            "stats": stats,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "retry_count": task.retry_count,
                }
                for task in dag.tasks.values()
            ],
        }

    def get_dead_letter_entries(self, dag_id: str | None = None) -> list[dict]:
        """Get entries from the dead letter queue."""
        entries = self.executor.dead_letter_queue.get_entries(dag_id)
        return [
            {
                "task_id": e.task_id,
                "dag_id": e.dag_id,
                "error": e.error,
                "retry_count": e.retry_count,
                "timestamp": e.timestamp,
            }
            for e in entries
        ]
