"""DAG execution engine with parallel execution and fault tolerance."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointManager
from .dead_letter import DeadLetterQueue
from .decomposition import DecompositionEngine
from .llm import LLMProvider
from .models import TaskDAG, TaskNode, TaskStatus, ToolCallRecord
from .tools import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a single task."""

    success: bool
    value: str | None = None
    error: str | None = None


@dataclass
class ExecutionConfig:
    """Configuration for DAG execution."""

    max_retries: int = 3
    base_retry_delay: float = 1.0
    task_timeout: float = 300.0
    max_parallel_tasks: int = 10
    enable_tools: bool = True
    max_tool_calls_per_task: int = 30
    allowed_tools: list[str] | None = None  # None = all tools
    skip_simplicity_check: bool = False  # Allow bypassing simplicity check for testing
    ask_for_help_on_failure: bool = True  # Ask user for help when critical tasks fail


class DAGExecutor:
    """Executes task DAGs with parallel execution and fault tolerance."""

    def __init__(
        self,
        llm: LLMProvider,
        decomposition_engine: DecompositionEngine,
        checkpoint_manager: CheckpointManager,
        config: ExecutionConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm
        self.decomposition_engine = decomposition_engine
        self.checkpoint_manager = checkpoint_manager
        self.config = config or ExecutionConfig()
        self.dead_letter_queue = DeadLetterQueue()
        self.tool_registry = tool_registry or create_default_registry()

    async def execute_dag(self, dag: TaskDAG) -> dict[str, Any]:
        """
        Execute all tasks in the DAG respecting dependencies.

        Args:
            dag: The TaskDAG to execute

        Returns:
            Dictionary with execution results and statistics
        """
        await self.checkpoint_manager.save_checkpoint(dag)

        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        while not dag.all_tasks_done():
            ready_tasks = dag.get_ready_tasks()

            if not ready_tasks and not dag.all_tasks_done():
                logger.warning("Deadlock detected: no tasks ready but DAG not complete")
                break

            # Check for complex tasks that need decomposition
            tasks_to_execute = []
            for task in ready_tasks:
                if task.is_complex:
                    try:
                        was_decomposed = await self.decomposition_engine.maybe_decompose_task(
                            dag, task
                        )
                        if was_decomposed:
                            await self.checkpoint_manager.save_checkpoint(dag)
                            continue
                    except ValueError as e:
                        logger.error(f"Decomposition error for task {task.id}: {e}")
                        task.is_complex = False

                tasks_to_execute.append(task)

            if not tasks_to_execute:
                continue

            # Execute ready tasks in parallel
            results = await self._parallel_execute(tasks_to_execute, dag)

            for task, result in results:
                if result.success:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.value
                    completed.add(task.id)
                else:
                    await self._handle_failure(dag, task, result.error or "Unknown error", failed, skipped)

            await self.checkpoint_manager.save_checkpoint(dag)

        return {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "stats": dag.get_completion_stats(),
        }

    async def _parallel_execute(
        self, tasks: list[TaskNode], dag: TaskDAG
    ) -> list[tuple[TaskNode, ExecutionResult]]:
        """Execute multiple tasks in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def execute_with_semaphore(task: TaskNode) -> tuple[TaskNode, ExecutionResult]:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                result = await self._execute_single_task(task, dag)
                return task, result

        coroutines = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            task = tasks[i]
            if isinstance(result, Exception):
                processed_results.append(
                    (task, ExecutionResult(success=False, error=str(result)))
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task(self, task: TaskNode, dag: TaskDAG) -> ExecutionResult:
        """Execute a single task with optional tool support."""
        try:
            # Build prompt with dependency results
            prompt = self._build_task_prompt(task, dag)
            system_prompt = "You are a helpful assistant completing a research sub-task."

            # If tools are disabled, use simple call
            if not self.config.enable_tools:
                response = await asyncio.wait_for(
                    self.llm.call(prompt=prompt, system_prompt=system_prompt),
                    timeout=self.config.task_timeout,
                )
                return ExecutionResult(success=True, value=response)

            # Tool-enabled execution with message loop
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            tools = self.tool_registry.get_openai_schema(self.config.allowed_tools)

            if self.config.enable_tools:
                system_prompt += (
                    "\n\nYou have access to tools that can help you complete tasks. "
                    "Use them when appropriate to gather information, perform calculations, "
                    "or interact with files."
                    "\n\nWhen using the python_execute tool:"
                    "\n- If the code requires external packages not in the standard library, first create "
                    "a temporary virtual environment, install the required packages, and then run your code."
                    "\n- Use subprocess to create and manage the virtual environment:"
                    "\n  1. Create venv: subprocess.run(['python', '-m', 'venv', '/tmp/temp_venv'])"
                    "\n  2. Install packages: subprocess.run(['/tmp/temp_venv/bin/pip', 'install', 'package_name'])"
                    "\n  3. Run code: subprocess.run(['/tmp/temp_venv/bin/python', '-c', 'your_code'])"
                    "\n- For simple calculations using only standard library, execute code directly without a venv."
                    "\n- Always clean up temporary files and environments when done."
                )

            tool_call_count = 0

            while True:
                # Check tool call limit
                if tool_call_count >= self.config.max_tool_calls_per_task:
                    logger.warning(
                        f"Task {task.id} reached max tool calls ({self.config.max_tool_calls_per_task})"
                    )
                    break

                # Call LLM with tools
                response = await asyncio.wait_for(
                    self.llm.call_with_tools(
                        messages=messages,
                        tools=tools,
                        system_prompt=system_prompt,
                    ),
                    timeout=self.config.task_timeout,
                )

                # If no tool calls, we're done
                if not response.tool_calls:
                    return ExecutionResult(success=True, value=response.content or "")

                # Add assistant message with tool calls
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages.append(assistant_msg)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_call_count += 1

                    result = await self.tool_registry.execute(
                        tool_call.name, tool_call.arguments
                    )

                    # Record the tool call in task history
                    task.tool_calls.append(
                        ToolCallRecord(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            result={
                                "success": result.success,
                                "output": result.output,
                                "error": result.error,
                            },
                        )
                    )

                    # Add tool result message
                    result_content = (
                        str(result.output) if result.success else f"Error: {result.error}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content,
                        }
                    )

                # Continue loop - LLM will process tool results

            # If we exit the loop due to max tool calls, return what we have
            return ExecutionResult(
                success=True,
                value=response.content or "Task completed (max tool calls reached)",
            )

        except asyncio.TimeoutError:
            return ExecutionResult(success=False, error="Task execution timed out")
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

    def _build_task_prompt(self, task: TaskNode, dag: TaskDAG) -> str:
        """Build the prompt for a task, including dependency results."""
        prompt_parts = [task.prompt]

        # Add context from dependencies
        dep_context = []
        for dep_id in task.depends_on:
            dep_task = dag.get_task(dep_id)
            if dep_task and dep_task.result:
                dep_context.append(f"## Result from '{dep_task.name or dep_id}':\n{dep_task.result}")
            elif dep_task and dep_task.status == TaskStatus.SKIPPED:
                dep_context.append(f"## Result from '{dep_task.name or dep_id}':\n[UNAVAILABLE - task was skipped]")

        if dep_context:
            prompt_parts.insert(0, "Context from previous tasks:\n" + "\n\n".join(dep_context) + "\n\n---\n")

        return "\n".join(prompt_parts)

    async def _handle_failure(
        self,
        dag: TaskDAG,
        task: TaskNode,
        error: str,
        failed: set[str],
        skipped: set[str],
    ) -> None:
        """Handle a task failure with retry logic."""
        task.retry_count += 1
        task.error = error

        if task.retry_count <= self.config.max_retries:
            # Exponential backoff
            wait_time = self.config.base_retry_delay * (2 ** (task.retry_count - 1))
            logger.info(f"Task {task.id} failed, retrying in {wait_time}s (attempt {task.retry_count})")
            await asyncio.sleep(wait_time)
            task.status = TaskStatus.PENDING
        else:
            # Max retries exceeded
            logger.error(f"Task {task.id} failed after {self.config.max_retries} retries: {error}")
            task.status = TaskStatus.FAILED
            failed.add(task.id)

            # Add to dead letter queue
            self.dead_letter_queue.add(task, dag.id, error)

            # Mark dependents as skipped
            self._mark_dependents_skipped(dag, task.id, skipped)

    def _mark_dependents_skipped(
        self, dag: TaskDAG, failed_task_id: str, skipped: set[str]
    ) -> None:
        """Recursively mark tasks that depend on a failed task as skipped."""
        for task in dag.tasks.values():
            if failed_task_id in task.depends_on:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
                    skipped.add(task.id)
                    logger.info(f"Task {task.id} skipped due to failed dependency {failed_task_id}")
                    # Recursively skip dependents
                    self._mark_dependents_skipped(dag, task.id, skipped)

    async def resume_from_checkpoint(self, dag_id: str) -> dict[str, Any] | None:
        """
        Resume execution from a checkpoint.

        Args:
            dag_id: The ID of the DAG to resume

        Returns:
            Execution results, or None if checkpoint not found
        """
        dag = await self.checkpoint_manager.load_checkpoint(dag_id)
        if dag is None:
            return None

        # Reset any running tasks to pending
        await self.checkpoint_manager.prepare_for_resume(dag)

        return await self.execute_dag(dag)
