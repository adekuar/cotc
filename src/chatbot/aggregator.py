"""Result aggregation for completed DAG tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .models import TaskDAG, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation, which may be a final answer or a help request."""

    answer: str
    needs_user_help: bool = False
    help_request: str | None = None
    failed_tasks: list[dict[str, Any]] | None = None


class ResultAggregator:
    """Aggregates results from completed tasks into a final answer."""

    def __init__(self, llm: Any, ask_for_help_on_failure: bool = True):
        self.llm = llm
        self.ask_for_help_on_failure = ask_for_help_on_failure

    def _get_task_failures(self, dag: TaskDAG) -> list[dict[str, Any]]:
        """Get details about task failures."""
        failures = []

        # Get explicitly failed tasks
        for task in dag.tasks.values():
            if task.status == TaskStatus.FAILED:
                failures.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "prompt": task.prompt,
                    "error": task.error,
                    "type": "task_failed",
                })

        # Get tasks with critical tool failures
        for task in dag.get_tasks_with_critical_failures():
            failed_tools = [
                {
                    "tool": tc.name,
                    "error": tc.result.get("error") if tc.result else "Unknown error",
                }
                for tc in task.tool_calls
                if tc.result and not tc.result.get("success", False)
            ]
            failures.append({
                "task_id": task.id,
                "task_name": task.name,
                "prompt": task.prompt,
                "failed_tools": failed_tools,
                "type": "tool_failures",
            })

        # Get tasks skipped due to upstream critical failures
        for task in dag.tasks.values():
            if task.status == TaskStatus.SKIPPED and task.error and "upstream" in task.error.lower():
                failures.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "prompt": task.prompt,
                    "error": task.error,
                    "type": "blocked_by_failure",
                })

        return failures

    def _generate_help_request(self, dag: TaskDAG, failures: list[dict[str, Any]]) -> str:
        """Generate a help request message for the user."""
        lines = [
            "I encountered issues while trying to complete your request and need your help.",
            "",
            "## What I was trying to do:",
            f"**Question:** {dag.user_question}",
            "",
            "## Issues encountered:",
        ]

        for failure in failures:
            task_name = failure.get("task_name") or failure.get("task_id")
            lines.append(f"\n### Task: {task_name}")
            lines.append(f"**Goal:** {failure.get('prompt', 'N/A')}")

            if failure["type"] == "task_failed":
                lines.append(f"**Error:** {failure.get('error', 'Unknown error')}")
            elif failure["type"] == "tool_failures":
                lines.append("**Failed operations:**")
                for tool_failure in failure.get("failed_tools", []):
                    lines.append(f"  - {tool_failure['tool']}: {tool_failure['error']}")
            elif failure["type"] == "blocked_by_failure":
                lines.append(f"**Status:** Skipped - {failure.get('error', 'blocked by upstream failure')}")

        lines.extend([
            "",
            "## How you can help:",
            "Please provide one of the following:",
        ])

        # Generate specific help suggestions based on failure types
        help_suggestions = set()
        for failure in failures:
            if failure["type"] == "tool_failures":
                for tool_failure in failure.get("failed_tools", []):
                    if tool_failure["tool"] == "web_download":
                        help_suggestions.add("- **Provide the data directly**: If you have the required data (e.g., CSV file, stock prices), please share it")
                        help_suggestions.add("- **Provide an alternative URL**: If you know a working URL for the data, please share it")
                    elif tool_failure["tool"] == "web_search":
                        help_suggestions.add("- **Provide search results**: Share relevant information or links you'd like me to analyze")
                    elif tool_failure["tool"] == "python_execute":
                        help_suggestions.add("- **Install required packages**: The code may need packages that aren't available")

        if not help_suggestions:
            help_suggestions.add("- **Provide missing information**: Share any data or context that could help complete the task")
            help_suggestions.add("- **Modify the request**: Simplify or adjust your request to work around the issues")

        lines.extend(sorted(help_suggestions))

        return "\n".join(lines)

    async def aggregate_results(self, dag: TaskDAG, force_answer: bool = False) -> AggregationResult:
        """Aggregate all task results into a final answer or help request.

        Args:
            dag: The task DAG with execution results
            force_answer: If True, provide an answer even with failures (legacy behavior)

        Returns:
            AggregationResult with either the final answer or a help request
        """
        # Check for critical failures first
        if self.ask_for_help_on_failure and not force_answer:
            failures = self._get_task_failures(dag)
            if failures:
                help_request = self._generate_help_request(dag, failures)
                return AggregationResult(
                    answer="",
                    needs_user_help=True,
                    help_request=help_request,
                    failed_tasks=failures,
                )

        # Collect completed task results
        results = []
        for task in dag.tasks.values():
            if task.status == TaskStatus.COMPLETED and task.result:
                # Skip decomposition marker results
                if task.result.startswith("Decomposed into"):
                    continue
                results.append(f"## {task.name or task.id}\n{task.result}")

        if not results:
            failed_count = sum(1 for t in dag.tasks.values() if t.status == TaskStatus.FAILED)
            if failed_count > 0:
                return AggregationResult(
                    answer=f"Unable to answer the question. {failed_count} task(s) failed during execution.",
                    needs_user_help=False,
                )
            return AggregationResult(answer="No results were produced.", needs_user_help=False)

        # If there's only one result, return it directly
        if len(results) == 1:
            answer = results[0].split("\n", 1)[-1]  # Remove the header
            return AggregationResult(answer=answer, needs_user_help=False)

        # Combine results using LLM
        combined_results = "\n\n".join(results)

        aggregation_prompt = f"""Based on the following sub-task results, provide a comprehensive answer to the original question.

Original question: {dag.user_question}

{dag.final_prompt}

Sub-task results:
{combined_results}

Provide a well-organized, comprehensive answer that synthesizes all the information above."""

        try:
            answer = await self.llm.call(
                prompt=aggregation_prompt,
                system_prompt="You are a helpful assistant that synthesizes information from multiple sources into clear, comprehensive answers.",
            )
            return AggregationResult(answer=answer, needs_user_help=False)
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Fallback: return raw results
            return AggregationResult(
                answer=f"Results (aggregation failed):\n\n{combined_results}",
                needs_user_help=False,
            )

    def get_partial_result_summary(self, dag: TaskDAG) -> dict[str, Any]:
        """Get a summary of partial results from the DAG."""
        stats = dag.get_completion_stats()

        return {
            "is_complete": stats["failed"] == 0 and stats["skipped"] == 0,
            "completion_rate": stats["completion_rate"],
            "total_tasks": stats["total"],
            "completed_tasks": stats["completed"],
            "failed_tasks": stats["failed"],
            "skipped_tasks": stats["skipped"],
        }
