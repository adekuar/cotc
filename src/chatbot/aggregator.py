"""Result aggregation for completed DAG tasks."""

from __future__ import annotations

import logging
from typing import Any

from .models import TaskDAG, TaskStatus

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates results from completed tasks into a final answer."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def aggregate_results(self, dag: TaskDAG) -> str:
        """Aggregate all task results into a final answer."""
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
                return f"Unable to answer the question. {failed_count} task(s) failed during execution."
            return "No results were produced."

        # If there's only one result, return it directly
        if len(results) == 1:
            return results[0].split("\n", 1)[-1]  # Remove the header

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
            return answer
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Fallback: return raw results
            return f"Results (aggregation failed):\n\n{combined_results}"

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
