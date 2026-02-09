"""Task decomposition engine using LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from .models import TaskDAG, TaskNode, TaskStatus, DecompositionResult

logger = logging.getLogger(__name__)

DECOMPOSITION_SYSTEM_PROMPT = """You are a task decomposition engine. Given a user's question or task,
break it down into smaller, independent sub-tasks that can be executed in parallel where possible.

Return a JSON object with the following structure:
{
    "tasks": [
        {
            "id": "unique_short_id",
            "name": "Brief descriptive name",
            "prompt": "Detailed prompt for this sub-task",
            "depends_on": ["id_of_dependency"],
            "is_complex": false
        }
    ],
    "final_prompt": "Instructions for aggregating the sub-task results into a final answer"
}

Guidelines:
- Each task should be self-contained and focused on a single aspect
- Use depends_on to specify task dependencies (tasks that must complete first)
- Mark tasks as is_complex: true if they might benefit from further decomposition
- Keep task IDs short and descriptive (e.g., "search_1", "calc_2")
- The final_prompt should explain how to combine all results into a coherent answer
- Aim for 2-6 sub-tasks for most questions
- Tasks without dependencies can run in parallel"""

COMPLEXITY_CHECK_PROMPT = """Analyze this task and determine if it's simple enough to execute directly,
or if it should be broken down into smaller sub-tasks.

Task: {task_prompt}

Return a JSON object:
{{
    "is_simple": true/false,
    "reason": "Brief explanation",
    "sub_tasks": [
        {{
            "id": "unique_id",
            "name": "Task name",
            "prompt": "Task prompt",
            "depends_on": []
        }}
    ]
}}

If is_simple is true, sub_tasks should be an empty list.
Only decompose if the task genuinely requires multiple distinct steps."""


class DecompositionEngine:
    """Decomposes complex questions into task DAGs."""

    def __init__(self, llm: Any):
        self.llm = llm
        self._max_decomposition_depth = 2

    async def decompose_question(self, question: str) -> TaskDAG:
        """Decompose a user question into a task DAG."""
        prompt = f"Decompose this question into sub-tasks:\n\n{question}"

        response = await self.llm.call_structured(
            prompt=prompt,
            response_format={"type": "json_object"},
            system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
        )

        dag = TaskDAG(user_question=question)

        tasks_data = response.get("tasks", [])
        if not tasks_data:
            # If decomposition fails, create a single task
            task = TaskNode(
                id="single",
                name="Answer question",
                prompt=question,
            )
            dag.add_task(task)
            dag.final_prompt = "Provide the answer directly."
            return dag

        # Create task nodes
        for task_data in tasks_data:
            task = TaskNode(
                id=task_data.get("id", ""),
                name=task_data.get("name", ""),
                prompt=task_data.get("prompt", ""),
                depends_on=task_data.get("depends_on", []),
                is_complex=task_data.get("is_complex", False),
            )
            dag.add_task(task)

        dag.final_prompt = response.get("final_prompt", "Combine all results into a comprehensive answer.")

        # Validate dependencies
        valid_ids = set(dag.tasks.keys())
        for task in dag.tasks.values():
            task.depends_on = [dep for dep in task.depends_on if dep in valid_ids]

        logger.info(f"Decomposed question into {len(dag.tasks)} tasks")
        return dag

    async def maybe_decompose_task(self, dag: TaskDAG, task: TaskNode) -> bool:
        """Check if a complex task should be further decomposed.

        Returns True if the task was decomposed (new sub-tasks added to DAG).
        """
        if not task.is_complex:
            return False

        prompt = COMPLEXITY_CHECK_PROMPT.format(task_prompt=task.prompt)

        try:
            response = await self.llm.call_structured(
                prompt=prompt,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"Complexity check failed for task {task.id}: {e}")
            task.is_complex = False
            return False

        if response.get("is_simple", True):
            task.is_complex = False
            return False

        sub_tasks_data = response.get("sub_tasks", [])
        if not sub_tasks_data:
            task.is_complex = False
            return False

        # Create sub-tasks
        sub_task_ids = []
        for st_data in sub_tasks_data:
            sub_task = TaskNode(
                id=st_data.get("id", ""),
                parent_id=task.id,
                name=st_data.get("name", ""),
                prompt=st_data.get("prompt", ""),
                depends_on=st_data.get("depends_on", []),
            )
            # Inherit the original task's dependencies
            for dep_id in task.depends_on:
                if dep_id not in sub_task.depends_on:
                    sub_task.depends_on.append(dep_id)

            dag.add_task(sub_task)
            sub_task_ids.append(sub_task.id)

        # Create an aggregation task that depends on all sub-tasks
        agg_task = TaskNode(
            id=f"{task.id}_agg",
            parent_id=task.id,
            name=f"Aggregate: {task.name}",
            prompt=f"Combine the results of the sub-tasks to answer: {task.prompt}",
            depends_on=sub_task_ids,
        )
        dag.add_task(agg_task)

        # Mark the original task as completed (replaced by sub-tasks)
        task.status = TaskStatus.COMPLETED
        task.result = f"Decomposed into {len(sub_task_ids)} sub-tasks"

        # Update any tasks that depended on the original task to depend on the aggregation task
        for other_task in dag.tasks.values():
            if task.id in other_task.depends_on and other_task.id != agg_task.id:
                other_task.depends_on = [
                    agg_task.id if dep == task.id else dep
                    for dep in other_task.depends_on
                ]

        logger.info(f"Decomposed task {task.id} into {len(sub_task_ids)} sub-tasks")
        return True
