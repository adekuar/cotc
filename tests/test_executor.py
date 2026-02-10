"""Tests for the DAG executor, including critical failure handling."""

from __future__ import annotations

import pytest

from src.chatbot.models import TaskDAG, TaskNode, TaskStatus, ToolCallRecord


class TestCriticalFailureHandling:
    """Tests for critical failure detection and downstream task blocking."""

    def test_has_tool_failures_flag_default_false(self):
        """has_tool_failures should default to False."""
        task = TaskNode(id="t1", prompt="Test task")
        assert task.has_tool_failures is False

    def test_has_critical_tool_failures_with_all_failures(self):
        """Task should detect critical failures when all tools fail."""
        task = TaskNode(id="t1", prompt="Test task")
        task.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={"url": "http://example.com"},
                result={"success": False, "error": "HTTP 401"},
            ),
            ToolCallRecord(
                id="tc2",
                name="web_download",
                arguments={"url": "http://example2.com"},
                result={"success": False, "error": "HTTP 503"},
            ),
        ]
        assert task.has_critical_tool_failures() is True

    def test_has_critical_tool_failures_with_majority_failures(self):
        """Task should detect critical failures when >50% tools fail."""
        task = TaskNode(id="t1", prompt="Test task")
        task.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={"url": "http://example.com"},
                result={"success": False, "error": "HTTP 401"},
            ),
            ToolCallRecord(
                id="tc2",
                name="web_download",
                arguments={"url": "http://example2.com"},
                result={"success": False, "error": "HTTP 503"},
            ),
            ToolCallRecord(
                id="tc3",
                name="web_search",
                arguments={"query": "test"},
                result={"success": True, "output": "Results"},
            ),
        ]
        # 2/3 failed = 66% > 50%
        assert task.has_critical_tool_failures() is True

    def test_has_critical_tool_failures_with_minority_failures(self):
        """Task should NOT detect critical failures when <50% tools fail."""
        task = TaskNode(id="t1", prompt="Test task")
        task.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={"url": "http://example.com"},
                result={"success": False, "error": "HTTP 401"},
            ),
            ToolCallRecord(
                id="tc2",
                name="web_search",
                arguments={"query": "test1"},
                result={"success": True, "output": "Results 1"},
            ),
            ToolCallRecord(
                id="tc3",
                name="web_search",
                arguments={"query": "test2"},
                result={"success": True, "output": "Results 2"},
            ),
        ]
        # 1/3 failed = 33% < 50%
        assert task.has_critical_tool_failures() is False


class TestGetReadyTasksWithFailures:
    """Tests for get_ready_tasks blocking on critical failures."""

    def test_downstream_task_not_ready_when_upstream_has_failures(self):
        """Downstream tasks should NOT be ready when upstream has critical failures."""
        dag = TaskDAG(user_question="test")

        # Task 1: completed but with critical tool failures
        task1 = TaskNode(id="t1", name="Download Data", prompt="Download stock data")
        task1.status = TaskStatus.COMPLETED
        task1.has_tool_failures = True
        task1.result = "I attempted to download but encountered issues..."

        # Task 2: depends on task 1
        task2 = TaskNode(
            id="t2",
            name="Train Model",
            prompt="Train model on downloaded data",
            depends_on=["t1"],
        )

        dag.add_task(task1)
        dag.add_task(task2)

        ready = dag.get_ready_tasks()

        # Task 2 should NOT be in ready list because task 1 has critical failures
        assert task2 not in ready
        assert len(ready) == 0

    def test_downstream_task_ready_when_upstream_succeeds(self):
        """Downstream tasks should be ready when upstream succeeds without failures."""
        dag = TaskDAG(user_question="test")

        # Task 1: completed successfully
        task1 = TaskNode(id="t1", name="Download Data", prompt="Download stock data")
        task1.status = TaskStatus.COMPLETED
        task1.has_tool_failures = False
        task1.result = "Downloaded successfully"

        # Task 2: depends on task 1
        task2 = TaskNode(
            id="t2",
            name="Train Model",
            prompt="Train model on downloaded data",
            depends_on=["t1"],
        )

        dag.add_task(task1)
        dag.add_task(task2)

        ready = dag.get_ready_tasks()

        # Task 2 should be ready
        assert task2 in ready

    def test_parallel_task_ready_when_sibling_has_failures(self):
        """Parallel tasks should still be ready even if siblings have failures."""
        dag = TaskDAG(user_question="test")

        # Task 1: completed but with critical failures
        task1 = TaskNode(id="t1", name="Task 1", prompt="Do something")
        task1.status = TaskStatus.COMPLETED
        task1.has_tool_failures = True

        # Task 2: independent parallel task (no dependency on task 1)
        task2 = TaskNode(id="t2", name="Task 2", prompt="Do something else")

        dag.add_task(task1)
        dag.add_task(task2)

        ready = dag.get_ready_tasks()

        # Task 2 should be ready since it doesn't depend on task 1
        assert task2 in ready

    def test_chain_of_dependencies_blocked(self):
        """Entire chain should be blocked when first task has failures."""
        dag = TaskDAG(user_question="test")

        # Task 1: completed with failures
        task1 = TaskNode(id="t1", name="Task 1", prompt="First task")
        task1.status = TaskStatus.COMPLETED
        task1.has_tool_failures = True

        # Task 2: depends on task 1
        task2 = TaskNode(id="t2", name="Task 2", prompt="Second task", depends_on=["t1"])

        # Task 3: depends on task 2
        task3 = TaskNode(id="t3", name="Task 3", prompt="Third task", depends_on=["t2"])

        dag.add_task(task1)
        dag.add_task(task2)
        dag.add_task(task3)

        ready = dag.get_ready_tasks()

        # Neither task 2 nor task 3 should be ready
        assert task2 not in ready
        assert task3 not in ready
        assert len(ready) == 0


class TestTasksWithCriticalFailures:
    """Tests for get_tasks_with_critical_failures method."""

    def test_returns_completed_tasks_with_failures(self):
        """Should return completed tasks that have critical failures."""
        dag = TaskDAG(user_question="test")

        task1 = TaskNode(id="t1", name="Task 1", prompt="First task")
        task1.status = TaskStatus.COMPLETED
        task1.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={},
                result={"success": False, "error": "Failed"},
            ),
        ]

        task2 = TaskNode(id="t2", name="Task 2", prompt="Second task")
        task2.status = TaskStatus.COMPLETED
        # No tool calls, so no failures

        dag.add_task(task1)
        dag.add_task(task2)

        failures = dag.get_tasks_with_critical_failures()

        assert task1 in failures
        assert task2 not in failures

    def test_does_not_return_pending_tasks(self):
        """Should not return pending tasks even if they have tool failures."""
        dag = TaskDAG(user_question="test")

        task = TaskNode(id="t1", name="Task 1", prompt="First task")
        task.status = TaskStatus.PENDING
        task.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={},
                result={"success": False, "error": "Failed"},
            ),
        ]

        dag.add_task(task)

        failures = dag.get_tasks_with_critical_failures()

        assert len(failures) == 0


class TestNeedsUserHelp:
    """Tests for needs_user_help method."""

    def test_returns_true_for_failed_tasks(self):
        """Should return True when there are failed tasks."""
        dag = TaskDAG(user_question="test")

        task = TaskNode(id="t1", name="Task 1", prompt="First task")
        task.status = TaskStatus.FAILED
        task.error = "Task failed"

        dag.add_task(task)

        assert dag.needs_user_help() is True

    def test_returns_true_for_critical_tool_failures(self):
        """Should return True when there are critical tool failures."""
        dag = TaskDAG(user_question="test")

        task = TaskNode(id="t1", name="Task 1", prompt="First task")
        task.status = TaskStatus.COMPLETED
        task.tool_calls = [
            ToolCallRecord(
                id="tc1",
                name="web_download",
                arguments={},
                result={"success": False, "error": "Failed"},
            ),
        ]

        dag.add_task(task)

        assert dag.needs_user_help() is True

    def test_returns_false_when_all_successful(self):
        """Should return False when all tasks complete successfully."""
        dag = TaskDAG(user_question="test")

        task = TaskNode(id="t1", name="Task 1", prompt="First task")
        task.status = TaskStatus.COMPLETED
        task.result = "Success"

        dag.add_task(task)

        assert dag.needs_user_help() is False
