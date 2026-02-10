"""Tests for the decomposition engine, including simplicity checking."""

from __future__ import annotations

import pytest

from src.chatbot.decomposition import DecompositionEngine
from src.chatbot.models import TaskDAG


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict] = []

    async def call_structured(
        self, prompt: str, response_format: dict, system_prompt: str | None = None
    ) -> dict:
        self.calls.append({
            "prompt": prompt,
            "response_format": response_format,
            "system_prompt": system_prompt,
        })
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {}


class TestSimplicityCheck:
    """Tests for the check_simplicity method."""

    @pytest.mark.asyncio
    async def test_simple_question_returns_true(self):
        """Simple questions should return True."""
        mock_llm = MockLLM([
            {"verdict": "SIMPLE", "reason": "Basic arithmetic", "suggested_approach": "Calculate directly"}
        ])
        engine = DecompositionEngine(mock_llm)

        result = await engine.check_simplicity("What is 1+3?")

        assert result is True
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_complex_question_returns_false(self):
        """Complex questions should return False."""
        mock_llm = MockLLM([
            {"verdict": "COMPLEX", "reason": "Multiple countries to research", "suggested_approach": "Decompose by country"}
        ])
        engine = DecompositionEngine(mock_llm)

        result = await engine.check_simplicity("Compare GDP of US, China, and Germany")

        assert result is False
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_defaults_to_simple_on_missing_verdict(self):
        """Should default to SIMPLE if verdict is missing."""
        mock_llm = MockLLM([{"reason": "No verdict provided"}])
        engine = DecompositionEngine(mock_llm)

        result = await engine.check_simplicity("Some question")

        assert result is True

    @pytest.mark.asyncio
    async def test_case_insensitive_verdict(self):
        """Verdict should be case-insensitive."""
        mock_llm = MockLLM([{"verdict": "simple", "reason": "lowercase"}])
        engine = DecompositionEngine(mock_llm)

        result = await engine.check_simplicity("Some question")

        assert result is True


class TestDecomposeQuestion:
    """Tests for the decompose_question method."""

    @pytest.mark.asyncio
    async def test_simple_question_creates_single_task(self):
        """Simple questions should create a single direct task."""
        mock_llm = MockLLM([
            {"verdict": "SIMPLE", "reason": "Basic question", "suggested_approach": "Direct"}
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("What is 1+3?")

        assert len(dag.tasks) == 1
        assert "direct" in dag.tasks
        assert dag.tasks["direct"].name == "Answer directly"
        assert dag.tasks["direct"].prompt == "What is 1+3?"
        assert mock_llm.call_count == 1  # Only simplicity check, no decomposition

    @pytest.mark.asyncio
    async def test_complex_question_decomposes(self):
        """Complex questions should be decomposed into multiple tasks."""
        mock_llm = MockLLM([
            {"verdict": "COMPLEX", "reason": "Multiple entities", "suggested_approach": "Decompose"},
            {
                "tasks": [
                    {"id": "task1", "name": "Research US GDP", "prompt": "Find US GDP"},
                    {"id": "task2", "name": "Research China GDP", "prompt": "Find China GDP"},
                ],
                "final_prompt": "Compare the GDPs"
            }
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("Compare GDP of US and China")

        assert len(dag.tasks) == 2
        assert "task1" in dag.tasks
        assert "task2" in dag.tasks
        assert mock_llm.call_count == 2  # Simplicity check + decomposition

    @pytest.mark.asyncio
    async def test_skip_simplicity_check_flag(self):
        """skip_simplicity_check should bypass the simplicity check."""
        mock_llm = MockLLM([
            {
                "tasks": [
                    {"id": "task1", "name": "Answer", "prompt": "What is 1+3?"},
                ],
                "final_prompt": "Provide the answer"
            }
        ])
        engine = DecompositionEngine(mock_llm, skip_simplicity_check=True)

        dag = await engine.decompose_question("What is 1+3?")

        # Should go straight to decomposition without simplicity check
        assert mock_llm.call_count == 1
        assert len(mock_llm.calls) == 1
        # The call should be for decomposition, not simplicity check
        assert "Decompose this question" in mock_llm.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_empty_decomposition_creates_single_task(self):
        """If decomposition returns no tasks, create a single task."""
        mock_llm = MockLLM([
            {"verdict": "COMPLEX", "reason": "Seems complex", "suggested_approach": "Decompose"},
            {"tasks": [], "final_prompt": ""}
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("Some question")

        assert len(dag.tasks) == 1
        assert "single" in dag.tasks


class TestIntegrationScenarios:
    """Integration-style tests for common scenarios."""

    @pytest.mark.asyncio
    async def test_factual_question_is_simple(self):
        """Factual questions like capital cities should be simple."""
        mock_llm = MockLLM([
            {"verdict": "SIMPLE", "reason": "Single factual lookup", "suggested_approach": "Direct answer"}
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("What is the capital of France?")

        assert len(dag.tasks) == 1
        assert dag.tasks["direct"].prompt == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_calculation_is_simple(self):
        """Mathematical calculations should be simple."""
        mock_llm = MockLLM([
            {"verdict": "SIMPLE", "reason": "Single calculation", "suggested_approach": "Use calculator"}
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("Calculate factorial of 20")

        assert len(dag.tasks) == 1

    @pytest.mark.asyncio
    async def test_multi_entity_comparison_is_complex(self):
        """Comparing multiple entities should be complex."""
        mock_llm = MockLLM([
            {"verdict": "COMPLEX", "reason": "Multiple countries to analyze", "suggested_approach": "Parallel research"},
            {
                "tasks": [
                    {"id": "us", "name": "Research US", "prompt": "Analyze US climate"},
                    {"id": "uk", "name": "Research UK", "prompt": "Analyze UK climate"},
                    {"id": "jp", "name": "Research Japan", "prompt": "Analyze Japan climate"},
                ],
                "final_prompt": "Compare climate impacts"
            }
        ])
        engine = DecompositionEngine(mock_llm)

        dag = await engine.decompose_question("Analyze climate change impacts in US, UK, and Japan")

        assert len(dag.tasks) == 3
