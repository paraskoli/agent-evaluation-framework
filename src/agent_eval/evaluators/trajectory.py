"""Trajectory-specific metrics: Tool precision/recall, step efficiency, and loop thrashing."""

import json
from typing import List, Set
from agent_eval.evaluators.base import BaseEvaluator
from agent_eval.models import AgentTrajectory, BenchmarkCase, MetricScore


class ToolPrecisionRecallEvaluator(BaseEvaluator):
    """Evaluates whether the agent selected the correct tools (Precision, Recall, F1)."""

    def __init__(self, threshold: float = 0.80, weight: float = 1.5):
        super().__init__(name="tool_selection_f1", threshold=threshold, weight=weight)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        expected: Set[str] = set(case.expected_tools)

        # Collect unique invoked tool names
        invoked: Set[str] = set()
        for step in trajectory.steps:
            for call in step.tool_calls:
                invoked.add(call.name)

        if not expected and not invoked:
            precision, recall, f1 = 1.0, 1.0, 1.0
        elif not expected and invoked:
            # Called tools when none were required
            precision, recall, f1 = 0.0, 1.0, 0.0
        elif expected and not invoked:
            # Failed to call required tools
            precision, recall, f1 = 1.0, 0.0, 0.0
        else:
            intersection = expected.intersection(invoked)
            precision = len(intersection) / len(invoked) if invoked else 0.0
            recall = len(intersection) / len(expected) if expected else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        score = round(f1, 4)
        passed = score >= self.threshold

        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": score,
                "expected_tools": list(expected),
                "invoked_tools": list(invoked),
                "missing_tools": list(expected - invoked),
                "extra_tools": list(invoked - expected),
            },
        )


class StepEfficiencyEvaluator(BaseEvaluator):
    """Evaluates trajectory step efficiency relative to the optimal step count."""

    def __init__(self, threshold: float = 0.60, weight: float = 1.0):
        super().__init__(name="step_efficiency", threshold=threshold, weight=weight)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        optimal = max(1, case.optimal_steps)
        actual = max(1, trajectory.total_steps)

        # If actual <= optimal, efficiency is 1.0
        # If actual > optimal, ratio decreases smoothly
        if actual <= optimal:
            efficiency = 1.0
        else:
            efficiency = optimal / actual

        score = round(efficiency, 4)
        passed = score >= self.threshold

        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "optimal_steps": optimal,
                "actual_steps": actual,
                "efficiency_ratio": score,
            },
        )


class LoopThrashingEvaluator(BaseEvaluator):
    """Detects infinite loops and redundant repeated tool calls with identical arguments."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(name="loop_thrashing", threshold=threshold, weight=weight)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        seen_calls: List[str] = []
        duplicate_calls: List[str] = []

        for step in trajectory.steps:
            for call in step.tool_calls:
                call_signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                if call_signature in seen_calls:
                    duplicate_calls.append(call_signature)
                else:
                    seen_calls.append(call_signature)

        total_calls = len(seen_calls) + len(duplicate_calls)
        if total_calls == 0 or not duplicate_calls:
            score = 1.0
        else:
            # Penalize proportional to duplicate tool calls
            score = max(0.0, round(1.0 - (len(duplicate_calls) / total_calls), 4))

        passed = score >= self.threshold

        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "duplicate_count": len(duplicate_calls),
                "duplicate_signatures": duplicate_calls,
                "loop_detected": len(duplicate_calls) > 0,
            },
        )
