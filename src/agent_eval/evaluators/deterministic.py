"""Deterministic and rule-based trajectory evaluators."""

import re
from typing import List
from agent_eval.evaluators.base import BaseEvaluator
from agent_eval.models import AgentTrajectory, BenchmarkCase, MetricScore


class SchemaAdherenceEvaluator(BaseEvaluator):
    """Evaluates whether all tool invocations adhere to valid schema and execute without exceptions."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(name="schema_adherence", threshold=threshold, weight=weight)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        total_calls = 0
        successful_calls = 0
        failures: List[str] = []

        for step in trajectory.steps:
            for call in step.tool_calls:
                total_calls += 1
                if call.success and call.error is None:
                    successful_calls += 1
                else:
                    failures.append(f"{call.name}: {call.error or 'Failed execution'}")

        if total_calls == 0:
            # If the task required no tools and none were called, 100% adherence
            score = 1.0 if not case.expected_tools else 0.0
        else:
            score = round(successful_calls / total_calls, 4)

        passed = score >= self.threshold
        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "total_tool_calls": total_calls,
                "successful_calls": successful_calls,
                "failures": failures,
            },
        )


class GroundTruthMatchEvaluator(BaseEvaluator):
    """Evaluates whether key numbers and essential terms from ground truth are present in final answer."""

    def __init__(self, threshold: float = 0.75, weight: float = 1.2):
        super().__init__(name="ground_truth_match", threshold=threshold, weight=weight)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        if not case.ground_truth_answer:
            return MetricScore(
                name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"message": "No ground truth specified for this case"},
            )

        gt = case.ground_truth_answer.lower()
        pred = trajectory.final_answer.lower()

        # Extract numerical tokens (robust float parsing)
        raw_gt_numbers = re.findall(r"\d+(?:\.\d+)?", gt)
        raw_pred_numbers = re.findall(r"\d+(?:\.\d+)?", pred)

        gt_floats = []
        for n in raw_gt_numbers:
            try:
                gt_floats.append(float(n))
            except ValueError:
                pass

        pred_floats = []
        for n in raw_pred_numbers:
            try:
                pred_floats.append(float(n))
            except ValueError:
                pass

        # Check numeric recall within 0.05 tolerance
        if gt_floats:
            matched_count = sum(1 for g in gt_floats if any(abs(g - p) < 0.05 for p in pred_floats))
            numeric_recall = matched_count / len(gt_floats)
        else:
            numeric_recall = 1.0

        # Extract alphanumeric significant words (> 3 chars)
        gt_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", gt))
        pred_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", pred))
        if gt_words:
            matched_words = gt_words.intersection(pred_words)
            keyword_recall = len(matched_words) / len(gt_words)
        else:
            keyword_recall = 1.0

        # Weighted score: 65% numeric accuracy, 35% keyword overlap
        score = round((0.65 * numeric_recall) + (0.35 * keyword_recall), 4)
        passed = score >= self.threshold

        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "numeric_recall": round(numeric_recall, 4),
                "keyword_recall": round(keyword_recall, 4),
                "matched_count": matched_count,
                "expected_numbers": gt_floats,
            },
        )


class LatencyBudgetEvaluator(BaseEvaluator):
    """Evaluates whether agent execution completed within acceptable latency constraints."""

    def __init__(self, max_allowed_ms: float = 15000.0, threshold: float = 1.0, weight: float = 0.5):
        super().__init__(name="latency_budget", threshold=threshold, weight=weight)
        self.max_allowed_ms = max_allowed_ms

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        latency = trajectory.total_latency_ms
        if latency <= self.max_allowed_ms:
            score = 1.0
        else:
            overage = (latency - self.max_allowed_ms) / self.max_allowed_ms
            score = max(0.0, round(1.0 - overage, 4))

        passed = score >= self.threshold
        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={
                "actual_latency_ms": round(latency, 2),
                "max_allowed_ms": self.max_allowed_ms,
            },
        )
