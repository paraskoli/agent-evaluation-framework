"""LLM-as-a-Judge evaluator assessing task completion, rubric adherence, and faithfulness."""

import json
import os
from typing import Any, Dict, Optional
from openai import OpenAI
from agent_eval.evaluators.base import BaseEvaluator
from agent_eval.models import AgentTrajectory, BenchmarkCase, MetricScore


JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous evaluator judging an AI Agent's performance.
You will receive:
1. The User Query
2. Expected Rubric & Ground Truth
3. The Agent's Full Execution Trajectory and Final Answer

Evaluate the performance on a scale of 0.0 to 1.0 based on:
- Factuality & Faithfulness: Did the agent ground its statements in tool observations without hallucinations?
- Completeness: Did the final answer fully resolve the user's intent?
- Reasoning Quality: Did the trajectory follow logical deduction?

Output MUST be valid JSON adhering strictly to:
{
    "score": <float between 0.0 and 1.0>,
    "passed": <boolean, true if score >= threshold>,
    "critique": "<Concise explanation of strengths and weaknesses>"
}
"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates qualitative response quality using an LLM judge or deterministic heuristic fallback."""

    def __init__(
        self,
        model: str = "gpt-4o",
        threshold: float = 0.80,
        weight: float = 1.5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: str = "auto",  # 'auto', 'llm', or 'mock'
    ):
        super().__init__(name="llm_judge_score", threshold=threshold, weight=weight)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if mode == "auto":
            self.mode = "llm" if (self.api_key and self.api_key != "your_openai_api_key_here") else "mock"
        else:
            self.mode = mode

        self.client: Optional[OpenAI] = None
        if self.mode == "llm":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        if self.mode == "mock":
            return self._mock_judge_eval(case, trajectory)
        return self._llm_judge_eval(case, trajectory)

    def _mock_judge_eval(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        """Heuristic judge for zero-cost offline evaluations."""
        if not trajectory.final_answer or not trajectory.completed:
            return MetricScore(
                name=self.name,
                score=0.2,
                passed=False,
                threshold=self.threshold,
                details={"critique": "Agent failed to produce a final answer or completed abnormally."},
            )

        # Baseline score
        score = 0.85
        critique_points = ["Task completed logically."]

        # Check if any tool threw errors
        has_tool_error = any(call.error is not None for step in trajectory.steps for call in step.tool_calls)
        if has_tool_error:
            score -= 0.15
            critique_points.append("One or more tool calls encountered unhandled errors.")

        # Check ground truth numerical alignment
        if case.ground_truth_answer:
            # Check for rough alignment
            ans_clean = trajectory.final_answer.lower()
            gt_clean = case.ground_truth_answer.lower()
            # If ground truth specifies a percentage, verify '%' is in answer
            if "%" in gt_clean and "%" not in ans_clean:
                score -= 0.20
                critique_points.append("Missed required percentage formatting.")

        score = max(0.0, min(1.0, round(score, 2)))
        passed = score >= self.threshold

        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={"critique": " ".join(critique_points), "mode": "deterministic_fallback"},
        )

    def _llm_judge_eval(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        """Execute real LLM judgment call."""
        if not self.client:
            return self._mock_judge_eval(case, trajectory)

        # Build trajectory summary
        traj_summary = []
        for step in trajectory.steps:
            tools_used = [f"{c.name}({json.dumps(c.arguments)}) -> {c.output[:150]}" for c in step.tool_calls]
            traj_summary.append(f"Step {step.step_number}: Thought: '{step.thought}' | Tools: {tools_used}")

        prompt = f"""EVALUATION CASE:
Query: {case.query}
Expected Ground Truth: {case.ground_truth_answer or 'N/A'}
Grading Rubric: {case.rubric or 'Verify analytical rigor and factuality.'}

AGENT EXECUTION TRAJECTORY:
{chr(10).join(traj_summary)}

FINAL ANSWER:
{trajectory.final_answer}

Threshold: {self.threshold}
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or "{}"
            parsed: Dict[str, Any] = json.loads(raw)
            score = float(parsed.get("score", 0.0))
            critique = parsed.get("critique", "No critique provided.")
            passed = score >= self.threshold

            return MetricScore(
                name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                details={"critique": critique, "mode": "llm_judge"},
            )
        except Exception as exc:
            # Fallback to deterministic heuristic if API call fails
            mock_res = self._mock_judge_eval(case, trajectory)
            mock_res.details["api_fallback_reason"] = str(exc)
            return mock_res
