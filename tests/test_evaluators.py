"""Tests for all trajectory and deterministic evaluators."""

from agent_eval.evaluators.deterministic import (
    GroundTruthMatchEvaluator,
    LatencyBudgetEvaluator,
    SchemaAdherenceEvaluator,
)
from agent_eval.evaluators.judge import LLMJudgeEvaluator
from agent_eval.evaluators.trajectory import (
    LoopThrashingEvaluator,
    StepEfficiencyEvaluator,
    ToolPrecisionRecallEvaluator,
)
from agent_eval.models import (
    AgentTrajectory,
    BenchmarkCase,
    ToolCall,
    TrajectoryStep,
)


def sample_case():
    return BenchmarkCase(
        id="tc_sample",
        query="What was Apple's gross profit margin in 2023?",
        expected_tools=["query_financial_metrics", "calculate_ratio"],
        optimal_steps=2,
        ground_truth_answer="Apple's gross margin was 44.13%",
    )


def test_schema_adherence_evaluator():
    case = sample_case()
    traj = AgentTrajectory(
        query=case.query,
        steps=[
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCall(name="query_financial_metrics", arguments={}, output="{}", latency_ms=10.0, success=True),
                    ToolCall(name="calculate_ratio", arguments={}, output="{}", latency_ms=5.0, success=True),
                ],
            )
        ],
    )
    evaluator = SchemaAdherenceEvaluator()
    score = evaluator.evaluate(case, traj)
    assert score.score == 1.0
    assert score.passed is True


def test_tool_precision_recall_evaluator():
    case = sample_case()
    # Agent calls exactly the expected tools
    traj = AgentTrajectory(
        query=case.query,
        steps=[
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCall(name="query_financial_metrics", arguments={}, output="{}"),
                    ToolCall(name="calculate_ratio", arguments={}, output="{}"),
                ],
            )
        ],
    )
    evaluator = ToolPrecisionRecallEvaluator(threshold=0.8)
    score = evaluator.evaluate(case, traj)
    assert score.score == 1.0
    assert score.passed is True

    # Agent calls an extra unneeded tool (lowering precision)
    traj_extra = AgentTrajectory(
        query=case.query,
        steps=[
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCall(name="query_financial_metrics", arguments={}, output="{}"),
                    ToolCall(name="calculate_ratio", arguments={}, output="{}"),
                    ToolCall(name="fetch_stock_price", arguments={}, output="{}"),
                ],
            )
        ],
    )
    score_extra = evaluator.evaluate(case, traj_extra)
    assert score_extra.details["precision"] < 1.0


def test_step_efficiency_evaluator():
    case = sample_case()  # optimal_steps = 2
    # Agent takes 2 steps -> 100% efficient
    traj_opt = AgentTrajectory(query=case.query, total_steps=2)
    score_opt = StepEfficiencyEvaluator().evaluate(case, traj_opt)
    assert score_opt.score == 1.0
    assert score_opt.passed is True

    # Agent takes 4 steps -> 50% efficient
    traj_slow = AgentTrajectory(query=case.query, total_steps=4)
    score_slow = StepEfficiencyEvaluator().evaluate(case, traj_slow)
    assert score_slow.score == 0.5


def test_loop_thrashing_evaluator():
    case = sample_case()
    # Trajectory with redundant repeated calls
    traj = AgentTrajectory(
        query=case.query,
        steps=[
            TrajectoryStep(
                step_number=1,
                tool_calls=[ToolCall(name="fetch_stock_price", arguments={"ticker": "AAPL"})],
            ),
            TrajectoryStep(
                step_number=2,
                tool_calls=[ToolCall(name="fetch_stock_price", arguments={"ticker": "AAPL"})],
            ),
        ],
    )
    evaluator = LoopThrashingEvaluator()
    score = evaluator.evaluate(case, traj)
    assert score.details["loop_detected"] is True
    assert score.passed is False


def test_ground_truth_match_evaluator():
    case = sample_case()
    traj = AgentTrajectory(
        query=case.query,
        final_answer="Apple's gross margin was approximately 44.13% in fiscal 2023.",
    )
    evaluator = GroundTruthMatchEvaluator(threshold=0.7)
    score = evaluator.evaluate(case, traj)
    assert score.passed is True
    assert score.details["numeric_recall"] == 1.0


def test_latency_budget_evaluator():
    case = sample_case()
    traj_fast = AgentTrajectory(query=case.query, total_latency_ms=1200.0)
    score = LatencyBudgetEvaluator(max_allowed_ms=5000.0).evaluate(case, traj_fast)
    assert score.score == 1.0
    assert score.passed is True


def test_llm_judge_evaluator_fallback():
    case = sample_case()
    traj = AgentTrajectory(
        query=case.query,
        final_answer="Apple's gross margin was 44.13%.",
        completed=True,
    )
    evaluator = LLMJudgeEvaluator(mode="mock", threshold=0.7)
    score = evaluator.evaluate(case, traj)
    assert score.passed is True
    assert score.score >= 0.7
