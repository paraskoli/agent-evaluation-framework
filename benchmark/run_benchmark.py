#!/usr/bin/env python3
"""CLI Runner for Agent Evaluation Framework.

Executes test benchmark cases against an AI agent, computes trajectory and qualitative
metrics, and produces comprehensive visual reports and CI pass/fail gates.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure src/ is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_eval.agent.core import FinancialAgent
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
    BenchmarkCase,
    BenchmarkSummary,
    EvaluationResult,
)
from agent_eval.reporting.reporter import BenchmarkReporter


def load_dataset(dataset_path: str) -> List[BenchmarkCase]:
    """Load benchmark cases from JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at: {dataset_path}")

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(BenchmarkCase(**data))
    return cases


def run_benchmark(
    dataset_path: str,
    model: str = "gpt-4o-mini",
    mode: str = "auto",
    min_pass_rate: float = 80.0,
    output_md: str = "reports/benchmark_report.md",
    output_json: str = "reports/benchmark_results.json",
) -> BenchmarkSummary:
    """Run benchmark evaluation suite."""
    reporter = BenchmarkReporter()
    cases = load_dataset(dataset_path)

    reporter.console.print(
        f"[bold blue]Starting Agent Benchmark Suite[/bold blue] | "
        f"Cases: [bold]{len(cases)}[/bold] | "
        f"Mode: [bold]{mode}[/bold] | "
        f"Target Pass Rate: [bold]{min_pass_rate}%[/bold]"
    )

    # Initialize agent
    agent = FinancialAgent(model=model, mode=mode)

    # Initialize evaluator pipeline
    evaluators = [
        SchemaAdherenceEvaluator(threshold=1.0, weight=1.0),
        ToolPrecisionRecallEvaluator(threshold=0.8, weight=1.5),
        StepEfficiencyEvaluator(threshold=0.6, weight=1.0),
        LoopThrashingEvaluator(threshold=1.0, weight=1.0),
        GroundTruthMatchEvaluator(threshold=0.7, weight=1.2),
        LatencyBudgetEvaluator(max_allowed_ms=15000.0, threshold=1.0, weight=0.5),
        LLMJudgeEvaluator(model="gpt-4o", threshold=0.75, weight=1.5, mode=mode),
    ]

    results: List[EvaluationResult] = []

    for idx, case in enumerate(cases, 1):
        reporter.console.print(f"  [{idx}/{len(cases)}] Executing case: [cyan]{case.id}[/cyan] ...", end="\r")

        # 1. Run agent on query to generate trajectory
        trajectory = agent.run(query=case.query, task_id=case.id)

        # 2. Evaluate trajectory across all evaluators
        metric_scores = []
        weighted_score_sum = 0.0
        total_weight = 0.0
        all_passed = True
        judge_critique = None

        for evaluator in evaluators:
            m_score = evaluator.evaluate(case, trajectory)
            metric_scores.append(m_score)
            weighted_score_sum += m_score.score * evaluator.weight
            total_weight += evaluator.weight

            if not m_score.passed:
                all_passed = False

            if m_score.name == "llm_judge_score":
                judge_critique = m_score.details.get("critique")

        overall_score = round(weighted_score_sum / total_weight, 4) if total_weight > 0 else 0.0

        results.append(
            EvaluationResult(
                case_id=case.id,
                passed=all_passed,
                overall_score=overall_score,
                metric_scores=metric_scores,
                trajectory=trajectory,
                judge_critique=judge_critique,
            )
        )

    # Calculate aggregate summary
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    pass_rate = round((passed_cases / total_cases) * 100.0, 2) if total_cases > 0 else 0.0
    mean_overall_score = round(sum(r.overall_score for r in results) / total_cases, 4) if total_cases > 0 else 0.0

    # Extract averages for key metrics
    def avg_for_metric(name: str) -> float:
        scores = []
        for r in results:
            for m in r.metric_scores:
                if m.name == name:
                    scores.append(m.score)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    avg_step_eff = avg_for_metric("step_efficiency")
    avg_f1 = avg_for_metric("tool_selection_f1")
    avg_latency = round(sum(r.trajectory.total_latency_ms for r in results) / total_cases, 2) if total_cases > 0 else 0.0
    total_cost = round(sum(r.trajectory.estimated_cost_usd for r in results), 6)

    # Tool precision/recall breakdown
    precisions = []
    recalls = []
    for r in results:
        for m in r.metric_scores:
            if m.name == "tool_selection_f1":
                precisions.append(m.details.get("precision", 0.0))
                recalls.append(m.details.get("recall", 0.0))
    avg_prec = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    avg_rec = round(sum(recalls) / len(recalls), 4) if recalls else 0.0

    summary = BenchmarkSummary(
        suite_name="Financial Agent Trajectory Benchmark",
        model_name=agent.model if agent.mode == "llm" else f"{agent.model} (simulated)",
        total_cases=total_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
        mean_overall_score=mean_overall_score,
        avg_step_efficiency=avg_step_eff,
        avg_tool_precision=avg_prec,
        avg_tool_recall=avg_rec,
        avg_latency_ms=avg_latency,
        total_cost_usd=total_cost,
        results=results,
    )

    # Output reports
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    reporter.print_terminal_summary(summary)
    reporter.generate_markdown_report(summary, output_path=output_md)
    reporter.save_json(summary, output_path=output_json)

    reporter.console.print(f"[bold green]✔ Markdown report saved to:[/bold green] {output_md}")
    reporter.console.print(f"[bold green]✔ JSON results exported to:[/bold green] {output_json}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Agent Evaluation Framework Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark/datasets/financial_agent_benchmark.jsonl",
        help="Path to benchmark JSONL dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model identifier to evaluate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "llm", "mock"],
        help="Execution mode ('auto', 'llm', or 'mock')",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=80.0,
        help="Minimum pass rate required to exit with code 0 (useful for CI/CD)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/benchmark_report.md",
        help="Path to save generated Markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/benchmark_results.json",
        help="Path to save detailed JSON output",
    )

    args = parser.parse_args()

    summary = run_benchmark(
        dataset_path=args.dataset,
        model=args.model,
        mode=args.mode,
        min_pass_rate=args.min_pass_rate,
        output_md=args.output_report,
        output_json=args.output_json,
    )

    # Exit code gate for CI/CD integration
    if summary.pass_rate < args.min_pass_rate:
        print(
            f"❌ BENCHMARK FAILED: Pass rate {summary.pass_rate:.1f}% below required {args.min_pass_rate:.1f}% threshold."
        )
        sys.exit(1)
    else:
        print(f"✅ BENCHMARK PASSED: Pass rate {summary.pass_rate:.1f}% meets threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
