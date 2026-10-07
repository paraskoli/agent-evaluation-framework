"""Integration test for full benchmark pipeline execution."""

from pathlib import Path
from benchmark.run_benchmark import run_benchmark


def test_full_benchmark_run(tmp_path: Path):
    md_out = str(tmp_path / "test_report.md")
    json_out = str(tmp_path / "test_results.json")
    dataset_path = "benchmark/datasets/financial_agent_benchmark.jsonl"

    summary = run_benchmark(
        dataset_path=dataset_path,
        model="mock-agent",
        mode="mock",
        min_pass_rate=75.0,
        output_md=md_out,
        output_json=json_out,
    )

    assert summary.total_cases > 0
    assert summary.pass_rate >= 75.0
    assert summary.avg_step_efficiency > 0.0
    assert Path(md_out).exists()
    assert Path(json_out).exists()
