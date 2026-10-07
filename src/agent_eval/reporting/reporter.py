"""Benchmark reporter: Formats evaluation results into terminal dashboards and Markdown reports."""

import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_eval.models import BenchmarkSummary


class BenchmarkReporter:
    """Renders benchmark evaluations to terminal and Markdown files."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_terminal_summary(self, summary: BenchmarkSummary) -> None:
        """Print a formatted CLI dashboard using Rich."""
        self.console.print("\n")
        color = "green" if summary.pass_rate >= 80 else "red"
        header_text = (
            f"[bold cyan]Agent Evaluation Harness[/bold cyan] | "
            f"Model: [bold yellow]{summary.model_name}[/bold yellow] | "
            f"Pass Rate: [bold {color}]{summary.pass_rate:.1f}%[/bold {color}]"
        )
        self.console.print(Panel(header_text, expand=False))

        # Main metrics summary table
        metrics_table = Table(title="Overall Benchmark Performance", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="dim", width=26)
        metrics_table.add_column("Value", justify="right")

        metrics_table.add_row("Total Test Cases", str(summary.total_cases))
        metrics_table.add_row("Passed Cases", f"[green]{summary.passed_cases}[/green]")
        metrics_table.add_row("Failed Cases", f"[red]{summary.total_cases - summary.passed_cases}[/red]")
        metrics_table.add_row("Pass Rate", f"{summary.pass_rate:.1f}%")
        metrics_table.add_row("Mean Score", f"{summary.mean_overall_score:.3f}")
        metrics_table.add_row("Avg Tool Precision", f"{summary.avg_tool_precision:.2f}")
        metrics_table.add_row("Avg Tool Recall", f"{summary.avg_tool_recall:.2f}")
        metrics_table.add_row("Avg Step Efficiency", f"{summary.avg_step_efficiency:.2f}")
        metrics_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.1f} ms")
        metrics_table.add_row("Total Estimated Cost", f"${summary.total_cost_usd:.6f}")

        self.console.print(metrics_table)

        # Case-by-case table
        case_table = Table(title="Test Case Breakdown", show_header=True, header_style="bold blue")
        case_table.add_column("Case ID", width=10)
        case_table.add_column("Status", justify="center", width=8)
        case_table.add_column("Score", justify="right", width=8)
        case_table.add_column("Steps", justify="right", width=6)
        case_table.add_column("Latency (ms)", justify="right", width=12)
        case_table.add_column("Tools Called", style="dim")

        for res in summary.results:
            tools_used = []
            for s in res.trajectory.steps:
                tools_used.extend([tc.name for tc in s.tool_calls])
            tools_str = ", ".join(list(set(tools_used))) or "none"

            status_str = "[bold green]PASS[/bold green]" if res.passed else "[bold red]FAIL[/bold red]"
            case_table.add_row(
                res.case_id,
                status_str,
                f"{res.overall_score:.2f}",
                str(res.trajectory.total_steps),
                f"{res.trajectory.total_latency_ms:.1f}",
                tools_str,
            )

        self.console.print(case_table)
        self.console.print("\n")

    def generate_markdown_report(self, summary: BenchmarkSummary, output_path: str = "benchmark_report.md") -> str:
        """Generate a GitHub-flavored Markdown report suitable for PR comments or documentation."""
        status_badge = "PASSING" if summary.pass_rate >= 80.0 else "FAILING"
        badge_color = "brightgreen" if summary.pass_rate >= 80.0 else "red"

        md_lines = [
            f"# Agent Benchmark Evaluation Report",
            f"",
            f"![Benchmark Status](https://img.shields.io/badge/Benchmark-{status_badge}-{badge_color}) ",
            f"![Pass Rate](https://img.shields.io/badge/Pass%20Rate-{summary.pass_rate:.1f}%25-blue) ",
            f"![Model](https://img.shields.io/badge/Model-{summary.model_name}-informational)",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Result | Benchmark Target |",
            f"| :--- | :--- | :--- |",
            f"| **Overall Pass Rate** | **{summary.pass_rate:.1f}%** | &ge; 80.0% |",
            f"| **Total Cases Evaluated** | `{summary.total_cases}` | - |",
            f"| **Tool Selection Precision** | `{summary.avg_tool_precision:.3f}` | &ge; 0.85 |",
            f"| **Tool Selection Recall** | `{summary.avg_tool_recall:.3f}` | &ge; 0.85 |",
            f"| **Trajectory Step Efficiency** | `{summary.avg_step_efficiency:.3f}` | &ge; 0.70 |",
            f"| **Mean Latency** | `{summary.avg_latency_ms:.1f} ms` | &le; 15000 ms |",
            f"| **Total Cost** | `${summary.total_cost_usd:.6f}` | - |",
            f"",
            f"## Detailed Test Trajectory Breakdown",
            f"",
            f"| Case ID | Status | Overall Score | Steps (Act/Opt) | Latency | Tools Invoked |",
            f"| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for res in summary.results:
            tools_used = []
            for s in res.trajectory.steps:
                tools_used.extend([tc.name for tc in s.tool_calls])
            tools_str = "`, `".join(list(set(tools_used)))
            tools_fmt = f"`{tools_str}`" if tools_str else "*None*"
            status_icon = "✅ PASS" if res.passed else "❌ FAIL"

            md_lines.append(
                f"| `{res.case_id}` | {status_icon} | `{res.overall_score:.2f}` | "
                f"`{res.trajectory.total_steps}` | `{res.trajectory.total_latency_ms:.1f} ms` | {tools_fmt} |"
            )

        md_lines.append("")
        md_lines.append("### Diagnostic Notes & Critiques")
        md_lines.append("")
        for res in summary.results:
            if not res.passed or res.judge_critique:
                md_lines.append(f"- **`{res.case_id}`**: {res.judge_critique or 'Criteria met.'}")

        content = "\n".join(md_lines)
        Path(output_path).write_text(content, encoding="utf-8")
        return content

    def save_json(self, summary: BenchmarkSummary, output_path: str = "benchmark_results.json") -> None:
        """Export raw evaluation summary to JSON format."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)
