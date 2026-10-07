"""Streamlit Web Dashboard for Agent Evaluation Framework.

Provides an interactive visual dashboard for:
1. Executive benchmark metrics and pass-rate KPIs
2. Step-by-step trajectory inspection with tool-call traces
3. Live interactive playground to run custom queries and evaluate agents in real time
"""

import json
import os
import sys
from pathlib import Path
import streamlit as st

# Ensure src/ is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
from agent_eval.models import BenchmarkCase, BenchmarkSummary, EvaluationResult
from benchmark.run_benchmark import load_dataset, run_benchmark

# Page setup
st.set_page_config(
    page_title="Agent Evaluation Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4f46e5;
    }
    .status-pass {
        color: #16a34a;
        font-weight: bold;
    }
    .status-fail {
        color: #dc2626;
        font-weight: bold;
    }
    .step-box {
        border-left: 3px solid #3b82f6;
        padding-left: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_results() -> BenchmarkSummary:
    """Load latest benchmark results from disk or run default mock benchmark."""
    json_path = Path("reports/benchmark_results.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return BenchmarkSummary(**data)
    # If not yet generated, run default benchmark
    return run_benchmark(
        dataset_path="benchmark/datasets/financial_agent_benchmark.jsonl",
        model="gpt-4o-mini",
        mode="mock",
    )


def main():
    st.title("🤖 Agent Evaluation Studio")
    st.caption("Inspect Agent Trajectories, Tool Selection Precision, and CI/CD Benchmark Results")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.selectbox(
            "Agent Execution Mode",
            options=["mock", "llm"],
            format_func=lambda x: "Simulated Agent (Zero Cost)" if x == "mock" else "Live LLM (OpenAI API)",
            help="Simulated mode runs deterministic heuristics for instant testing without API keys.",
        )
        model = st.selectbox("Agent Model", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        min_pass_rate = st.slider("Target CI Pass Rate (%)", min_value=50.0, max_value=100.0, value=80.0, step=5.0)

        st.markdown("---")
        st.subheader("Benchmark Controls")
        if st.button("🚀 Re-Run Full Benchmark", use_container_width=True):
            with st.spinner("Running agent benchmark suite across test cases..."):
                summary = run_benchmark(
                    dataset_path="benchmark/datasets/financial_agent_benchmark.jsonl",
                    model=model,
                    mode=mode,
                    min_pass_rate=min_pass_rate,
                )
                st.session_state["benchmark_summary"] = summary
                st.success("Benchmark completed successfully!")

    # Retrieve current summary
    summary: BenchmarkSummary = st.session_state.get("benchmark_summary") or get_cached_results()

    # Tabs
    tab_dash, tab_inspect, tab_playground = st.tabs([
        "📊 Benchmark Dashboard",
        "🔍 Trajectory Inspector",
        "🧪 Live Agent Playground",
    ])

    # -------------------------------------------------------------
    # TAB 1: BENCHMARK DASHBOARD
    # -------------------------------------------------------------
    with tab_dash:
        st.subheader("Executive Performance Summary")

        # Top KPI metrics row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric(
                label="Overall Pass Rate",
                value=f"{summary.pass_rate:.1f}%",
                delta="PASS" if summary.pass_rate >= min_pass_rate else "FAIL",
                delta_color="normal" if summary.pass_rate >= min_pass_rate else "inverse",
            )
        with kpi2:
            st.metric(label="Mean Score", value=f"{summary.mean_overall_score:.3f}")
        with kpi3:
            st.metric(label="Tool Selection F1", value=f"{summary.avg_tool_precision:.2f}")
        with kpi4:
            st.metric(label="Step Efficiency", value=f"{summary.avg_step_efficiency:.2f}")
        with kpi5:
            st.metric(label="Avg Latency", value=f"{summary.avg_latency_ms:.1f} ms")

        st.markdown("---")
        st.subheader("Test Case Breakdown")

        # Prepare tabular display
        table_rows = []
        for r in summary.results:
            tools = list({tc.name for s in r.trajectory.steps for tc in s.tool_calls})
            table_rows.append({
                "Case ID": r.case_id,
                "Status": "✅ PASS" if r.passed else "❌ FAIL",
                "Overall Score": f"{r.overall_score:.2f}",
                "Steps (Actual / Optimal)": f"{r.trajectory.total_steps}",
                "Latency (ms)": f"{r.trajectory.total_latency_ms:.1f}",
                "Tools Called": ", ".join(tools) or "None",
            })
        st.dataframe(table_rows, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: TRAJECTORY INSPECTOR
    # -------------------------------------------------------------
    with tab_inspect:
        st.subheader("Step-by-Step Trajectory Inspection")

        case_ids = [r.case_id for r in summary.results]
        selected_case_id = st.selectbox("Select Test Case to Inspect:", options=case_ids)

        selected_res = next(r for r in summary.results if r.case_id == selected_case_id)
        traj = selected_res.trajectory

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(f"**Task Query:** *{traj.query}*")
            st.markdown(f"**Final Answer:** `{traj.final_answer}`")

            st.markdown("#### 👣 Execution Trajectory Steps")
            for step in traj.steps:
                with st.expander(f"Step {step.step_number} — Thought & Actions ({step.step_latency_ms:.1f} ms)", expanded=True):
                    if step.thought:
                        st.info(f"🧠 **Model Thought:** {step.thought}")

                    if step.tool_calls:
                        for call in step.tool_calls:
                            st.markdown(f"🛠️ **Tool Invocation:** `{call.name}`")
                            st.caption(f"Status: {'✅ Success' if call.success else '❌ Failed'} | Latency: {call.latency_ms:.1f} ms")
                            st.json(call.arguments)
                            st.markdown(f"**Output:** `{call.output}`")
                    else:
                        st.caption("No external tools called in this step.")

        with col_right:
            st.markdown("#### 🎯 Metric Scorecard")
            score_data = []
            for m in selected_res.metric_scores:
                score_data.append({
                    "Metric": m.name,
                    "Score": f"{m.score:.2f}",
                    "Status": "✅ Pass" if m.passed else "❌ Fail",
                    "Threshold": f"{m.threshold:.2f}",
                })
            st.dataframe(score_data, use_container_width=True)

            if selected_res.judge_critique:
                st.markdown("#### ⚖️ LLM Judge Critique")
                st.success(selected_res.judge_critique)

    # -------------------------------------------------------------
    # TAB 3: LIVE AGENT PLAYGROUND
    # -------------------------------------------------------------
    with tab_playground:
        st.subheader("Interactive Agent & Evaluator Playground")
        st.write("Submit any analytical query to run the agent live and generate an instant trajectory evaluation.")

        sample_prompt = "What was Microsoft's revenue and gross profit in 2024, and what is its gross margin percentage?"
        user_query = st.text_area("User Query:", value=sample_prompt, height=80)

        if st.button("▶️ Run Agent & Evaluate", type="primary"):
            with st.spinner("Agent running multi-step trajectory..."):
                agent = FinancialAgent(model=model, mode=mode)
                live_traj = agent.run(user_query, task_id="playground_custom")

                st.markdown("### 🏁 Agent Final Response")
                st.success(live_traj.final_answer)

                st.markdown("### ⏱️ Trajectory Trace")
                for s in live_traj.steps:
                    st.write(f"**Step {s.step_number}:** {s.thought}")
                    for tc in s.tool_calls:
                        st.code(f"{tc.name}({json.dumps(tc.arguments)}) -> {tc.output}", language="python")

                # Run evaluators on custom run
                mock_case = BenchmarkCase(
                    id="playground_custom",
                    query=user_query,
                    expected_tools=["query_financial_metrics", "calculate_ratio"],
                    optimal_steps=2,
                    ground_truth_answer=live_traj.final_answer,
                )
                evaluators = [
                    SchemaAdherenceEvaluator(),
                    ToolPrecisionRecallEvaluator(),
                    StepEfficiencyEvaluator(),
                    LoopThrashingEvaluator(),
                ]

                st.markdown("### 📊 Instant Evaluation Scores")
                c1, c2, c3, c4 = st.columns(4)
                for i, ev in enumerate(evaluators):
                    res = ev.evaluate(mock_case, live_traj)
                    with [c1, c2, c3, c4][i]:
                        st.metric(label=res.name.replace("_", " ").title(), value=f"{res.score:.2f}", delta="PASS" if res.passed else "FAIL")


if __name__ == "__main__":
    main()
