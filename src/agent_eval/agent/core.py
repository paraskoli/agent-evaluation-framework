"""Core agent execution engine with multi-step reasoning, tool execution, and trajectory logging."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

from agent_eval.agent.tools import TOOL_DEFINITIONS, execute_tool
from agent_eval.models import AgentTrajectory, ToolCall, TrajectoryStep


SYSTEM_PROMPT = """You are a senior financial analyst AI agent.
Your objective is to answer analytical questions using verified corporate financial data and SEC disclosures.

CRITICAL OPERATIONAL RULES:
1. Always look up exact figures with `query_financial_metrics` or `search_market_filings` before making claims. Never guess numbers.
2. For margins, percentages, or ratios, always compute them using `calculate_ratio` to guarantee mathematical precision.
3. Be concise and synthesize your findings clearly in your final response.
"""


class FinancialAgent:
    """Multi-step ReAct agent with rigorous trajectory instrumentation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_steps: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: str = "auto",  # 'auto', 'llm', or 'mock'
    ):
        self.model = model
        self.max_steps = max_steps
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # In 'auto' mode, if no API key is provided, gracefully fall back to deterministic mock simulation
        if mode == "auto":
            self.mode = "llm" if (self.api_key and self.api_key != "your_openai_api_key_here") else "mock"
        else:
            self.mode = mode

        self.client: Optional[OpenAI] = None
        if self.mode == "llm":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def run(self, query: str, task_id: Optional[str] = None) -> AgentTrajectory:
        """Execute the agent on a task and record the complete trajectory."""
        start_time = time.perf_counter()

        if self.mode == "mock":
            return self._run_mock_agent(query, task_id, start_time)
        return self._run_llm_agent(query, task_id, start_time)

    def _run_llm_agent(self, query: str, task_id: Optional[str], start_time: float) -> AgentTrajectory:
        """Execute using real OpenAI tool-calling API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check your OPENAI_API_KEY.")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        steps: List[TrajectoryStep] = []
        prompt_tokens_accum = 0
        completion_tokens_accum = 0
        final_answer = ""
        completed = False
        error_msg = None

        for step_idx in range(1, self.max_steps + 1):
            step_start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as exc:
                error_msg = f"LLM API Error at step {step_idx}: {str(exc)}"
                break

            choice = response.choices[0]
            message = choice.message
            usage = response.usage
            if usage:
                prompt_tokens_accum += usage.prompt_tokens
                completion_tokens_accum += usage.completion_tokens

            tool_calls_raw = message.tool_calls or []
            thought = message.content or ""

            # If model didn't call any tools, it is ready to give final answer
            if not tool_calls_raw:
                final_answer = thought
                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                steps.append(
                    TrajectoryStep(
                        step_number=step_idx,
                        thought=thought,
                        tool_calls=[],
                        observation="Final answer provided.",
                        step_latency_ms=step_elapsed,
                    )
                )
                completed = True
                break

            # Process tool calls
            executed_calls: List[ToolCall] = []
            observations = []

            # Append assistant message with tool calls to conversation history
            messages.append(message)

            for call in tool_calls_raw:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": call.function.arguments}

                output, success, err, latency = execute_tool(name, args)
                executed_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        output=output,
                        latency_ms=latency,
                        success=success,
                        error=err,
                    )
                )
                observations.append(f"Tool {name}: {output}")

                # Append tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                })

            step_elapsed = (time.perf_counter() - step_start) * 1000.0
            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought=thought,
                    tool_calls=executed_calls,
                    observation="\n".join(observations),
                    step_latency_ms=step_elapsed,
                )
            )

        if not completed and not final_answer and steps:
            final_answer = "Max steps reached without concluding task."

        total_elapsed = (time.perf_counter() - start_time) * 1000.0
        # Estimate cost ($0.15/1M input, $0.60/1M output for gpt-4o-mini baseline)
        estimated_cost = (prompt_tokens_accum * 0.15 + completion_tokens_accum * 0.60) / 1_000_000.0

        return AgentTrajectory(
            task_id=task_id,
            query=query,
            steps=steps,
            final_answer=final_answer,
            completed=completed,
            error=error_msg,
            total_steps=len(steps),
            total_latency_ms=total_elapsed,
            prompt_tokens=prompt_tokens_accum,
            completion_tokens=completion_tokens_accum,
            estimated_cost_usd=round(estimated_cost, 6),
        )

    def _run_mock_agent(self, query: str, task_id: Optional[str], start_time: float) -> AgentTrajectory:
        """Deterministic simulation of agent execution for zero-cost testing and CI."""
        q_lower = query.lower()
        steps: List[TrajectoryStep] = []
        step_idx = 1

        # Extract company names or ticker symbols
        name_to_ticker = {
            "apple": "AAPL",
            "aapl": "AAPL",
            "microsoft": "MSFT",
            "msft": "MSFT",
            "nvidia": "NVDA",
            "nvda": "NVDA",
            "google": "GOOGL",
            "googl": "GOOGL",
            "alphabet": "GOOGL",
            "tesla": "TSLA",
            "tsla": "TSLA",
        }
        ticker = "AAPL"
        for name_key, sym in name_to_ticker.items():
            if re.search(rf"\b{name_key}\b", q_lower):
                ticker = sym
                break

        # Extract year if mentioned, else default to 2024
        year_match = re.findall(r"\b(202[2-4])\b", q_lower)
        year = int(year_match[0]) if year_match else 2024

        executed_tools: List[ToolCall] = []

        # Heuristic 1: Stock price lookup
        if "price" in q_lower or "stock quote" in q_lower or "trading at" in q_lower:
            out, ok, err, lat = execute_tool("fetch_stock_price", {"ticker": ticker})
            executed_tools.append(ToolCall(name="fetch_stock_price", arguments={"ticker": ticker}, output=out, latency_ms=lat, success=ok, error=err))
            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought=f"User asked for stock price of {ticker}. I will fetch the current price quote.",
                    tool_calls=[executed_tools[-1]],
                    observation=out,
                    step_latency_ms=lat + 5.0,
                )
            )
            step_idx += 1
            data = json.loads(out)
            final_ans = f"{ticker} is currently trading at ${data.get('price_usd', 0.0):.2f} USD."

        # Heuristic 2: Filing / Qualitative disclosure lookup
        elif "filing" in q_lower or "sec" in q_lower or "10-k" in q_lower or "md&a" in q_lower or "disclosures" in q_lower:
            out, ok, err, lat = execute_tool("search_market_filings", {"ticker": ticker, "year": year})
            executed_tools.append(ToolCall(name="search_market_filings", arguments={"ticker": ticker, "year": year}, output=out, latency_ms=lat, success=ok, error=err))
            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought=f"Query seeks qualitative disclosures from SEC filings for {ticker}.",
                    tool_calls=[executed_tools[-1]],
                    observation=out,
                    step_latency_ms=lat + 5.0,
                )
            )
            step_idx += 1
            final_ans = f"According to SEC filings for {ticker} ({year}), business highlights include: {out}."

        # Heuristic 3: Margins / Ratios (Requires query_financial_metrics AND calculate_ratio)
        elif "margin" in q_lower or "ratio" in q_lower or "percentage" in q_lower or "growth" in q_lower:
            # Step 1: Query metrics
            out1, ok1, err1, lat1 = execute_tool("query_financial_metrics", {"ticker": ticker, "metric": "gross_profit", "year": year})
            out2, ok2, err2, lat2 = execute_tool("query_financial_metrics", {"ticker": ticker, "metric": "revenue", "year": year})
            call1 = ToolCall(name="query_financial_metrics", arguments={"ticker": ticker, "metric": "gross_profit", "year": year}, output=out1, latency_ms=lat1, success=ok1, error=err1)
            call2 = ToolCall(name="query_financial_metrics", arguments={"ticker": ticker, "metric": "revenue", "year": year}, output=out2, latency_ms=lat2, success=ok2, error=err2)

            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought=f"To calculate the margin for {ticker} in {year}, I need gross profit and revenue.",
                    tool_calls=[call1, call2],
                    observation=f"{out1}\n{out2}",
                    step_latency_ms=lat1 + lat2 + 6.0,
                )
            )
            step_idx += 1

            # Step 2: Calculate ratio
            gp_data = json.loads(out1)
            rev_data = json.loads(out2)
            gp_val = gp_data.get("value", 169.15)
            rev_val = rev_data.get("value", 383.29)

            out_calc, ok_c, err_c, lat_c = execute_tool("calculate_ratio", {"numerator": gp_val, "denominator": rev_val, "calculation_type": "margin"})
            call_calc = ToolCall(name="calculate_ratio", arguments={"numerator": gp_val, "denominator": rev_val, "calculation_type": "margin"}, output=out_calc, latency_ms=lat_c, success=ok_c, error=err_c)

            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought="Now calculating the gross profit margin ratio.",
                    tool_calls=[call_calc],
                    observation=out_calc,
                    step_latency_ms=lat_c + 4.0,
                )
            )
            step_idx += 1
            calc_json = json.loads(out_calc)
            margin_pct = calc_json.get("percentage", 44.13)
            final_ans = f"In {year}, {ticker}'s revenue was ${rev_val}B and gross profit was ${gp_val}B, resulting in a gross margin of {margin_pct}%."

        # Heuristic 4: Fundamental single-metric lookup
        else:
            metric = "revenue"
            if "net income" in q_lower or "profit" in q_lower:
                metric = "net_income"
            elif "r&d" in q_lower or "rnd" in q_lower:
                metric = "rnd_spend"
            elif "pe" in q_lower or "valuation" in q_lower:
                metric = "pe_ratio"

            out, ok, err, lat = execute_tool("query_financial_metrics", {"ticker": ticker, "metric": metric, "year": year})
            tool_call = ToolCall(name="query_financial_metrics", arguments={"ticker": ticker, "metric": metric, "year": year}, output=out, latency_ms=lat, success=ok, error=err)
            steps.append(
                TrajectoryStep(
                    step_number=step_idx,
                    thought=f"Querying financial database for {ticker} {metric} in {year}.",
                    tool_calls=[tool_call],
                    observation=out,
                    step_latency_ms=lat + 5.0,
                )
            )
            step_idx += 1
            data = json.loads(out)
            val = data.get("value", "N/A")
            unit = data.get("unit", "")
            final_ans = f"For fiscal year {year}, {ticker}'s {metric} was {val} {unit}."

        total_elapsed = (time.perf_counter() - start_time) * 1000.0
        # Simulated token counts
        prompt_tokens = 320 * len(steps)
        comp_tokens = 85 * len(steps)
        cost = (prompt_tokens * 0.15 + comp_tokens * 0.60) / 1_000_000.0

        return AgentTrajectory(
            task_id=task_id,
            query=query,
            steps=steps,
            final_answer=final_ans,
            completed=True,
            error=None,
            total_steps=len(steps),
            total_latency_ms=total_elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            estimated_cost_usd=round(cost, 6),
        )
