"""Tests for FinancialAgent tool calling and trajectory generation."""

import json
from agent_eval.agent.core import FinancialAgent
from agent_eval.agent.tools import (
    calculate_ratio,
    execute_tool,
    fetch_stock_price,
    query_financial_metrics,
)


def test_query_financial_metrics():
    res_str = query_financial_metrics("AAPL", "revenue", 2023)
    data = json.loads(res_str)
    assert data["ticker"] == "AAPL"
    assert data["year"] == 2023
    assert data["value"] == 383.29


def test_calculate_ratio():
    res_str = calculate_ratio(169.15, 383.29, "margin")
    data = json.loads(res_str)
    assert "percentage" in data
    assert data["percentage"] == 44.13


def test_fetch_stock_price():
    res_str = fetch_stock_price("AAPL")
    data = json.loads(res_str)
    assert data["ticker"] == "AAPL"
    assert data["price_usd"] == 224.25


def test_execute_tool_dispatcher():
    out, success, err, lat = execute_tool("fetch_stock_price", {"ticker": "MSFT"})
    assert success is True
    assert err is None
    assert "MSFT" in out
    assert lat >= 0.0


def test_mock_agent_trajectory():
    agent = FinancialAgent(mode="mock")
    traj = agent.run("What was Apple's gross margin in 2023?", task_id="test_01")

    assert traj.task_id == "test_01"
    assert traj.completed is True
    assert traj.total_steps >= 1
    assert "44.13%" in traj.final_answer

    # Verify tool calls were properly logged
    tools_called = [c.name for step in traj.steps for c in step.tool_calls]
    assert "query_financial_metrics" in tools_called
    assert "calculate_ratio" in tools_called
