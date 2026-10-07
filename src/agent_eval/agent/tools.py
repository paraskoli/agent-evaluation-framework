"""Tool definitions and mock financial data provider for the Financial Intelligence Agent."""

import json
import time
from typing import Any, Dict, Optional, Tuple

# Sample structured corporate financial data (in billions USD where applicable)
FINANCIAL_DATABASE: Dict[str, Dict[int, Dict[str, float]]] = {
    "AAPL": {
        2022: {"revenue": 394.33, "gross_profit": 170.78, "net_income": 99.80, "pe_ratio": 24.5, "rnd_spend": 26.25},
        2023: {"revenue": 383.29, "gross_profit": 169.15, "net_income": 97.00, "pe_ratio": 29.8, "rnd_spend": 29.92},
        2024: {"revenue": 391.04, "gross_profit": 180.68, "net_income": 101.50, "pe_ratio": 33.2, "rnd_spend": 31.37},
    },
    "MSFT": {
        2022: {"revenue": 198.27, "gross_profit": 135.62, "net_income": 72.74, "pe_ratio": 26.1, "rnd_spend": 24.51},
        2023: {"revenue": 211.92, "gross_profit": 146.05, "net_income": 72.36, "pe_ratio": 32.5, "rnd_spend": 27.20},
        2024: {"revenue": 245.12, "gross_profit": 170.70, "net_income": 88.14, "pe_ratio": 36.4, "rnd_spend": 29.54},
    },
    "NVDA": {
        2022: {"revenue": 26.97, "gross_profit": 15.36, "net_income": 4.37, "pe_ratio": 48.0, "rnd_spend": 7.34},
        2023: {"revenue": 26.97, "gross_profit": 15.36, "net_income": 4.37, "pe_ratio": 62.0, "rnd_spend": 7.34},
        2024: {"revenue": 60.92, "gross_profit": 44.30, "net_income": 29.76, "pe_ratio": 72.5, "rnd_spend": 8.68},
    },
    "GOOGL": {
        2022: {"revenue": 282.84, "gross_profit": 156.63, "net_income": 59.97, "pe_ratio": 19.3, "rnd_spend": 39.54},
        2023: {"revenue": 307.39, "gross_profit": 174.34, "net_income": 73.80, "pe_ratio": 25.1, "rnd_spend": 45.43},
        2024: {"revenue": 349.50, "gross_profit": 199.20, "net_income": 88.30, "pe_ratio": 24.0, "rnd_spend": 48.90},
    },
}

FILINGS_ARCHIVE: Dict[str, Dict[str, str]] = {
    "AAPL": {
        "10-K_2023": "Item 7: MD&A: Services revenue increased 9% due to expansion in advertising, App Store, and Apple Music. Gross margin was 44.13%.",
        "10-K_2024": "Item 7: MD&A: Revenue returned to growth at $391.04B. Gross margin expanded to 46.21% supported by favorable product mix and services strength.",
    },
    "NVDA": {
        "10-K_2024": "Item 7: Compute & Networking revenue grew 217% driven by demand for the NVIDIA HGX platform and Hopper architecture GPUs.",
    },
    "MSFT": {
        "10-K_2024": "Item 7: Intelligent Cloud revenue increased 19% to $105.4B driven by Azure and other cloud services growth of 30%.",
    },
}

STOCK_PRICES: Dict[str, float] = {
    "AAPL": 224.25,
    "MSFT": 418.10,
    "NVDA": 128.50,
    "GOOGL": 166.40,
    "TSLA": 248.80,
}


def query_financial_metrics(ticker: str, metric: str, year: int) -> str:
    """Retrieve verified fundamental financial metrics for a public company."""
    t = ticker.upper().strip()
    m = metric.lower().strip()
    if t not in FINANCIAL_DATABASE:
        return json.dumps({"error": f"Ticker '{t}' not found in database. Available tickers: {list(FINANCIAL_DATABASE.keys())}"})
    if year not in FINANCIAL_DATABASE[t]:
        return json.dumps({"error": f"Year {year} not available for '{t}'. Available years: {list(FINANCIAL_DATABASE[t].keys())}"})
    metrics = FINANCIAL_DATABASE[t][year]
    if m not in metrics:
        return json.dumps({
            "error": f"Metric '{m}' not found. Available metrics: {list(metrics.keys())}",
            "available_metrics": list(metrics.keys())
        })
    return json.dumps({
        "ticker": t,
        "year": year,
        "metric": m,
        "value": metrics[m],
        "unit": "USD Billions" if m in ["revenue", "gross_profit", "net_income", "rnd_spend"] else "Ratio"
    })


def calculate_ratio(numerator: float, denominator: float, calculation_type: str = "percentage") -> str:
    """Perform deterministic mathematical and financial ratio calculations."""
    if denominator == 0:
        return json.dumps({"error": "Division by zero is undefined."})
    
    val = numerator / denominator
    calc_type = calculation_type.lower()
    if calc_type in ["percentage", "margin"]:
        pct = val * 100
        return json.dumps({
            "type": calculation_type,
            "raw_ratio": round(val, 6),
            "percentage": round(pct, 2),
            "formatted": f"{round(pct, 2)}%"
        })
    elif calc_type in ["growth_rate", "yoy"]:
        # When numerator is (new - old) and denominator is old
        pct = val * 100
        return json.dumps({
            "type": calculation_type,
            "growth_rate_pct": round(pct, 2),
            "formatted": f"{round(pct, 2)}%"
        })
    else:
        return json.dumps({
            "type": calculation_type,
            "ratio": round(val, 4),
            "formatted": f"{round(val, 4)}x"
        })


def search_market_filings(ticker: str, filing_type: str = "10-K", year: Optional[int] = None) -> str:
    """Search SEC regulatory filings and disclosures for qualitative commentary."""
    t = ticker.upper().strip()
    if t not in FILINGS_ARCHIVE:
        return json.dumps({"error": f"No filings archived for '{t}'"})
    
    filings = FILINGS_ARCHIVE[t]
    matches = {}
    for key, text in filings.items():
        if filing_type.lower() in key.lower():
            if year is None or str(year) in key:
                matches[key] = text
    
    if not matches:
        return json.dumps({"status": "no_match", "message": f"No filing found matching {filing_type} for {t} in {year}"})
    return json.dumps({"ticker": t, "matches": matches})


def fetch_stock_price(ticker: str) -> str:
    """Fetch current market stock price quote for a ticker."""
    t = ticker.upper().strip()
    if t not in STOCK_PRICES:
        return json.dumps({"error": f"Ticker '{t}' price not available."})
    return json.dumps({"ticker": t, "price_usd": STOCK_PRICES[t], "currency": "USD"})


# OpenAI tool schema specifications
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_financial_metrics",
            "description": "Look up audited financial metrics (revenue, gross_profit, net_income, pe_ratio, rnd_spend) for a company and year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, MSFT, NVDA, GOOGL"},
                    "metric": {"type": "string", "description": "Metric name: revenue, gross_profit, net_income, pe_ratio, or rnd_spend"},
                    "year": {"type": "integer", "description": "Fiscal year, e.g. 2022, 2023, 2024"}
                },
                "required": ["ticker", "metric", "year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ratio",
            "description": "Calculate financial ratios, profit margins, or percentages given numerator and denominator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {"type": "number", "description": "Top term of the fraction (e.g. gross profit)"},
                    "denominator": {"type": "number", "description": "Bottom term of the fraction (e.g. revenue)"},
                    "calculation_type": {
                        "type": "string",
                        "enum": ["percentage", "margin", "growth_rate", "multiple"],
                        "description": "Format for output"
                    }
                },
                "required": ["numerator", "denominator"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_market_filings",
            "description": "Search SEC 10-K regulatory disclosures for qualitative business insights and MD&A details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL, MSFT, NVDA"},
                    "filing_type": {"type": "string", "default": "10-K", "description": "Type of SEC filing"},
                    "year": {"type": "integer", "description": "Filing year, e.g. 2023 or 2024"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_stock_price",
            "description": "Fetch current stock price quote for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, TSLA"}
                },
                "required": ["ticker"]
            }
        }
    }
]

# Dispatcher mapping
TOOL_MAP = {
    "query_financial_metrics": query_financial_metrics,
    "calculate_ratio": calculate_ratio,
    "search_market_filings": search_market_filings,
    "fetch_stock_price": fetch_stock_price,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Tuple[str, bool, Optional[str], float]:
    """Execute a tool by name with arguments, measuring execution time and status."""
    start_time = time.perf_counter()
    if name not in TOOL_MAP:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return json.dumps({"error": f"Tool '{name}' is not registered."}), False, f"Unknown tool: {name}", elapsed

    try:
        fn = TOOL_MAP[name]
        result = fn(**arguments)
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return str(result), True, None, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return json.dumps({"error": f"Exception occurred while running '{name}': {str(exc)}"}), False, str(exc), elapsed
