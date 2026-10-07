.PHONY: help install test bench bench-llm ui lint clean

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Agent Evaluation Framework - Available Commands:"
	@echo "  make install     Install package and development dependencies"
	@echo "  make test        Run unit and integration tests with pytest"
	@echo "  make bench       Run offline simulated benchmark suite (zero-cost, no API key needed)"
	@echo "  make bench-llm   Run live benchmark using OpenAI API (requires OPENAI_API_KEY)"
	@echo "  make ui          Launch Streamlit web dashboard for trajectory visualization"
	@echo "  make lint        Run linting checks"
	@echo "  make clean       Remove cached bytecode and benchmark temporary artifacts"

install:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -v tests/

bench:
	PYTHONPATH=src $(PYTHON) benchmark/run_benchmark.py --mode mock --min-pass-rate 80.0

bench-llm:
	PYTHONPATH=src $(PYTHON) benchmark/run_benchmark.py --mode llm --model gpt-4o-mini --min-pass-rate 80.0

ui:
	streamlit run app.py

lint:
	$(PYTHON) -m ruff check src/ tests/ benchmark/

clean:
	rm -rf __pycache__ .pytest_cache reports/ *.egg-info build dist .ruff_cache
