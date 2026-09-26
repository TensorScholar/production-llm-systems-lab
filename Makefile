.PHONY: setup lint type test a1 demo-mock validate schema clean

setup:
	pip install -e ".[dev]"
	@echo "setup complete"

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src/llmlab

test:
	pytest tests/unit tests/property tests/golden -q

test-all:
	pytest -q

a1:
	python -m llmlab.cli a1 experiments/A_kv_cache/A1_calculator/manifest.yaml

demo-mock:
	python -m llmlab.cli demo-mock --out build/demo

validate:
	python -m llmlab.cli validate --results results/

schema:
	python -m llmlab.cli schema --out docs/schemas/

clean:
	rm -rf build/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
