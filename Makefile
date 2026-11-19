.PHONY: smoke test lint install

smoke:
	python cli.py smoke

test:
	pytest -q

lint:
	ruff check .

install:
	pip install -e ".[dev]"
