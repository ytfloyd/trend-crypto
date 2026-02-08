.PHONY: test lint typecheck check

test:
	python -m pytest -q

lint:
	python -m ruff check .

typecheck:
	mypy src/ --ignore-missing-imports

check: lint typecheck test
