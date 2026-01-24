.PHONY: lint format typecheck test

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check . --fix
	black .

typecheck:
	mypy .

test:
	pytest
