PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: venv check fmt

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -e '.[dev]'

check:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/pytest -q

fmt:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
