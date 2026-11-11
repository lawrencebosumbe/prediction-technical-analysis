.PHONY: install test test-fast lint build

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

test-fast:
	python -m pytest -m "not slow"

lint:
	python -m ruff check src tests
	python -m mypy src

build:
	python -c "from setuptools.build_meta import build_wheel; print(build_wheel('dist'))"

