PYTHON=python3

.PHONY: docs test

test:
	$(PYTHON) -m unittest

docs:
	sphinx-build -M html docs/source "$@"

typecheck:
	mypy fggs
