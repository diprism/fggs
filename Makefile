export PYTHON=python3
export MYPY=mypy

.PHONY: docs test

all: test typecheck

test:
	$(PYTHON) -m unittest
	test/test_sum_product.sh

docs:
	sphinx-build -M html docs/source "$@"

typecheck:
	$(MYPY) --config-file= fggs # work around bug in mypy 0.920
