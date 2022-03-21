PYTHON=python3

.PHONY: docs test

test:
	$(PYTHON) -m unittest
	test/test_sum_product.sh

docs:
	sphinx-build -M html docs/source "$@"

typecheck:
	mypy --config-file= fggs # work around bug in mypy 0.920
