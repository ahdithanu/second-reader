PY := .venv/bin/python

.PHONY: setup run mock test clean-results

setup:
	uv venv --python 3.13
	uv pip install -e ".[dev]"

run:
	$(PY) -m second_reader run --items 150 --tools both

mock:
	$(PY) -m second_reader run --items 150 --tools both --mock

test:
	$(PY) -m pytest -q

clean-results:
	rm -f results/calibration.csv results/metrics.md results/example_trace.json
