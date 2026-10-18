.PHONY: install test lint demo bench data clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .

demo:
	python examples/demo.py

# Vector-only vs graph retrieval on the bundled synthetic multi-hop questions (about 40 s).
bench:
	python benchmarks/multihop.py

# Regenerate the synthetic corpus and questions (byte-identical to the committed files).
data:
	python data/make_data.py

clean:
	rm -rf .pytest_cache .ruff_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
