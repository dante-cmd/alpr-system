.PHONY: install lint test api docker eval stress clean

install:
	python -m venv .venv
	.venv/bin/pip install torch==2.12.0+cpu torchvision==0.27.0+cpu \
	    --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check alpr tests
	.venv/bin/mypy alpr

test:
	.venv/bin/pytest

api:
	.venv/bin/python -m alpr.api.main

docker:
	docker compose -f alpr/docker/docker-compose.yml up --build

eval:
	.venv/bin/python -m alpr.eval.evaluate \
		--images data/datasets/test \
		--annotations data/datasets/test_annotations.json \
		--output outputs/eval_results.json

stress:
	.venv/bin/locust -f tests/stress/locustfile.py --host http://localhost:8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov
