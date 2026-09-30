UV ?= uv
.PHONY: install up down demo legacy-demo test check schemas
install:
	$(UV) sync --python 3.12
up:
	docker compose up -d --build --wait
down:
	docker compose down
demo: up
	docker compose exec -T scientific-harness-api python examples/toy_experiment/run.py
legacy-demo: up
	docker compose exec -T scientific-harness-api python examples/run_demo.py
test:
	$(UV) run pytest -q
check:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
schemas:
	$(UV) run python examples/generate_schemas.py
