# Variables
PIPENV = pipenv
DOCKER_COMPOSE = docker-compose
IMAGE_NAME = drought_predictor
CONTAINER_NAME = drought_prediction
PORT = 5002

# Setup pipenv environment and install dependencies
.PHONY: install
install:
	@$(PIPENV) install -r requirements.txt 2>/dev/null

# Run tests
.PHONY: test
test:
	@$(PIPENV) run python -m unittest discover -s tests

# Run code linter
.PHONY: lint
lint:
	@$(PIPENV) run flake8 .

# Format code
.PHONY: format
format:
	@$(PIPENV) run black .

# Run pre-commit hooks
.PHONY: pre-commit
pre-commit:
	@$(PIPENV) run pre-commit run --all-files

# Build Docker image
.PHONY: build
build:
	@docker build -t $(IMAGE_NAME) .

# Run Docker container
.PHONY: run
run: build
	@docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Stop and remove Docker container
.PHONY: stop
stop:
	@docker stop $(CONTAINER_NAME)
	@docker rm $(CONTAINER_NAME)

# Start Docker Compose
.PHONY: compose-up
compose-up:
	@$(DOCKER_COMPOSE) up -d

# Stop Docker Compose
.PHONY: compose-down
compose-down:
	@$(DOCKER_COMPOSE) down

# Start Prefect server
.PHONY: start-prefect
start-prefect:
	@nohup $(PIPENV) run prefect server start > prefect_server.log 2>&1 &

# Train model
.PHONY: train
train:
	@$(PIPENV) run python scripts/train.py

# Run CI/CD pipeline (lint, format, test)
.PHONY: ci
ci: lint format test

# Clean up the project directory
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov
	rm -rf .hypothesis
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@docker rmi $(IMAGE_NAME)
	@docker volume prune -f

# Install and run everything
.PHONY: all
all: install pre-commit lint format compose-up start-prefect train build

# Help
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  install          Setup pipenv environment and install dependencies"
	@echo "  test             Run unit tests"
	@echo "  lint             Run code linter (flake8)"
	@echo "  format           Format code with black"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo "  build            Build Docker image"
	@echo "  run              Build and run Docker container"
	@echo "  stop             Stop and remove Docker container"
	@echo "  compose-up       Start Docker Compose"
	@echo "  compose-down     Stop Docker Compose"
	@echo "  start-prefect    Start Prefect server"
	@echo "  train            Train the model"
	@echo "  ci               Run CI/CD pipeline (lint, format, test)"
	@echo "  clean            Clean up the project directory"
	@echo "  all              Install and run everything"
