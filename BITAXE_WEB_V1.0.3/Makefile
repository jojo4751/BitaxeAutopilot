# BitAxe Web Management System - Makefile
# Provides convenient commands for development and deployment

.PHONY: help build dev prod test clean backup monitor logs

# Default target
help:
	@echo "BitAxe Web Management System - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-build    - Build and start development environment"
	@echo "  make jupyter      - Start Jupyter notebook for ML development"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Deploy production environment"
	@echo "  make staging      - Deploy staging environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-e2e     - Run end-to-end tests only"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo ""
	@echo "Management:"
	@echo "  make build        - Build Docker images"
	@echo "  make clean        - Stop and remove all containers"
	@echo "  make backup       - Create system backup"
	@echo "  make restore      - Restore from backup"
	@echo "  make monitor      - Monitor system status"
	@echo "  make logs         - View application logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate   - Run database migrations"
	@echo "  make db-seed      - Seed database with sample data"
	@echo "  make db-reset     - Reset database (WARNING: destroys data)"
	@echo ""
	@echo "Utilities:"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code"
	@echo "  make install      - Install dependencies"
	@echo "  make shell        - Open shell in application container"

# Development environment
dev:
	@echo "Starting development environment..."
	@chmod +x scripts/deploy.sh
	@./scripts/deploy.sh dev

dev-build:
	@echo "Building and starting development environment..."
	@docker-compose -f docker-compose.dev.yml build --no-cache
	@make dev

jupyter:
	@echo "Starting Jupyter notebook..."
	@docker-compose -f docker-compose.dev.yml up -d jupyter
	@echo "Jupyter available at http://localhost:8888"

# Production deployment
prod:
	@echo "Deploying production environment..."
	@chmod +x scripts/deploy.sh
	@./scripts/deploy.sh production

staging:
	@echo "Deploying staging environment..."
	@chmod +x scripts/deploy.sh
	@./scripts/deploy.sh staging

# Testing
test:
	@echo "Running all tests..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/integration/ -v

test-e2e:
	@echo "Running end-to-end tests..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/e2e/ -v

test-coverage:
	@echo "Running tests with coverage report..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# Build
build:
	@echo "Building Docker images..."
	@docker-compose build --parallel

build-no-cache:
	@echo "Building Docker images without cache..."
	@docker-compose build --no-cache --parallel

# Management
clean:
	@echo "Stopping and removing containers..."
	@docker-compose -f docker-compose.yml down --remove-orphans --volumes
	@docker-compose -f docker-compose.dev.yml down --remove-orphans --volumes
	@docker system prune -f

backup:
	@echo "Creating system backup..."
	@chmod +x scripts/backup.sh
	@./scripts/backup.sh

restore:
	@echo "Restoring from backup..."
	@echo "Please specify backup file: make restore BACKUP_FILE=path/to/backup.tar.gz"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Error: BACKUP_FILE not specified"; \
		exit 1; \
	fi
	@echo "Restoring from $(BACKUP_FILE)..."
	# Add restore logic here

monitor:
	@echo "Monitoring system status..."
	@chmod +x scripts/monitor.sh
	@./scripts/monitor.sh status

monitor-watch:
	@echo "Starting continuous monitoring..."
	@chmod +x scripts/monitor.sh
	@./scripts/monitor.sh watch

logs:
	@echo "Viewing application logs..."
	@docker-compose logs -f bitaxe-web

logs-all:
	@echo "Viewing all service logs..."
	@docker-compose logs -f

# Database operations
db-migrate:
	@echo "Running database migrations..."
	@docker-compose exec bitaxe-web python -c "from database import DatabaseManager; from models.miner_models import Base; db = DatabaseManager(); db.init_database(); Base.metadata.create_all(db.engine)"

db-seed:
	@echo "Seeding database with sample data..."
	@docker-compose exec bitaxe-web python scripts/seed_database.py

db-reset:
	@echo "WARNING: This will destroy all data. Are you sure? (y/N)"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "Resetting database..."
	@docker-compose exec postgres psql -U bitaxe -d bitaxe -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	@make db-migrate

# Code quality
lint:
	@echo "Running code linting..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev black .
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev isort .

type-check:
	@echo "Running type checks..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev mypy . --ignore-missing-imports

# Installation
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@pip install -r requirements-dev.txt
	@pip install -r requirements-test.txt

install-prod:
	@echo "Installing production dependencies..."
	@pip install -r requirements.txt
	@pip install -r requirements-prod.txt

# Utilities
shell:
	@echo "Opening shell in application container..."
	@docker-compose exec bitaxe-web bash

shell-dev:
	@echo "Opening shell in development container..."
	@docker-compose -f docker-compose.dev.yml exec bitaxe-web-dev bash

psql:
	@echo "Opening PostgreSQL shell..."
	@docker-compose exec postgres psql -U bitaxe -d bitaxe

redis-cli:
	@echo "Opening Redis CLI..."
	@docker-compose exec redis redis-cli

# Environment setup
setup-env:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please review and update .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Documentation
docs:
	@echo "Generating documentation..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev sphinx-build -b html docs docs/_build

docs-serve:
	@echo "Serving documentation..."
	@cd docs/_build && python -m http.server 8080

# Security
security-scan:
	@echo "Running security scan..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev bandit -r . -x tests/
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev safety check

# Performance
benchmark:
	@echo "Running performance benchmarks..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev python -m pytest tests/ -m performance --benchmark-only

load-test:
	@echo "Running load tests..."
	@docker-compose -f docker-compose.dev.yml exec -T bitaxe-web-dev locust -f tests/load_tests.py --headless -u 10 -r 2 -t 60s --host=http://localhost:5000

# CI/CD helpers
ci-test:
	@echo "Running CI tests..."
	@docker-compose -f docker-compose.dev.yml run --rm bitaxe-web-dev python -m pytest tests/ -v --junitxml=test-results.xml

ci-build:
	@echo "Building for CI..."
	@docker build -t bitaxe-web:latest .

# Version management
version:
	@echo "Current version:"
	@cat VERSION 2>/dev/null || echo "No VERSION file found"

tag:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified. Usage: make tag VERSION=1.0.0"; \
		exit 1; \
	fi
	@echo "$(VERSION)" > VERSION
	@git add VERSION
	@git commit -m "Bump version to $(VERSION)"
	@git tag -a "v$(VERSION)" -m "Version $(VERSION)"
	@echo "Tagged version $(VERSION)"