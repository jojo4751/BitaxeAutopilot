#!/bin/bash

# BitAxe Web Management System Deployment Script
# Usage: ./scripts/deploy.sh [environment]
# Environments: dev, staging, production

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT=${1:-production}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log "Setting up environment: $ENVIRONMENT"
    
    cd "$PROJECT_DIR"
    
    # Copy environment file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            log "Creating .env file from .env.example"
            cp .env.example .env
            warn "Please review and update .env file with your configuration"
        else
            error ".env.example file not found"
            exit 1
        fi
    fi
    
    # Set compose file based on environment
    case $ENVIRONMENT in
        "dev"|"development")
            export COMPOSE_FILE="docker-compose.dev.yml"
            export COMPOSE_PROJECT_NAME="bitaxe-dev"
            ;;
        "staging")
            export COMPOSE_FILE="docker-compose.yml"
            export COMPOSE_PROJECT_NAME="bitaxe-staging"
            ;;
        "production"|"prod")
            export COMPOSE_FILE="docker-compose.yml"
            export COMPOSE_PROJECT_NAME="bitaxe-prod"
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
    
    log "Environment setup complete"
}

# Build images
build_images() {
    log "Building Docker images..."
    
    # Build with no cache for production
    if [[ "$ENVIRONMENT" == "production" || "$ENVIRONMENT" == "prod" ]]; then
        docker-compose -f "$COMPOSE_FILE" build --no-cache --parallel
    else
        docker-compose -f "$COMPOSE_FILE" build --parallel
    fi
    
    log "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Stop existing services
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "healthy\|Up"; then
            log "Services are healthy"
            break
        fi
        
        warn "Services not yet healthy, attempt $attempt/$max_attempts"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "Services failed to become healthy"
        docker-compose -f "$COMPOSE_FILE" logs
        exit 1
    fi
    
    log "Services deployed successfully"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    docker-compose -f "$COMPOSE_FILE" exec -T bitaxe-web wait_for_db || {
        error "Database not ready"
        exit 1
    }
    
    # Run migrations
    docker-compose -f "$COMPOSE_FILE" exec -T bitaxe-web python -c "
from database import DatabaseManager
from models.miner_models import Base
db_manager = DatabaseManager()
db_manager.init_database()
Base.metadata.create_all(db_manager.engine)
print('Database migrations completed')
"
    
    log "Database migrations completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create Grafana dashboards directory if it doesn't exist
    mkdir -p "$PROJECT_DIR/docker/grafana/dashboards"
    
    # Wait for Grafana to be ready
    local max_attempts=15
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:3000/api/health &> /dev/null; then
            log "Grafana is ready"
            break
        fi
        
        warn "Grafana not ready, attempt $attempt/$max_attempts"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log "Monitoring setup completed"
}

# Perform health checks
health_checks() {
    log "Performing health checks..."
    
    local services=("bitaxe-web" "postgres" "redis")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log "$service is running"
        else
            error "$service is not running"
            return 1
        fi
    done
    
    # Test application endpoint
    if [[ "$ENVIRONMENT" == "dev" || "$ENVIRONMENT" == "development" ]]; then
        local app_url="http://localhost:5000"
    else
        local app_url="http://localhost:80"
    fi
    
    if curl -f "$app_url/health" &> /dev/null; then
        log "Application health check passed"
    else
        error "Application health check failed"
        return 1
    fi
    
    log "All health checks passed"
}

# Show deployment info
show_deployment_info() {
    log "Deployment completed successfully!"
    
    echo
    info "=== Deployment Information ==="
    info "Environment: $ENVIRONMENT"
    info "Compose File: $COMPOSE_FILE"
    info "Project Name: $COMPOSE_PROJECT_NAME"
    echo
    
    if [[ "$ENVIRONMENT" == "dev" || "$ENVIRONMENT" == "development" ]]; then
        info "Application: http://localhost:5000"
        info "Jupyter: http://localhost:8888"
        info "Database: localhost:5433"
        info "Redis: localhost:6380"
    else
        info "Application: http://localhost"
        info "Database: localhost:5432"
        info "Redis: localhost:6379"
    fi
    
    info "Grafana: http://localhost:3000"
    info "Prometheus: http://localhost:9090"
    echo
    
    info "Useful commands:"
    info "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
    info "  Stop services: docker-compose -f $COMPOSE_FILE down"
    info "  Restart services: docker-compose -f $COMPOSE_FILE restart"
    info "  View status: docker-compose -f $COMPOSE_FILE ps"
    echo
}

# Cleanup on failure
cleanup_on_failure() {
    error "Deployment failed. Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    exit 1
}

# Main deployment process
main() {
    log "Starting deployment process for environment: $ENVIRONMENT"
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    check_prerequisites
    setup_environment
    build_images
    deploy_services
    run_migrations
    
    # Only setup monitoring for non-dev environments
    if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "development" ]]; then
        setup_monitoring
    fi
    
    health_checks
    show_deployment_info
    
    log "Deployment process completed successfully!"
}

# Print usage
usage() {
    echo "Usage: $0 [environment]"
    echo "Environments:"
    echo "  dev         - Development environment with hot reload"
    echo "  staging     - Staging environment"
    echo "  production  - Production environment (default)"
    echo
    echo "Examples:"
    echo "  $0 dev      - Deploy development environment"
    echo "  $0 production - Deploy production environment"
}

# Handle command line arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main