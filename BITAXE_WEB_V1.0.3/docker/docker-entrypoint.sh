#!/bin/bash
set -e

# BitAxe Web Management System Docker Entrypoint
# This script handles initialization and startup of the application

# Default values
DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-bitaxe}
DB_NAME=${DB_NAME:-bitaxe}
REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Wait for database to be ready
wait_for_db() {
    log "Waiting for database at ${DB_HOST}:${DB_PORT}..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
            log "Database is ready!"
            return 0
        fi
        
        warn "Database not ready, attempt $attempt/$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "Database failed to become ready after $max_attempts attempts"
    exit 1
}

# Wait for Redis to be ready
wait_for_redis() {
    log "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
    
    max_attempts=15
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            log "Redis is ready!"
            return 0
        fi
        
        warn "Redis not ready, attempt $attempt/$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "Redis failed to become ready after $max_attempts attempts"
    exit 1
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    export FLASK_APP=app.py
    export PYTHONPATH=/app:$PYTHONPATH
    
    # Run database migrations
    if [ -f "/app/migrations/alembic.ini" ]; then
        log "Running Alembic migrations..."
        cd /app && alembic upgrade head
    else
        log "No Alembic migrations found, initializing database directly..."
        cd /app && python -c "
from database import DatabaseManager
from models.miner_models import Base
import os

db_url = os.environ.get('DATABASE_URL')
if db_url and 'postgresql' in db_url:
    # Extract database path for DatabaseManager
    db_path = 'postgresql_connection'
    
db_manager = DatabaseManager()
db_manager.init_database()
Base.metadata.create_all(db_manager.engine)
print('Database initialized successfully')
"
    fi
}

# Initialize ML models directory
init_ml_models() {
    log "Initializing ML models directory..."
    
    mkdir -p /app/models
    mkdir -p /app/data
    mkdir -p /app/logs
    
    # Set permissions
    chown -R bitaxe:bitaxe /app/models /app/data /app/logs
    
    log "ML models directory initialized"
}

# Health check function
health_check() {
    log "Performing health check..."
    
    # Check if application responds
    if curl -f http://localhost:5000/health >/dev/null 2>&1; then
        log "Application health check passed"
        return 0
    else
        error "Application health check failed"
        return 1
    fi
}

# Main entrypoint logic
main() {
    log "Starting BitAxe Web Management System..."
    
    # Wait for dependencies
    wait_for_db
    wait_for_redis
    
    # Initialize components
    init_database
    init_ml_models
    
    # Set proper permissions
    chown -R bitaxe:bitaxe /app
    
    log "Initialization complete. Starting application..."
    
    # Execute the command passed to the container
    exec "$@"
}

# Handle different commands
case "$1" in
    "bash"|"sh")
        log "Starting interactive shell..."
        exec "$@"
        ;;
    "test")
        log "Running tests..."
        wait_for_db
        cd /app && python -m pytest tests/ -v
        ;;
    "migrate")
        log "Running database migrations only..."
        wait_for_db
        init_database
        ;;
    "health")
        health_check
        ;;
    *)
        main "$@"
        ;;
esac