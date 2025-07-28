#!/bin/bash

# BitAxe Web Management System Monitoring Script
# Monitors system health and performance

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Check service status
check_services() {
    log "Checking service status..."
    
    local services=("bitaxe-web" "postgres" "redis" "prometheus" "grafana")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log "✓ $service is running"
        else
            error "✗ $service is not running"
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        error "Failed services: ${failed_services[*]}"
        return 1
    fi
    
    log "All services are running"
}

# Check application health
check_app_health() {
    log "Checking application health..."
    
    local health_url="http://localhost:80/health"
    
    if curl -f -s "$health_url" > /dev/null; then
        log "✓ Application health check passed"
    else
        error "✗ Application health check failed"
        return 1
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U bitaxe -d bitaxe > /dev/null; then
        log "✓ Database is accessible"
    else
        error "✗ Database is not accessible"
        return 1
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log "✓ Redis is responding"
    else
        error "✗ Redis is not responding"
        return 1
    fi
}

# Monitor resource usage
check_resources() {
    log "Checking resource usage..."
    
    # Get container stats
    local stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}")
    
    echo "$stats" | while IFS=$'\t' read -r container cpu memory mem_perc; do
        if [[ "$container" != "CONTAINER" ]]; then
            # Extract CPU percentage (remove %)
            local cpu_num=$(echo "$cpu" | sed 's/%//')
            local mem_num=$(echo "$mem_perc" | sed 's/%//')
            
            info "$container: CPU: $cpu, Memory: $memory ($mem_perc)"
            
            # Alert on high resource usage
            if (( $(echo "$cpu_num > 80" | bc -l) )); then
                warn "$container has high CPU usage: $cpu"
            fi
            
            if (( $(echo "$mem_num > 80" | bc -l) )); then
                warn "$container has high memory usage: $mem_perc"
            fi
        fi
    done
}

# Check disk space
check_disk_space() {
    log "Checking disk space..."
    
    local usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    info "Disk usage: ${usage}%"
    
    if [ "$usage" -gt 90 ]; then
        error "Disk space critically low: ${usage}%"
        return 1
    elif [ "$usage" -gt 80 ]; then
        warn "Disk space getting low: ${usage}%"
    else
        log "✓ Disk space is adequate: ${usage}%"
    fi
}

# Check log files
check_logs() {
    log "Checking for errors in logs..."
    
    local services=("bitaxe-web" "postgres" "redis")
    local error_count=0
    
    for service in "${services[@]}"; do
        local recent_errors=$(docker-compose -f "$COMPOSE_FILE" logs --tail=100 "$service" 2>/dev/null | grep -i "error\|exception\|failed" | wc -l)
        
        if [ "$recent_errors" -gt 0 ]; then
            warn "$service has $recent_errors recent errors in logs"
            error_count=$((error_count + recent_errors))
        fi
    done
    
    if [ "$error_count" -eq 0 ]; then
        log "✓ No recent errors found in logs"
    else
        warn "Total recent errors found: $error_count"
    fi
}

# Generate monitoring report
generate_report() {
    local report_file="/tmp/bitaxe_monitoring_$(date +%Y%m%d_%H%M%S).txt"
    
    log "Generating monitoring report: $report_file"
    
    {
        echo "BitAxe Web Management System - Monitoring Report"
        echo "Generated: $(date)"
        echo "=================================================="
        echo
        
        echo "Service Status:"
        docker-compose -f "$COMPOSE_FILE" ps
        echo
        
        echo "Resource Usage:"
        docker stats --no-stream
        echo
        
        echo "Disk Usage:"
        df -h
        echo
        
        echo "Recent Logs (Last 20 lines per service):"
        for service in bitaxe-web postgres redis; do
            echo "--- $service ---"
            docker-compose -f "$COMPOSE_FILE" logs --tail=20 "$service" 2>/dev/null || echo "No logs available"
            echo
        done
        
    } > "$report_file"
    
    log "Monitoring report saved: $report_file"
}

# Watch mode - continuous monitoring
watch_mode() {
    log "Starting continuous monitoring (Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "=== BitAxe Monitoring Dashboard ==="
        echo "Last updated: $(date)"
        echo
        
        check_services || true
        echo
        check_app_health || true
        echo
        check_database || true
        echo
        check_redis || true
        echo
        check_resources || true
        echo
        check_disk_space || true
        echo
        
        sleep 30
    done
}

# Main monitoring function
main() {
    case "${1:-status}" in
        "status")
            log "Running comprehensive system check..."
            check_services
            check_app_health
            check_database
            check_redis
            check_resources
            check_disk_space
            check_logs
            log "System check completed"
            ;;
        "watch")
            watch_mode
            ;;
        "report")
            generate_report
            ;;
        "services")
            check_services
            ;;
        "health")
            check_app_health
            ;;
        "resources")
            check_resources
            ;;
        "logs")
            check_logs
            ;;
        *)
            echo "Usage: $0 [command]"
            echo "Commands:"
            echo "  status      - Run comprehensive system check (default)"
            echo "  watch       - Continuous monitoring dashboard"
            echo "  report      - Generate detailed monitoring report"
            echo "  services    - Check service status only"
            echo "  health      - Check application health only"
            echo "  resources   - Check resource usage only"
            echo "  logs        - Check for errors in logs only"
            exit 1
            ;;
    esac
}

# Change to project directory
cd "$PROJECT_DIR"

# Run main function
main "$@"