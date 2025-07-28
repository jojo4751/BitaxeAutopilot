# Multi-stage build for BitAxe Web Management System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r bitaxe && useradd -r -g bitaxe bitaxe

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt requirements-test.txt ./
RUN pip install -r requirements-dev.txt && \
    pip install -r requirements-test.txt

# Copy application code
COPY . .

# Change ownership to application user
RUN chown -R bitaxe:bitaxe /app

# Switch to application user
USER bitaxe

# Expose development port
EXPOSE 5000

# Development command
CMD ["python", "app.py"]

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/logs /app/data /app/models /app/static/uploads && \
    chown -R bitaxe:bitaxe /app

# Copy configuration files
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Switch to application user for application files
USER bitaxe

# Expose production ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Production command
USER root
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]