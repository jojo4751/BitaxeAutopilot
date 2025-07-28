-- Initialize BitAxe database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create database user with appropriate permissions
-- (Note: User is already created by POSTGRES_USER environment variable)

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE bitaxe TO bitaxe;

-- Set default search path
ALTER USER bitaxe SET search_path TO public;

-- Create schemas for organizing tables
CREATE SCHEMA IF NOT EXISTS mining;
CREATE SCHEMA IF NOT EXISTS ml_engine;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS auth;

-- Grant schema permissions
GRANT ALL ON SCHEMA mining TO bitaxe;
GRANT ALL ON SCHEMA ml_engine TO bitaxe;
GRANT ALL ON SCHEMA monitoring TO bitaxe;
GRANT ALL ON SCHEMA auth TO bitaxe;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA mining GRANT ALL ON TABLES TO bitaxe;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml_engine GRANT ALL ON TABLES TO bitaxe;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL ON TABLES TO bitaxe;
ALTER DEFAULT PRIVILEGES IN SCHEMA auth GRANT ALL ON TABLES TO bitaxe;

-- Initial configuration tables

-- Configuration table for system settings
CREATE TABLE IF NOT EXISTS public.system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration values
INSERT INTO public.system_config (key, value, description) VALUES 
    ('system_initialized', 'true', 'Indicates if the system has been initialized'),
    ('optimization_enabled', 'true', 'Enable ML optimization features'),
    ('monitoring_enabled', 'true', 'Enable monitoring and metrics collection'),
    ('weather_integration', 'false', 'Enable weather service integration'),
    ('max_concurrent_optimizations', '3', 'Maximum number of concurrent optimizations'),
    ('default_optimization_interval', '300', 'Default optimization interval in seconds'),
    ('temperature_alert_threshold', '85.0', 'Temperature threshold for alerts in Celsius'),
    ('efficiency_alert_threshold', '25.0', 'Minimum efficiency threshold for alerts')
ON CONFLICT (key) DO NOTHING;

-- Create index for faster config lookups
CREATE INDEX IF NOT EXISTS idx_system_config_key ON public.system_config(key);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON public.system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();