# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BITAXE Web Management System V1.0.3 - a Python Flask web application for monitoring and controlling BitAxe ASIC Bitcoin mining devices. The system provides real-time monitoring, manual control, automated benchmarking, and autopilot optimization features.

## Core Architecture

### Main Components
- **Flask Web App** (`app.py`): Main web interface with routes for status, control, benchmarks, and history
- **Data Logger** (`scripts/main.py`): Continuous data collection from miners via HTTP API
- **Autopilot System** (`scripts/bitaxe_autopilot.py`): Automated optimization with temperature safeguards and efficiency monitoring
- **Benchmark Engine** (`scripts/benchmark_runner.py`): Performance testing with different frequency/voltage combinations

### Database Structure
- SQLite database stores miner telemetry, benchmarks, events, and efficiency markers
- Tables: `logs`, `tuning_status`, `protocol`, `efficiency_markers`, `benchmark_results`
- Database path configured in `config/config.json`

### Configuration System
- Central config in `config/config.json` with miner IPs, settings, and thresholds
- Config loader (`config/config_loader.py`) provides runtime config access
- Supports profiles (max/eco), frequency/voltage lists, and temperature limits

## Running the Application

### Development Server
```bash
python app.py
```
Runs Flask in debug mode on default port 5000.

### Data Collection
```bash
python scripts/main.py
```
Starts continuous logging of miner data at configured intervals.

### Autopilot Mode
```bash
python scripts/bitaxe_autopilot.py
```
Runs automated optimization with temperature protection and efficiency monitoring.

## Key Features

### Benchmarking System
- Manual benchmarks via web interface (`/benchmark` route)
- Multi-miner benchmarks supported
- Automatic routine benchmarks in autopilot mode
- Results stored with efficiency calculations (GH/W)

### Safety Systems
- Temperature-based fallback settings (temp_limit: 73°C, temp_overheat: 75°C)
- Hashrate zero detection with automatic reboot
- Benchmark protection prevents interference during testing

### Web Interface Routes
- `/` or `/status`: Current miner status overview
- `/dashboard`: Real-time charts and graphs
- `/dashboard/<ip>`: Individual miner detailed view
- `/control`: Manual frequency/voltage control
- `/benchmarks`: Historical benchmark results
- `/events`: System event log
- `/history`: Historical data with date range selection

## Development Notes

### Miner Communication
- Uses HTTP API calls to individual miner IPs
- Standard endpoints: `/api/system` (PATCH for settings), `/api/system/restart` (POST)
- Timeout handling for network reliability

### Data Visualization
- Plotly integration for real-time charts (`utils/plot_utils.py`)
- Color-coded miners based on IP configuration
- CSV export functionality for historical data

### Thread Safety
- Benchmark operations run in separate threads
- Shared state managed via `scripts/benchmark_state.py`
- Database operations use connection-per-operation pattern

## Configuration Requirements

Update `config/config.json` with:
- Miner IP addresses in `config.ips` array
- Database path in `paths.database`
- Temperature limits in `settings.temp_limit` and `settings.temp_overheat`
- Frequency and voltage ranges in `settings.freq_list` and `settings.volt_list`