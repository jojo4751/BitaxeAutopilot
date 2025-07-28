from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Dict, Any

from bitaxe_logging.structured_logger import get_logger
from utils.error_handlers import api_error_boundary
from utils.rate_limiter import rate_limit
from auth.auth_service import require_auth, require_permissions, get_auth_service
from api.models import (
    APIResponse, ErrorResponse, PaginationParams, PaginatedResponse,
    MinerStatus, MinerSettings, MinerSettingsResponse, MinersListResponse,
    MinersSummary, MinersSummaryResponse,
    BenchmarkRequest, MultiBenchmarkRequest, BenchmarkStartResponse,
    BenchmarkResult, BenchmarkResultsResponse, BenchmarkStatus, BenchmarkStatusResponse,
    Event, EventsResponse, EventsQuery,
    ComponentHealth, SystemHealth, HealthResponse,
    ConfigUpdate, ConfigResponse,
    LoginRequest, TokenResponse, UserInfoResponse
)

logger = get_logger("bitaxe.api")

# Create API v1 blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')


# Authentication endpoints
@api_v1.route('/auth/login', methods=['POST'])
@api_error_boundary
@rate_limit('auth')
def login():
    """Authenticate user and return JWT token"""
    try:
        # Parse request
        data = request.get_json()
        login_req = LoginRequest(**data)
        
        logger.info("Login attempt", username=login_req.username)
        
        # Authenticate user
        auth_service = get_auth_service()
        user = auth_service.authenticate_user(login_req.username, login_req.password)
        
        if not user:
            logger.warning("Login failed: invalid credentials",
                         username=login_req.username)
            return jsonify(ErrorResponse(
                message="Invalid username or password",
                error={"code": "INVALID_CREDENTIALS"}
            ).dict()), 401
        
        # Generate token
        token_data = auth_service.generate_token(user)
        
        logger.info("Login successful", username=login_req.username)
        
        return jsonify(TokenResponse(
            success=True,
            message="Login successful",
            data=token_data
        ).dict())
        
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise


@api_v1.route('/auth/logout', methods=['POST'])
@api_error_boundary
@require_auth()
def logout():
    """Logout user and revoke token"""
    try:
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else ''
        
        if token:
            auth_service = get_auth_service()
            revoked = auth_service.revoke_token(token)
            
            logger.info("Logout successful",
                       username=request.current_user.get('username'),
                       token_revoked=revoked)
        
        return jsonify(APIResponse(
            success=True,
            message="Logout successful"
        ).dict())
        
    except Exception as e:
        logger.error("Logout error", error=str(e))
        raise


@api_v1.route('/auth/user', methods=['GET'])
@api_error_boundary
@require_auth()
def get_user_info():
    """Get current user information"""
    try:
        username = request.current_user.get('username')
        auth_service = get_auth_service()
        user = auth_service.get_user_info(username)
        
        if not user:
            return jsonify(ErrorResponse(
                message="User not found",
                error={"code": "USER_NOT_FOUND"}
            ).dict()), 404
        
        return jsonify(UserInfoResponse(
            success=True,
            data=user.to_dict()
        ).dict())
        
    except Exception as e:
        logger.error("Get user info error", error=str(e))
        raise


# Miner management endpoints
@api_v1.route('/miners', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_miners():
    """Get list of all miners with their current status"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        database_service = container.get_database_service()
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 50, type=int)
        params = PaginationParams(page=page, page_size=page_size)
        
        # Get miner data
        miners_data = database_service.get_latest_status()
        
        # Convert to response models
        miners = []
        for miner_data in miners_data:
            miner = MinerStatus(
                ip=miner_data.get('ip', ''),
                hostname=miner_data.get('hostname'),
                temperature=miner_data.get('temp'),
                hash_rate=miner_data.get('hashRate'),
                power=miner_data.get('power'),
                voltage=miner_data.get('voltage'),
                frequency=miner_data.get('frequency'),
                core_voltage=miner_data.get('coreVoltage'),
                fan_rpm=miner_data.get('fanrpm'),
                shares_accepted=miner_data.get('sharesAccepted'),
                shares_rejected=miner_data.get('sharesRejected'),
                uptime=miner_data.get('uptime'),
                version=miner_data.get('version'),
                efficiency=miner_data.get('hashRate', 0) / miner_data.get('power', 1) if miner_data.get('power', 0) > 0 else 0,
                last_seen=datetime.fromisoformat(miner_data.get('timestamp', datetime.now().isoformat()))
            )
            miners.append(miner)
        
        # Apply pagination
        start_idx = params.offset
        end_idx = start_idx + params.limit
        paginated_miners = miners[start_idx:end_idx]
        
        logger.info("Miners list retrieved",
                   total_miners=len(miners),
                   page=params.page,
                   page_size=params.page_size)
        
        return jsonify(PaginatedResponse.create(
            data=[miner.dict() for miner in paginated_miners],
            params=params,
            total_count=len(miners)
        ).dict())
        
    except Exception as e:
        logger.error("Get miners error", error=str(e))
        raise


@api_v1.route('/miners/<ip>', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_miner(ip: str):
    """Get specific miner status"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        database_service = container.get_database_service()
        
        # Get miner data
        miner_data = database_service.get_latest_status_by_ip(ip)
        
        if not miner_data:
            return jsonify(ErrorResponse(
                message=f"Miner {ip} not found or no recent data",
                error={"code": "MINER_NOT_FOUND", "ip": ip}
            ).dict()), 404
        
        miner = MinerStatus(
            ip=miner_data.get('ip', ip),
            hostname=miner_data.get('hostname'),
            temperature=miner_data.get('temp'),
            hash_rate=miner_data.get('hashRate'),
            power=miner_data.get('power'),
            voltage=miner_data.get('voltage'),
            frequency=miner_data.get('frequency'),
            core_voltage=miner_data.get('coreVoltage'),
            fan_rpm=miner_data.get('fanrpm'),
            shares_accepted=miner_data.get('sharesAccepted'),
            shares_rejected=miner_data.get('sharesRejected'),
            uptime=miner_data.get('uptime'),
            version=miner_data.get('version'),
            efficiency=miner_data.get('hashRate', 0) / miner_data.get('power', 1) if miner_data.get('power', 0) > 0 else 0,
            last_seen=datetime.fromisoformat(miner_data.get('timestamp', datetime.now().isoformat()))
        )
        
        logger.debug("Miner status retrieved", miner_ip=ip)
        
        return jsonify(APIResponse(
            success=True,
            data=miner.dict()
        ).dict())
        
    except Exception as e:
        logger.error("Get miner error", miner_ip=ip, error=str(e))
        raise


@api_v1.route('/miners/<ip>/settings', methods=['PUT'])
@api_error_boundary
@rate_limit('control')
@require_permissions('control')
def update_miner_settings(ip: str):
    """Update miner settings"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        miner_service = container.get_miner_service()
        
        # Parse request
        data = request.get_json()
        settings = MinerSettings(**data)
        
        logger.info("Miner settings update requested",
                   miner_ip=ip,
                   frequency=settings.frequency,
                   core_voltage=settings.core_voltage,
                   username=request.current_user.get('username'))
        
        # Update settings
        success = miner_service.set_miner_settings(
            ip, settings.frequency, settings.core_voltage, settings.autofanspeed
        )
        
        if not success:
            return jsonify(ErrorResponse(
                message=f"Failed to update settings for miner {ip}",
                error={"code": "SETTINGS_UPDATE_FAILED", "ip": ip}
            ).dict()), 500
        
        return jsonify(MinerSettingsResponse(
            success=True,
            message=f"Settings updated for miner {ip}",
            data={
                "ip": ip,
                "frequency": settings.frequency,
                "core_voltage": settings.core_voltage,
                "autofanspeed": settings.autofanspeed
            }
        ).dict())
        
    except Exception as e:
        logger.error("Update miner settings error", miner_ip=ip, error=str(e))
        raise


@api_v1.route('/miners/<ip>/restart', methods=['POST'])
@api_error_boundary
@rate_limit('control')
@require_permissions('control')
def restart_miner(ip: str):
    """Restart specific miner"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        miner_service = container.get_miner_service()
        
        logger.info("Miner restart requested",
                   miner_ip=ip,
                   username=request.current_user.get('username'))
        
        success = miner_service.restart_miner(ip)
        
        if not success:
            return jsonify(ErrorResponse(
                message=f"Failed to restart miner {ip}",
                error={"code": "RESTART_FAILED", "ip": ip}
            ).dict()), 500
        
        return jsonify(APIResponse(
            success=True,
            message=f"Restart command sent to miner {ip}",
            data={"ip": ip, "action": "restart"}
        ).dict())
        
    except Exception as e:
        logger.error("Restart miner error", miner_ip=ip, error=str(e))
        raise


@api_v1.route('/miners/summary', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_miners_summary():
    """Get miners summary statistics"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        miner_service = container.get_miner_service()
        
        summary_data = miner_service.get_miners_summary()
        
        summary = MinersSummary(
            total_miners=summary_data['total_miners'],
            online_miners=summary_data['online_miners'],
            offline_miners=summary_data['offline_miners'],
            total_hashrate=summary_data['total_hashrate'],
            total_power=summary_data['total_power'],
            total_efficiency=summary_data['total_efficiency'],
            average_temperature=summary_data['average_temperature'],
            timestamp=datetime.fromisoformat(summary_data['timestamp'])
        )
        
        logger.debug("Miners summary retrieved")
        
        return jsonify(MinersSummaryResponse(
            success=True,
            data=summary
        ).dict())
        
    except Exception as e:
        logger.error("Get miners summary error", error=str(e))
        raise


# Benchmark endpoints
@api_v1.route('/benchmarks', methods=['POST'])
@api_error_boundary
@require_permissions('benchmark')
def start_benchmark():
    """Start benchmark for a single miner"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        benchmark_service = container.get_benchmark_service()
        
        # Parse request
        data = request.get_json()
        benchmark_req = BenchmarkRequest(**data)
        
        logger.info("Benchmark start requested",
                   miner_ip=benchmark_req.ip,
                   frequency=benchmark_req.frequency,
                   core_voltage=benchmark_req.core_voltage,
                   duration=benchmark_req.duration,
                   username=request.current_user.get('username'))
        
        # Start benchmark
        success = benchmark_service.start_benchmark(
            benchmark_req.ip,
            benchmark_req.frequency,
            benchmark_req.core_voltage,
            benchmark_req.duration
        )
        
        if not success:
            return jsonify(ErrorResponse(
                message=f"Failed to start benchmark for miner {benchmark_req.ip}",
                error={"code": "BENCHMARK_START_FAILED", "ip": benchmark_req.ip}
            ).dict()), 500
        
        return jsonify(BenchmarkStartResponse(
            success=True,
            message=f"Benchmark started for miner {benchmark_req.ip}",
            data={
                "ip": benchmark_req.ip,
                "frequency": benchmark_req.frequency,
                "core_voltage": benchmark_req.core_voltage,
                "duration": benchmark_req.duration,
                "status": "started"
            }
        ).dict())
        
    except Exception as e:
        logger.error("Start benchmark error", error=str(e))
        raise


@api_v1.route('/benchmarks/multi', methods=['POST'])
@api_error_boundary
@require_permissions('benchmark')
def start_multi_benchmark():
    """Start benchmark for multiple miners"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        benchmark_service = container.get_benchmark_service()
        
        # Parse request
        data = request.get_json()
        benchmark_req = MultiBenchmarkRequest(**data)
        
        logger.info("Multi-benchmark start requested",
                   miner_ips=benchmark_req.ips,
                   frequency=benchmark_req.frequency,
                   core_voltage=benchmark_req.core_voltage,
                   duration=benchmark_req.duration,
                   username=request.current_user.get('username'))
        
        # Start benchmarks
        started_ips = benchmark_service.start_multi_benchmark(
            benchmark_req.ips,
            benchmark_req.frequency,
            benchmark_req.core_voltage,
            benchmark_req.duration
        )
        
        return jsonify(BenchmarkStartResponse(
            success=True,
            message=f"Multi-benchmark started for {len(started_ips)} miners",
            data={
                "requested_ips": benchmark_req.ips,
                "started_ips": started_ips,
                "frequency": benchmark_req.frequency,
                "core_voltage": benchmark_req.core_voltage,
                "duration": benchmark_req.duration,
                "status": "started"
            }
        ).dict())
        
    except Exception as e:
        logger.error("Start multi-benchmark error", error=str(e))
        raise


@api_v1.route('/benchmarks/status', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_benchmark_status():
    """Get current benchmark status"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        benchmark_service = container.get_benchmark_service()
        
        status_data = benchmark_service.get_benchmark_status()
        
        status = BenchmarkStatus(
            active_benchmarks=status_data['active_benchmarks'],
            total_active=status_data['total_active']
        )
        
        logger.debug("Benchmark status retrieved")
        
        return jsonify(BenchmarkStatusResponse(
            success=True,
            data=status
        ).dict())
        
    except Exception as e:
        logger.error("Get benchmark status error", error=str(e))
        raise


@api_v1.route('/benchmarks/results', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_benchmark_results():
    """Get benchmark results"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        database_service = container.get_database_service()
        
        # Get query parameters
        ip = request.args.get('ip')
        limit = request.args.get('limit', 50, type=int)
        
        if ip:
            results_data = database_service.get_benchmark_results_for_ip(ip, limit)
        else:
            results_data = database_service.get_benchmark_results(limit)
        
        # Convert to response models
        results = []
        for i, result_data in enumerate(results_data):
            if ip:
                # Format for IP-specific results
                result = BenchmarkResult(
                    id=i + 1,
                    ip=ip,
                    frequency=result_data[0],
                    core_voltage=result_data[1],
                    average_hashrate=result_data[2],
                    average_temperature=result_data[3],
                    efficiency_jth=result_data[4],
                    duration=600,  # Default duration
                    timestamp=datetime.fromisoformat(result_data[5])
                )
            else:
                # Format for all results
                result = BenchmarkResult(
                    id=i + 1,
                    ip=result_data[0],
                    frequency=result_data[1],
                    core_voltage=result_data[2],
                    average_hashrate=result_data[3],
                    average_temperature=result_data[4],
                    efficiency_jth=result_data[5],
                    duration=result_data[7] if len(result_data) > 7 else 600,
                    timestamp=datetime.fromisoformat(result_data[6])
                )
            results.append(result)
        
        logger.debug("Benchmark results retrieved",
                    ip_filter=ip,
                    results_count=len(results))
        
        return jsonify(BenchmarkResultsResponse(
            success=True,
            data=results
        ).dict())
        
    except Exception as e:
        logger.error("Get benchmark results error", error=str(e))
        raise


# Events endpoints
@api_v1.route('/events', methods=['GET'])
@api_error_boundary
@require_permissions('read')
def get_events():
    """Get system events"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        database_service = container.get_database_service()
        
        # Parse query parameters
        ip_filter = request.args.get('ip')
        event_type_filter = request.args.get('event_type')
        severity_filter = request.args.get('severity')
        limit = request.args.get('limit', 100, type=int)
        
        # Get events
        events_data = database_service.get_event_log(
            limit=limit,
            ip_filter=ip_filter,
            event_type_filter=event_type_filter
        )
        
        # Convert to response models
        events = []
        for i, event_data in enumerate(events_data):
            event = Event(
                id=i + 1,
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                ip=event_data['ip'],
                event_type=event_data['event_type'],
                message=event_data['message'],
                severity=event_data.get('severity', 'INFO')
            )
            
            # Apply severity filter if specified
            if severity_filter and event.severity != severity_filter:
                continue
                
            events.append(event)
        
        logger.debug("Events retrieved",
                    ip_filter=ip_filter,
                    event_type_filter=event_type_filter,
                    severity_filter=severity_filter,
                    events_count=len(events))
        
        return jsonify(EventsResponse(
            success=True,
            data=events
        ).dict())
        
    except Exception as e:
        logger.error("Get events error", error=str(e))
        raise


# Health endpoints
@api_v1.route('/health', methods=['GET'])
@api_error_boundary
def get_health():
    """Get system health status"""
    try:
        from health.health_checks import get_health_manager
        health_manager = get_health_manager()
        
        health_data = health_manager.get_overall_health()
        
        # Convert to response model
        health = SystemHealth(**health_data)
        
        logger.debug("System health retrieved")
        
        return jsonify(HealthResponse(
            success=True,
            data=health
        ).dict())
        
    except Exception as e:
        logger.error("Get health error", error=str(e))
        raise


@api_v1.route('/health/<component>', methods=['GET'])
@api_error_boundary
def get_component_health(component: str):
    """Get specific component health status"""
    try:
        from health.health_checks import get_health_manager
        health_manager = get_health_manager()
        
        health_data = health_manager.get_health_status(component)
        
        if 'details' in health_data and isinstance(health_data['details'], dict):
            # Convert to component health model
            component_health = ComponentHealth(
                component=health_data['component'],
                status=health_data['status'],
                message=health_data.get('message', ''),
                details=health_data['details'],
                timestamp=datetime.fromisoformat(health_data['timestamp']),
                duration_ms=health_data['details'].get('duration_ms', 0)
            )
            
            return jsonify(HealthResponse(
                success=True,
                data=component_health
            ).dict())
        else:
            return jsonify(ErrorResponse(
                message=f"Component {component} not found or no health data available",
                error={"code": "COMPONENT_NOT_FOUND", "component": component}
            ).dict()), 404
        
    except Exception as e:
        logger.error("Get component health error", component=component, error=str(e))
        raise


# Configuration endpoints
@api_v1.route('/config', methods=['PUT'])
@api_error_boundary
@require_permissions('admin')
def update_config():
    """Update configuration setting"""
    try:
        from services.service_container_v2 import get_container_v2
        container = get_container_v2()
        config_service = container.get_config_service()
        
        # Parse request
        data = request.get_json()
        config_update = ConfigUpdate(**data)
        
        logger.info("Configuration update requested",
                   key=config_update.key,
                   value=config_update.value,
                   username=request.current_user.get('username'))
        
        # Update configuration
        config_service.set(config_update.key, config_update.value)
        
        return jsonify(ConfigResponse(
            success=True,
            message=f"Configuration updated: {config_update.key}",
            data={
                "key": config_update.key,
                "value": config_update.value,
                "updated_at": datetime.now().isoformat()
            }
        ).dict())
        
    except Exception as e:
        logger.error("Update config error", error=str(e))
        raise


def register_api_blueprint(app):
    """Register API blueprint with Flask app"""
    app.register_blueprint(api_v1)
    
    logger.info("API v1 blueprint registered",
               prefix="/api/v1",
               endpoints=len([rule for rule in app.url_map.iter_rules() if rule.rule.startswith('/api/v1')]))