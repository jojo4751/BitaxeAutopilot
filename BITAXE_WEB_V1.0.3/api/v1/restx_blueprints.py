"""
Flask-RESTX enhanced API endpoints with OpenAPI documentation
"""

from flask import request
from flask_restx import Namespace, Resource, fields
from datetime import datetime
from typing import Dict, Any

from bitaxe_logging.structured_logger import get_logger
from utils.error_handlers import api_error_boundary
from auth.auth_service import require_auth, require_permissions, get_auth_service
from api.openapi_spec import (
    api, error_model, miner_status_model, miner_settings_model, miners_summary_model,
    benchmark_request_model, multi_benchmark_request_model, benchmark_result_model,
    benchmark_status_model, event_model, component_health_model, system_health_model,
    login_request_model, token_response_model, user_info_model, config_update_model,
    paginated_miners_response, miners_summary_response, benchmark_results_response,
    events_response, health_response, token_response_wrapper, user_info_response
)

logger = get_logger("bitaxe.api.restx")

# Create namespaces for organizing endpoints
auth_ns = Namespace('auth', description='Authentication operations')
miners_ns = Namespace('miners', description='Miner management operations')
benchmarks_ns = Namespace('benchmarks', description='Benchmark operations')
events_ns = Namespace('events', description='Event logging operations')
health_ns = Namespace('health', description='System health monitoring')
config_ns = Namespace('config', description='Configuration management')

# Add namespaces to API
api.add_namespace(auth_ns, path='/auth')
api.add_namespace(miners_ns, path='/miners')
api.add_namespace(benchmarks_ns, path='/benchmarks')
api.add_namespace(events_ns, path='/events')
api.add_namespace(health_ns, path='/health')
api.add_namespace(config_ns, path='/config')

# Authentication endpoints
@auth_ns.route('/login')
class AuthLogin(Resource):
    @auth_ns.doc('authenticate_user')
    @auth_ns.expect(login_request_model, validate=True)
    @auth_ns.response(200, 'Success', token_response_wrapper)
    @auth_ns.response(401, 'Invalid credentials', error_model)
    @auth_ns.response(400, 'Bad request', error_model)
    def post(self):
        """Authenticate user and return JWT token"""
        try:
            # Parse request
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return {'success': False, 'error': {'code': 'MISSING_CREDENTIALS', 'message': 'Username and password required'}}, 400
            
            logger.info("Login attempt", username=username)
            
            # Authenticate user
            auth_service = get_auth_service()
            user = auth_service.authenticate_user(username, password)
            
            if not user:
                logger.warning("Login failed: invalid credentials", username=username)
                return {'success': False, 'error': {'code': 'INVALID_CREDENTIALS', 'message': 'Invalid username or password'}}, 401
            
            # Generate token
            token_data = auth_service.generate_token(user)
            
            logger.info("Login successful", username=username)
            
            return {
                'success': True,
                'message': 'Login successful',
                'data': token_data
            }
            
        except Exception as e:
            logger.error("Login error", error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@auth_ns.route('/logout')
class AuthLogout(Resource):
    @auth_ns.doc('logout_user')
    @auth_ns.doc(security='Bearer')
    @auth_ns.response(200, 'Success', fields.Raw({'success': fields.Boolean, 'message': fields.String}))
    @auth_ns.response(401, 'Unauthorized', error_model)
    @require_auth()
    def post(self):
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
            
            return {'success': True, 'message': 'Logout successful'}
            
        except Exception as e:
            logger.error("Logout error", error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@auth_ns.route('/user')
class AuthUser(Resource):
    @auth_ns.doc('get_user_info')
    @auth_ns.doc(security='Bearer')
    @auth_ns.response(200, 'Success', user_info_response)
    @auth_ns.response(401, 'Unauthorized', error_model)
    @auth_ns.response(404, 'User not found', error_model)
    @require_auth()
    def get(self):
        """Get current user information"""
        try:
            username = request.current_user.get('username')
            auth_service = get_auth_service()
            user = auth_service.get_user_info(username)
            
            if not user:
                return {'success': False, 'error': {'code': 'USER_NOT_FOUND', 'message': 'User not found'}}, 404
            
            return {'success': True, 'data': user.to_dict()}
            
        except Exception as e:
            logger.error("Get user info error", error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


# Miner management endpoints
@miners_ns.route('')
class MinersList(Resource):
    @miners_ns.doc('get_miners')
    @miners_ns.doc(security='Bearer')
    @miners_ns.param('page', 'Page number (1-based)', type='integer', default=1)
    @miners_ns.param('page_size', 'Items per page', type='integer', default=50)
    @miners_ns.response(200, 'Success', paginated_miners_response)
    @miners_ns.response(401, 'Unauthorized', error_model)
    @require_permissions('read')
    def get(self):
        """Get list of all miners with their current status"""
        try:
            from services.service_container_v2 import get_container_v2
            container = get_container_v2()
            database_service = container.get_database_service()
            
            # Get pagination parameters
            page = request.args.get('page', 1, type=int)
            page_size = min(request.args.get('page_size', 50, type=int), 1000)
            
            # Get miner data
            miners_data = database_service.get_latest_status()
            
            # Convert to response format
            miners = []
            for miner_data in miners_data:
                miner = {
                    'ip': miner_data.get('ip', ''),
                    'hostname': miner_data.get('hostname'),
                    'temperature': miner_data.get('temp'),
                    'hash_rate': miner_data.get('hashRate'),
                    'power': miner_data.get('power'),
                    'voltage': miner_data.get('voltage'),
                    'frequency': miner_data.get('frequency'),
                    'core_voltage': miner_data.get('coreVoltage'),
                    'fan_rpm': miner_data.get('fanrpm'),
                    'shares_accepted': miner_data.get('sharesAccepted'),
                    'shares_rejected': miner_data.get('sharesRejected'),
                    'uptime': miner_data.get('uptime'),
                    'version': miner_data.get('version'),
                    'efficiency': miner_data.get('hashRate', 0) / miner_data.get('power', 1) if miner_data.get('power', 0) > 0 else 0,
                    'last_seen': miner_data.get('timestamp', datetime.now().isoformat())
                }
                miners.append(miner)
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_miners = miners[start_idx:end_idx]
            
            total_pages = (len(miners) + page_size - 1) // page_size
            
            logger.info("Miners list retrieved", total_miners=len(miners), page=page, page_size=page_size)
            
            return {
                'success': True,
                'data': paginated_miners,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': len(miners),
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                }
            }
            
        except Exception as e:
            logger.error("Get miners error", error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@miners_ns.route('/<string:ip>')
class MinerDetail(Resource):
    @miners_ns.doc('get_miner')
    @miners_ns.doc(security='Bearer')
    @miners_ns.response(200, 'Success', fields.Raw({'success': fields.Boolean, 'data': fields.Nested(miner_status_model)}))
    @miners_ns.response(401, 'Unauthorized', error_model)
    @miners_ns.response(404, 'Miner not found', error_model)
    @require_permissions('read')
    def get(self, ip):
        """Get specific miner status"""
        try:
            from services.service_container_v2 import get_container_v2
            container = get_container_v2()
            database_service = container.get_database_service()
            
            # Get miner data
            miner_data = database_service.get_latest_status_by_ip(ip)
            
            if not miner_data:
                return {'success': False, 'error': {'code': 'MINER_NOT_FOUND', 'message': f'Miner {ip} not found or no recent data', 'ip': ip}}, 404
            
            miner = {
                'ip': miner_data.get('ip', ip),
                'hostname': miner_data.get('hostname'),
                'temperature': miner_data.get('temp'),
                'hash_rate': miner_data.get('hashRate'),
                'power': miner_data.get('power'),
                'voltage': miner_data.get('voltage'),
                'frequency': miner_data.get('frequency'),
                'core_voltage': miner_data.get('coreVoltage'),
                'fan_rpm': miner_data.get('fanrpm'),
                'shares_accepted': miner_data.get('sharesAccepted'),
                'shares_rejected': miner_data.get('sharesRejected'),
                'uptime': miner_data.get('uptime'),
                'version': miner_data.get('version'),
                'efficiency': miner_data.get('hashRate', 0) / miner_data.get('power', 1) if miner_data.get('power', 0) > 0 else 0,
                'last_seen': miner_data.get('timestamp', datetime.now().isoformat())
            }
            
            logger.debug("Miner status retrieved", miner_ip=ip)
            
            return {'success': True, 'data': miner}
            
        except Exception as e:
            logger.error("Get miner error", miner_ip=ip, error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@miners_ns.route('/<string:ip>/settings')
class MinerSettings(Resource):
    @miners_ns.doc('update_miner_settings')
    @miners_ns.doc(security='Bearer')
    @miners_ns.expect(miner_settings_model, validate=True)
    @miners_ns.response(200, 'Success', fields.Raw({'success': fields.Boolean, 'message': fields.String, 'data': fields.Raw}))
    @miners_ns.response(401, 'Unauthorized', error_model)
    @miners_ns.response(400, 'Bad request', error_model)
    @miners_ns.response(500, 'Settings update failed', error_model)
    @require_permissions('control')
    def put(self, ip):
        """Update miner settings"""
        try:
            from services.service_container_v2 import get_container_v2
            container = get_container_v2()
            miner_service = container.get_miner_service()
            
            # Parse request
            data = request.get_json()
            
            if not data:
                return {'success': False, 'error': {'code': 'MISSING_DATA', 'message': 'Request body required'}}, 400
            
            frequency = data.get('frequency')
            core_voltage = data.get('core_voltage')
            autofanspeed = data.get('autofanspeed', True)
            
            if not frequency or not core_voltage:
                return {'success': False, 'error': {'code': 'MISSING_PARAMETERS', 'message': 'frequency and core_voltage required'}}, 400
            
            logger.info("Miner settings update requested",
                       miner_ip=ip,
                       frequency=frequency,
                       core_voltage=core_voltage,
                       username=request.current_user.get('username'))
            
            # Update settings
            success = miner_service.set_miner_settings(ip, frequency, core_voltage, autofanspeed)
            
            if not success:
                return {'success': False, 'error': {'code': 'SETTINGS_UPDATE_FAILED', 'message': f'Failed to update settings for miner {ip}', 'ip': ip}}, 500
            
            return {
                'success': True,
                'message': f'Settings updated for miner {ip}',
                'data': {
                    'ip': ip,
                    'frequency': frequency,
                    'core_voltage': core_voltage,
                    'autofanspeed': autofanspeed
                }
            }
            
        except Exception as e:
            logger.error("Update miner settings error", miner_ip=ip, error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@miners_ns.route('/<string:ip>/restart')
class MinerRestart(Resource):
    @miners_ns.doc('restart_miner')
    @miners_ns.doc(security='Bearer')
    @miners_ns.response(200, 'Success', fields.Raw({'success': fields.Boolean, 'message': fields.String, 'data': fields.Raw}))
    @miners_ns.response(401, 'Unauthorized', error_model)
    @miners_ns.response(500, 'Restart failed', error_model)
    @require_permissions('control')
    def post(self, ip):
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
                return {'success': False, 'error': {'code': 'RESTART_FAILED', 'message': f'Failed to restart miner {ip}', 'ip': ip}}, 500
            
            return {
                'success': True,
                'message': f'Restart command sent to miner {ip}',
                'data': {'ip': ip, 'action': 'restart'}
            }
            
        except Exception as e:
            logger.error("Restart miner error", miner_ip=ip, error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


@miners_ns.route('/summary')
class MinersSummary(Resource):
    @miners_ns.doc('get_miners_summary')
    @miners_ns.doc(security='Bearer')
    @miners_ns.response(200, 'Success', miners_summary_response)
    @miners_ns.response(401, 'Unauthorized', error_model)
    @require_permissions('read')
    def get(self):
        """Get miners summary statistics"""
        try:
            from services.service_container_v2 import get_container_v2
            container = get_container_v2()
            miner_service = container.get_miner_service()
            
            summary_data = miner_service.get_miners_summary()
            
            summary = {
                'total_miners': summary_data['total_miners'],
                'online_miners': summary_data['online_miners'],
                'offline_miners': summary_data['offline_miners'],
                'total_hashrate': summary_data['total_hashrate'],
                'total_power': summary_data['total_power'],
                'total_efficiency': summary_data['total_efficiency'],
                'average_temperature': summary_data['average_temperature'],
                'timestamp': summary_data['timestamp']
            }
            
            logger.debug("Miners summary retrieved")
            
            return {'success': True, 'data': summary}
            
        except Exception as e:
            logger.error("Get miners summary error", error=str(e))
            return {'success': False, 'error': {'code': 'INTERNAL_ERROR', 'message': 'Internal server error'}}, 500


# Register the enhanced API
def register_restx_api(app):
    """Register Flask-RESTX API with Flask app"""
    api.init_app(app)
    
    logger.info("Flask-RESTX API registered",
               version=api.version,
               title=api.title,
               doc_url=api.doc)