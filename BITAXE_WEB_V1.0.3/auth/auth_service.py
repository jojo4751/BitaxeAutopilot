import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps
from flask import request, jsonify, current_app

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import (
    ValidationError, ServiceError, ErrorCode
)

logger = get_logger("bitaxe.auth")


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization related errors"""
    pass


class User:
    """User model for authentication"""
    
    def __init__(self, username: str, password_hash: str, roles: List[str], 
                 permissions: List[str], is_active: bool = True):
        self.username = username
        self.password_hash = password_hash
        self.roles = roles
        self.permissions = permissions
        self.is_active = is_active
        self.created_at = datetime.now()
        self.last_login = None
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches the hash"""
        return self._hash_password(password) == self.password_hash
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions or 'admin' in self.roles
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @classmethod
    def create_user(cls, username: str, password: str, roles: List[str], 
                   permissions: List[str]) -> 'User':
        """Create new user with hashed password"""
        password_hash = cls._hash_password(password)
        return cls(username, password_hash, roles, permissions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (without sensitive data)"""
        return {
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.users: Dict[str, User] = {}
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default users
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default users"""
        # Admin user
        admin_user = User.create_user(
            username="admin",
            password="admin123",  # Should be changed in production
            roles=["admin", "operator"],
            permissions=["read", "write", "control", "admin", "benchmark"]
        )
        self.users["admin"] = admin_user
        
        # Read-only user
        readonly_user = User.create_user(
            username="readonly",
            password="readonly123",
            roles=["viewer"],
            permissions=["read"]
        )
        self.users["readonly"] = readonly_user
        
        # Operator user
        operator_user = User.create_user(
            username="operator",
            password="operator123",
            roles=["operator"],
            permissions=["read", "write", "control"]
        )
        self.users["operator"] = operator_user
        
        logger.info("Default users initialized",
                   total_users=len(self.users),
                   usernames=list(self.users.keys()))
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        try:
            user = self.users.get(username)
            if not user:
                logger.warning("Authentication failed: user not found",
                             username=username)
                return None
            
            if not user.is_active:
                logger.warning("Authentication failed: user inactive",
                             username=username)
                return None
            
            if not user.check_password(password):
                logger.warning("Authentication failed: invalid password",
                             username=username)
                return None
            
            # Update last login
            user.last_login = datetime.now()
            
            logger.info("User authenticated successfully",
                       username=username,
                       roles=user.roles)
            
            return user
            
        except Exception as e:
            logger.error("Authentication error",
                        username=username,
                        error=str(e))
            return None
    
    def generate_token(self, user: User) -> Dict[str, Any]:
        """Generate JWT token for authenticated user"""
        try:
            payload = {
                'username': user.username,
                'roles': user.roles,
                'permissions': user.permissions,
                'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                'iat': datetime.utcnow(),
                'jti': secrets.token_urlsafe(16)  # Unique token ID
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Store active token info
            token_info = {
                'username': user.username,
                'created_at': datetime.utcnow(),
                'expires_at': payload['exp'],
                'token_id': payload['jti']
            }
            self.active_tokens[payload['jti']] = token_info
            
            logger.info("Token generated",
                       username=user.username,
                       token_id=payload['jti'],
                       expires_at=payload['exp'].isoformat())
            
            return {
                'access_token': token,
                'token_type': 'bearer',
                'expires_in': self.token_expiry_hours * 3600,
                'expires_at': payload['exp'].isoformat()
            }
            
        except Exception as e:
            logger.error("Token generation failed",
                        username=user.username,
                        error=str(e))
            raise ServiceError(
                f"Failed to generate token for user {user.username}",
                "auth_service",
                ErrorCode.SERVICE_UNAVAILABLE,
                cause=e
            )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is in active tokens list
            token_id = payload.get('jti')
            if token_id not in self.active_tokens:
                logger.warning("Token verification failed: token not active",
                             token_id=token_id)
                return None
            
            # Check if user still exists and is active
            username = payload.get('username')
            user = self.users.get(username)
            if not user or not user.is_active:
                logger.warning("Token verification failed: user inactive",
                             username=username)
                return None
            
            logger.debug("Token verified successfully",
                        username=username,
                        token_id=token_id)
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Token verification failed: invalid token",
                          error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error",
                        error=str(e))
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            token_id = payload.get('jti')
            
            if token_id in self.active_tokens:
                del self.active_tokens[token_id]
                
                logger.info("Token revoked",
                           username=payload.get('username'),
                           token_id=token_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Token revocation error", error=str(e))
            return False
    
    def revoke_user_tokens(self, username: str) -> int:
        """Revoke all tokens for a specific user"""
        revoked_count = 0
        tokens_to_remove = []
        
        for token_id, token_info in self.active_tokens.items():
            if token_info['username'] == username:
                tokens_to_remove.append(token_id)
        
        for token_id in tokens_to_remove:
            del self.active_tokens[token_id]
            revoked_count += 1
        
        if revoked_count > 0:
            logger.info("User tokens revoked",
                       username=username,
                       revoked_count=revoked_count)
        
        return revoked_count
    
    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from active tokens list"""
        now = datetime.utcnow()
        expired_tokens = []
        
        for token_id, token_info in self.active_tokens.items():
            if token_info['expires_at'] < now:
                expired_tokens.append(token_id)
        
        for token_id in expired_tokens:
            del self.active_tokens[token_id]
        
        if expired_tokens:
            logger.info("Expired tokens cleaned up",
                       cleaned_count=len(expired_tokens))
        
        return len(expired_tokens)
    
    def get_user_info(self, username: str) -> Optional[User]:
        """Get user information"""
        return self.users.get(username)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        sessions = []
        for token_id, token_info in self.active_tokens.items():
            sessions.append({
                'token_id': token_id,
                'username': token_info['username'],
                'created_at': token_info['created_at'].isoformat(),
                'expires_at': token_info['expires_at'].isoformat()
            })
        
        return sessions


# Global auth service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get global auth service instance"""
    global _auth_service
    if _auth_service is None:
        secret_key = current_app.config.get('SECRET_KEY', 'fallback-secret-key')
        _auth_service = AuthService(secret_key)
    return _auth_service


def require_auth(permissions: Optional[List[str]] = None, roles: Optional[List[str]] = None):
    """Decorator to require authentication and optionally specific permissions/roles"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_service = get_auth_service()
            
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                logger.warning("Authentication required: no token provided",
                             endpoint=request.endpoint)
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'AUTHENTICATION_REQUIRED',
                        'message': 'Authentication token required'
                    }
                }), 401
            
            token = auth_header.split(' ')[1]
            payload = auth_service.verify_token(token)
            
            if not payload:
                logger.warning("Authentication failed: invalid token",
                             endpoint=request.endpoint)
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'INVALID_TOKEN',
                        'message': 'Invalid or expired token'
                    }
                }), 401
            
            # Check permissions if specified
            if permissions:
                user_permissions = payload.get('permissions', [])
                user_roles = payload.get('roles', [])
                
                has_permission = (
                    any(perm in user_permissions for perm in permissions) or
                    'admin' in user_roles
                )
                
                if not has_permission:
                    logger.warning("Authorization failed: insufficient permissions",
                                 username=payload.get('username'),
                                 required_permissions=permissions,
                                 user_permissions=user_permissions)
                    return jsonify({
                        'success': False,
                        'error': {
                            'code': 'INSUFFICIENT_PERMISSIONS',
                            'message': 'Insufficient permissions for this operation'
                        }
                    }), 403
            
            # Check roles if specified
            if roles:
                user_roles = payload.get('roles', [])
                
                has_role = any(role in user_roles for role in roles)
                
                if not has_role:
                    logger.warning("Authorization failed: insufficient roles",
                                 username=payload.get('username'),
                                 required_roles=roles,
                                 user_roles=user_roles)
                    return jsonify({
                        'success': False,
                        'error': {
                            'code': 'INSUFFICIENT_ROLES',
                            'message': 'Insufficient roles for this operation'
                        }
                    }), 403
            
            # Add user info to request context
            request.current_user = payload
            
            logger.debug("Authentication successful",
                        username=payload.get('username'),
                        endpoint=request.endpoint)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_permissions(*permissions):
    """Convenience decorator for requiring specific permissions"""
    return require_auth(permissions=list(permissions))


def require_roles(*roles):
    """Convenience decorator for requiring specific roles"""
    return require_auth(roles=list(roles))


def optional_auth(f):
    """Decorator that adds user info if authenticated but doesn't require it"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_service = get_auth_service()
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        request.current_user = None
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = auth_service.verify_token(token)
            
            if payload:
                request.current_user = payload
                logger.debug("Optional authentication successful",
                           username=payload.get('username'),
                           endpoint=request.endpoint)
        
        return f(*args, **kwargs)
    
    return decorated_function