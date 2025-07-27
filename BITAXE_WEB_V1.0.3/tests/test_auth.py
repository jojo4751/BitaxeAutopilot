"""
Tests for authentication and authorization
"""

import pytest
import jwt
from datetime import datetime, timedelta

from auth.auth_service import AuthService, User, AuthenticationError, AuthorizationError
from tests.conftest import assert_json_response, assert_api_success, assert_error_response


class TestUser:
    """Test User model"""
    
    def test_create_user(self):
        """Test user creation"""
        user = User.create_user(
            username="testuser",
            password="testpass",
            roles=["operator"],
            permissions=["read", "write"]
        )
        
        assert user.username == "testuser"
        assert user.roles == ["operator"]
        assert user.permissions == ["read", "write"]
        assert user.is_active is True
        assert user.check_password("testpass")
        assert not user.check_password("wrongpass")
    
    def test_user_permissions(self):
        """Test user permission checking"""
        user = User.create_user(
            username="testuser",
            password="testpass",
            roles=["operator"],
            permissions=["read", "write"]
        )
        
        assert user.has_permission("read")
        assert user.has_permission("write")
        assert not user.has_permission("admin")
        assert user.has_role("operator")
        assert not user.has_role("admin")
    
    def test_admin_override(self):
        """Test admin role overrides permissions"""
        admin_user = User.create_user(
            username="admin",
            password="adminpass",
            roles=["admin"],
            permissions=["read"]
        )
        
        # Admin should have all permissions regardless of explicit permissions
        assert admin_user.has_permission("read")
        assert admin_user.has_permission("write")
        assert admin_user.has_permission("control")
        assert admin_user.has_permission("admin")
    
    def test_to_dict(self):
        """Test user serialization"""
        user = User.create_user(
            username="testuser",
            password="testpass",
            roles=["operator"],
            permissions=["read", "write"]
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["username"] == "testuser"
        assert user_dict["roles"] == ["operator"]
        assert user_dict["permissions"] == ["read", "write"]
        assert user_dict["is_active"] is True
        assert "password_hash" not in user_dict  # Should not expose sensitive data


class TestAuthService:
    """Test AuthService functionality"""
    
    def test_service_initialization(self):
        """Test auth service initialization"""
        auth_service = AuthService("test-secret")
        
        # Should have default users
        assert "admin" in auth_service.users
        assert "readonly" in auth_service.users
        assert "operator" in auth_service.users
        
        # Check default user properties
        admin_user = auth_service.users["admin"]
        assert admin_user.has_role("admin")
        assert admin_user.has_permission("admin")
    
    def test_user_authentication(self, auth_service):
        """Test user authentication"""
        # Valid credentials
        user = auth_service.authenticate_user("admin", "admin123")
        assert user is not None
        assert user.username == "admin"
        
        # Invalid username
        user = auth_service.authenticate_user("nonexistent", "password")
        assert user is None
        
        # Invalid password
        user = auth_service.authenticate_user("admin", "wrongpassword")
        assert user is None
    
    def test_token_generation(self, auth_service):
        """Test JWT token generation"""
        user = auth_service.authenticate_user("admin", "admin123")
        token_data = auth_service.generate_token(user)
        
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert "expires_in" in token_data
        assert "expires_at" in token_data
        assert token_data["token_type"] == "bearer"
        
        # Verify token can be decoded
        token = token_data["access_token"]
        payload = jwt.decode(token, auth_service.secret_key, algorithms=['HS256'])
        
        assert payload["username"] == "admin"
        assert "admin" in payload["roles"]
        assert "admin" in payload["permissions"]
    
    def test_token_verification(self, auth_service):
        """Test JWT token verification"""
        user = auth_service.authenticate_user("admin", "admin123")
        token_data = auth_service.generate_token(user)
        token = token_data["access_token"]
        
        # Valid token
        payload = auth_service.verify_token(token)
        assert payload is not None
        assert payload["username"] == "admin"
        
        # Invalid token
        payload = auth_service.verify_token("invalid.token.here")
        assert payload is None
        
        # Expired token
        expired_payload = {
            'username': 'admin',
            'roles': ['admin'],
            'permissions': ['admin'],
            'exp': datetime.utcnow() - timedelta(hours=1),  # Expired
            'iat': datetime.utcnow() - timedelta(hours=2),
            'jti': 'test-token-id'
        }
        expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm='HS256')
        payload = auth_service.verify_token(expired_token)
        assert payload is None
    
    def test_token_revocation(self, auth_service):
        """Test token revocation"""
        user = auth_service.authenticate_user("admin", "admin123")
        token_data = auth_service.generate_token(user)
        token = token_data["access_token"]
        
        # Token should be valid initially
        payload = auth_service.verify_token(token)
        assert payload is not None
        
        # Revoke token
        revoked = auth_service.revoke_token(token)
        assert revoked is True
        
        # Token should no longer be valid
        payload = auth_service.verify_token(token)
        assert payload is None
    
    def test_user_tokens_revocation(self, auth_service):
        """Test revoking all tokens for a user"""
        user = auth_service.authenticate_user("admin", "admin123")
        
        # Generate multiple tokens
        token1_data = auth_service.generate_token(user)
        token2_data = auth_service.generate_token(user)
        
        token1 = token1_data["access_token"]
        token2 = token2_data["access_token"]
        
        # Both tokens should be valid
        assert auth_service.verify_token(token1) is not None
        assert auth_service.verify_token(token2) is not None
        
        # Revoke all user tokens
        revoked_count = auth_service.revoke_user_tokens("admin")
        assert revoked_count == 2
        
        # Both tokens should be invalid
        assert auth_service.verify_token(token1) is None
        assert auth_service.verify_token(token2) is None
    
    def test_expired_tokens_cleanup(self, auth_service):
        """Test cleanup of expired tokens"""
        user = auth_service.authenticate_user("admin", "admin123")
        
        # Create expired token manually
        expired_payload = {
            'username': 'admin',
            'roles': ['admin'],
            'permissions': ['admin'],
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2),
            'jti': 'expired-token-id'
        }
        
        # Add expired token to active tokens list
        token_info = {
            'username': 'admin',
            'created_at': datetime.utcnow() - timedelta(hours=2),
            'expires_at': datetime.utcnow() - timedelta(hours=1),
            'token_id': 'expired-token-id'
        }
        auth_service.active_tokens['expired-token-id'] = token_info
        
        # Generate valid token
        token_data = auth_service.generate_token(user)
        
        # Should have 2 active tokens (1 expired, 1 valid)
        assert len(auth_service.active_tokens) == 2
        
        # Cleanup expired tokens
        cleaned_count = auth_service.cleanup_expired_tokens()
        assert cleaned_count == 1
        assert len(auth_service.active_tokens) == 1
    
    def test_get_active_sessions(self, auth_service):
        """Test getting active sessions"""
        user = auth_service.authenticate_user("admin", "admin123")
        token_data = auth_service.generate_token(user)
        
        sessions = auth_service.get_active_sessions()
        assert len(sessions) == 1
        
        session = sessions[0]
        assert session['username'] == 'admin'
        assert 'token_id' in session
        assert 'created_at' in session
        assert 'expires_at' in session


class TestAuthAPI:
    """Test authentication API endpoints"""
    
    def test_login_success(self, client):
        """Test successful login"""
        response = client.post('/api/v1/auth/login', 
                             json={'username': 'admin', 'password': 'admin123'})
        
        data = assert_json_response(response, 200)
        assert_api_success(data)
        
        assert 'data' in data
        token_data = data['data']
        assert 'access_token' in token_data
        assert 'token_type' in token_data
        assert token_data['token_type'] == 'bearer'
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post('/api/v1/auth/login',
                             json={'username': 'admin', 'password': 'wrongpass'})
        
        data = assert_json_response(response, 401)
        assert_error_response(data, 'INVALID_CREDENTIALS')
    
    def test_login_missing_data(self, client):
        """Test login with missing data"""
        response = client.post('/api/v1/auth/login', json={})
        
        data = assert_json_response(response, 400)
        assert_error_response(data, 'MISSING_CREDENTIALS')
    
    def test_logout_success(self, client, auth_headers):
        """Test successful logout"""
        response = client.post('/api/v1/auth/logout', headers=auth_headers)
        
        data = assert_json_response(response, 200)
        assert_api_success(data)
        assert data['message'] == 'Logout successful'
    
    def test_logout_no_token(self, client):
        """Test logout without token"""
        response = client.post('/api/v1/auth/logout')
        
        data = assert_json_response(response, 401)
        assert_error_response(data, 'AUTHENTICATION_REQUIRED')
    
    def test_get_user_info(self, client, auth_headers):
        """Test getting user info"""
        response = client.get('/api/v1/auth/user', headers=auth_headers)
        
        data = assert_json_response(response, 200)
        assert_api_success(data)
        
        user_info = data['data']
        assert user_info['username'] == 'admin'
        assert 'admin' in user_info['roles']
        assert 'admin' in user_info['permissions']
    
    def test_permission_enforcement(self, client, readonly_headers):
        """Test permission enforcement on protected endpoints"""
        # Readonly user should not be able to update miner settings
        response = client.put('/api/v1/miners/192.168.1.100/settings',
                            headers=readonly_headers,
                            json={'frequency': 800, 'core_voltage': 1200})
        
        data = assert_json_response(response, 403)
        assert_error_response(data, 'INSUFFICIENT_PERMISSIONS')
    
    def test_role_based_access(self, client, operator_headers):
        """Test role-based access control"""
        # Operator should be able to control miners but not admin functions
        response = client.put('/api/v1/config',
                            headers=operator_headers,
                            json={'key': 'test.setting', 'value': 'test'})
        
        data = assert_json_response(response, 403)
        assert_error_response(data, 'INSUFFICIENT_PERMISSIONS')


@pytest.mark.integration
class TestAuthIntegration:
    """Integration tests for authentication system"""
    
    def test_full_auth_flow(self, client):
        """Test complete authentication flow"""
        # 1. Login
        login_response = client.post('/api/v1/auth/login',
                                   json={'username': 'admin', 'password': 'admin123'})
        
        login_data = assert_json_response(login_response, 200)
        token = login_data['data']['access_token']
        
        # 2. Use token to access protected endpoint
        headers = {'Authorization': f'Bearer {token}'}
        protected_response = client.get('/api/v1/auth/user', headers=headers)
        
        user_data = assert_json_response(protected_response, 200)
        assert user_data['data']['username'] == 'admin'
        
        # 3. Logout
        logout_response = client.post('/api/v1/auth/logout', headers=headers)
        assert_json_response(logout_response, 200)
        
        # 4. Try to access protected endpoint with revoked token
        revoked_response = client.get('/api/v1/auth/user', headers=headers)
        assert_json_response(revoked_response, 401)
    
    def test_concurrent_sessions(self, client):
        """Test multiple concurrent sessions"""
        # Create two sessions for same user
        login1 = client.post('/api/v1/auth/login',
                           json={'username': 'admin', 'password': 'admin123'})
        login2 = client.post('/api/v1/auth/login',
                           json={'username': 'admin', 'password': 'admin123'})
        
        token1 = login1.get_json()['data']['access_token']
        token2 = login2.get_json()['data']['access_token']
        
        # Both tokens should be valid
        headers1 = {'Authorization': f'Bearer {token1}'}
        headers2 = {'Authorization': f'Bearer {token2}'}
        
        response1 = client.get('/api/v1/auth/user', headers=headers1)
        response2 = client.get('/api/v1/auth/user', headers=headers2)
        
        assert_json_response(response1, 200)
        assert_json_response(response2, 200)
        
        # Logout from first session
        client.post('/api/v1/auth/logout', headers=headers1)
        
        # First token should be invalid, second should still work
        response1 = client.get('/api/v1/auth/user', headers=headers1)
        response2 = client.get('/api/v1/auth/user', headers=headers2)
        
        assert_json_response(response1, 401)
        assert_json_response(response2, 200)