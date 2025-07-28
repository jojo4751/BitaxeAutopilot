"""
Load Testing Framework

Comprehensive load testing for the BitAxe ML-powered autonomous mining system.
Uses Locust framework for distributed load testing with realistic user scenarios.
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitAxeLoadTestUser(HttpUser):
    """
    Locust user class simulating realistic BitAxe system usage patterns
    """
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        self.auth_token = None
        self.miner_ips = []
        self.user_id = random.randint(1000, 9999)
        
        # Login and setup
        self.login()
        self.get_miners_list()
    
    def login(self):
        """Authenticate user"""
        login_data = {
            "username": f"test_user_{self.user_id}",
            "password": "test_password"
        }
        
        with self.client.post("/api/v1/auth/login", 
                             json=login_data, 
                             catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.auth_token = data.get('access_token')
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Login failed with status {response.status_code}")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    def get_miners_list(self):
        """Get list of available miners"""
        with self.client.get("/api/v1/miners", 
                            headers=self.get_auth_headers(),
                            catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.miner_ips = [miner['ip_address'] for miner in data.get('miners', [])]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Failed to get miners: {response.status_code}")
    
    @task(10)
    def view_dashboard(self):
        """View main dashboard - most common operation"""
        with self.client.get("/", 
                            headers=self.get_auth_headers(),
                            name="dashboard") as response:
            if response.status_code != 200:
                response.failure(f"Dashboard failed: {response.status_code}")
    
    @task(8)
    def get_system_status(self):
        """Get system status information"""
        with self.client.get("/api/v1/status", 
                            headers=self.get_auth_headers(),
                            name="system_status") as response:
            if response.status_code != 200:
                response.failure(f"System status failed: {response.status_code}")
    
    @task(15)
    def get_miners_status(self):
        """Get all miners status - frequently accessed"""
        with self.client.get("/api/v1/miners", 
                            headers=self.get_auth_headers(),
                            name="miners_status") as response:
            if response.status_code != 200:
                response.failure(f"Miners status failed: {response.status_code}")
    
    @task(6)
    def get_specific_miner(self):
        """Get specific miner details"""
        if not self.miner_ips:
            return
        
        miner_ip = random.choice(self.miner_ips)
        with self.client.get(f"/api/v1/miners/{miner_ip}", 
                            headers=self.get_auth_headers(),
                            name="specific_miner") as response:
            if response.status_code != 200:
                response.failure(f"Specific miner failed: {response.status_code}")
    
    @task(4)
    def update_miner_config(self):
        """Update miner configuration"""
        if not self.miner_ips:
            return
        
        miner_ip = random.choice(self.miner_ips)
        config_data = {
            "pool_url": f"stratum+tcp://pool{random.randint(1,3)}.example.com:4334",
            "pool_username": f"user_{self.user_id}",
            "frequency": random.randint(400, 600)
        }
        
        with self.client.put(f"/api/v1/miners/{miner_ip}/config", 
                            json=config_data,
                            headers=self.get_auth_headers(),
                            name="update_miner_config") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Update miner config failed: {response.status_code}")
    
    @task(3)
    def start_optimization(self):
        """Start ML optimization for a miner"""
        if not self.miner_ips:
            return
        
        miner_ip = random.choice(self.miner_ips)
        optimization_data = {
            "target_metric": "efficiency",
            "duration_minutes": random.randint(10, 60),
            "aggressive_mode": random.choice([True, False])
        }
        
        with self.client.post(f"/api/v1/miners/{miner_ip}/optimize", 
                             json=optimization_data,
                             headers=self.get_auth_headers(),
                             name="start_optimization") as response:
            if response.status_code not in [200, 201, 202]:
                response.failure(f"Start optimization failed: {response.status_code}")
    
    @task(2)
    def get_optimization_history(self):
        """Get optimization history"""
        if not self.miner_ips:
            return
        
        miner_ip = random.choice(self.miner_ips)
        with self.client.get(f"/api/v1/miners/{miner_ip}/optimizations", 
                            headers=self.get_auth_headers(),
                            name="optimization_history") as response:
            if response.status_code != 200:
                response.failure(f"Optimization history failed: {response.status_code}")
    
    @task(5)
    def get_mining_stats(self):
        """Get mining statistics"""
        params = {
            "period": random.choice(["1h", "24h", "7d"]),
            "metric": random.choice(["hashrate", "temperature", "efficiency"])
        }
        
        with self.client.get("/api/v1/stats", 
                            params=params,
                            headers=self.get_auth_headers(),
                            name="mining_stats") as response:
            if response.status_code != 200:
                response.failure(f"Mining stats failed: {response.status_code}")
    
    @task(1)
    def export_data(self):
        """Export mining data - resource intensive operation"""
        export_data = {
            "format": random.choice(["csv", "json"]),
            "period": "24h",
            "miners": random.sample(self.miner_ips, min(3, len(self.miner_ips))) if self.miner_ips else []
        }
        
        with self.client.post("/api/v1/export", 
                             json=export_data,
                             headers=self.get_auth_headers(),
                             name="export_data") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Export data failed: {response.status_code}")
    
    @task(7)
    def get_alerts(self):
        """Get system alerts"""
        with self.client.get("/api/v1/alerts", 
                            headers=self.get_auth_headers(),
                            name="alerts") as response:
            if response.status_code != 200:
                response.failure(f"Alerts failed: {response.status_code}")
    
    @task(1)
    def websocket_simulation(self):
        """Simulate WebSocket connection for real-time updates"""
        # Note: Locust doesn't natively support WebSocket testing
        # This is a placeholder for WebSocket load testing
        with self.client.get("/api/v1/realtime/connect", 
                            headers=self.get_auth_headers(),
                            name="websocket_connect") as response:
            if response.status_code != 200:
                response.failure(f"WebSocket connect failed: {response.status_code}")


class MLIntensiveUser(HttpUser):
    """
    User class focused on ML-intensive operations
    """
    
    wait_time = between(2, 8)  # Longer wait times for ML operations
    weight = 2  # Lower weight - fewer ML-intensive users
    
    def on_start(self):
        self.auth_token = None
        self.user_id = random.randint(5000, 5999)
        self.login()
    
    def login(self):
        login_data = {
            "username": f"ml_user_{self.user_id}",
            "password": "ml_password"
        }
        
        with self.client.post("/api/v1/auth/login", json=login_data) as response:
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('access_token')
    
    def get_auth_headers(self):
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    @task(5)
    def bulk_optimization(self):
        """Start bulk optimization across multiple miners"""
        optimization_data = {
            "target_miners": "all",
            "optimization_type": "efficiency",
            "duration_minutes": 30,
            "max_concurrent": random.randint(5, 15)
        }
        
        with self.client.post("/api/v1/optimize/bulk", 
                             json=optimization_data,
                             headers=self.get_auth_headers(),
                             name="bulk_optimization") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Bulk optimization failed: {response.status_code}")
    
    @task(3)
    def ml_model_training(self):
        """Trigger ML model retraining"""
        training_data = {
            "model_type": random.choice(["efficiency", "temperature", "hashrate"]),
            "training_data_days": random.randint(7, 30),
            "validation_split": 0.2
        }
        
        with self.client.post("/api/v1/ml/train", 
                             json=training_data,
                             headers=self.get_auth_headers(),
                             name="ml_training") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"ML training failed: {response.status_code}")
    
    @task(4)
    def advanced_analytics(self):
        """Get advanced analytics and predictions"""
        analytics_params = {
            "analysis_type": random.choice(["prediction", "anomaly_detection", "trend_analysis"]),
            "time_horizon": random.choice(["1h", "6h", "24h"]),
            "confidence_threshold": random.uniform(0.7, 0.95)
        }
        
        with self.client.get("/api/v1/analytics/advanced", 
                            params=analytics_params,
                            headers=self.get_auth_headers(),
                            name="advanced_analytics") as response:
            if response.status_code != 200:
                response.failure(f"Advanced analytics failed: {response.status_code}")
    
    @task(2)
    def model_performance_metrics(self):
        """Get ML model performance metrics"""
        with self.client.get("/api/v1/ml/metrics", 
                            headers=self.get_auth_headers(),
                            name="ml_metrics") as response:
            if response.status_code != 200:
                response.failure(f"ML metrics failed: {response.status_code}")


class AdminUser(HttpUser):
    """
    Admin user with access to administrative functions
    """
    
    wait_time = between(5, 15)  # Longer intervals for admin operations
    weight = 1  # Very few admin users
    
    def on_start(self):
        self.auth_token = None
        self.admin_id = random.randint(9000, 9999)
        self.login()
    
    def login(self):
        login_data = {
            "username": f"admin_{self.admin_id}",
            "password": "admin_password"
        }
        
        with self.client.post("/api/v1/auth/login", json=login_data) as response:
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('access_token')
    
    def get_auth_headers(self):
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    @task(3)
    def system_health_check(self):
        """Get comprehensive system health"""
        with self.client.get("/api/v1/admin/health", 
                            headers=self.get_auth_headers(),
                            name="admin_health") as response:
            if response.status_code != 200:
                response.failure(f"Admin health check failed: {response.status_code}")
    
    @task(2)
    def user_management(self):
        """User management operations"""
        action = random.choice(["list", "create", "update"])
        
        if action == "list":
            with self.client.get("/api/v1/admin/users", 
                                headers=self.get_auth_headers(),
                                name="admin_users_list") as response:
                if response.status_code != 200:
                    response.failure(f"Admin users list failed: {response.status_code}")
        
        elif action == "create":
            user_data = {
                "username": f"load_test_user_{random.randint(10000, 99999)}",
                "email": f"test{random.randint(1000, 9999)}@example.com",
                "role": random.choice(["user", "operator"])
            }
            with self.client.post("/api/v1/admin/users", 
                                 json=user_data,
                                 headers=self.get_auth_headers(),
                                 name="admin_create_user") as response:
                if response.status_code not in [200, 201]:
                    response.failure(f"Admin create user failed: {response.status_code}")
    
    @task(1)
    def system_configuration(self):
        """Update system configuration"""
        config_data = {
            "optimization_interval": random.randint(30, 300),
            "max_concurrent_optimizations": random.randint(5, 20),
            "alert_thresholds": {
                "temperature": random.randint(80, 90),
                "hashrate_drop": random.randint(10, 20)
            }
        }
        
        with self.client.put("/api/v1/admin/config", 
                            json=config_data,
                            headers=self.get_auth_headers(),
                            name="admin_config") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Admin config update failed: {response.status_code}")
    
    @task(1)
    def backup_data(self):
        """Trigger system backup"""
        backup_data = {
            "include_logs": True,
            "include_metrics": True,
            "compression": "gzip"
        }
        
        with self.client.post("/api/v1/admin/backup", 
                             json=backup_data,
                             headers=self.get_auth_headers(),
                             name="admin_backup") as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Admin backup failed: {response.status_code}")


# Event handlers for custom metrics collection
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Custom request event handler"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    logger.info("Load test started")
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running in distributed mode")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    logger.info("Load test completed")
    
    # Log final statistics
    stats = environment.runner.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Max response time: {stats.total.max_response_time:.2f}ms")


# Custom load test shapes for different scenarios
class StepLoadShape:
    """
    Custom load shape that increases load in steps
    """
    
    step_time = 60  # seconds
    step_load = 10  # users per step
    spawn_rate = 2  # users per second
    time_limit = 600  # total test time in seconds
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)


if __name__ == "__main__":
    # Example of running load tests programmatically
    import subprocess
    import sys
    
    print("Starting BitAxe Load Tests...")
    
    # Basic load test command
    cmd = [
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--host", "http://localhost:5000",
        "--users", "50",
        "--spawn-rate", "5",
        "--run-time", "5m",
        "--headless",
        "--html", "load_test_report.html"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Load test completed!")
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Failed to run load test: {e}")