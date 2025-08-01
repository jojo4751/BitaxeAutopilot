/**
 * BitAxe WebSocket Client
 * Real-time data streaming client for live dashboard updates
 */

class BitAxeWebSocketClient {
    constructor(options = {}) {
        this.options = {
            autoReconnect: true,
            reconnectDelay: 3000,
            maxReconnectAttempts: 5,
            pingInterval: 30000,
            ...options
        };
        
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.eventHandlers = {};
        this.subscriptions = {
            miners: [],
            alerts: true,
            systemStats: true,
            profitability: true
        };
        
        // Bind methods
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.onConnect = this.onConnect.bind(this);
        this.onDisconnect = this.onDisconnect.bind(this);
        this.onError = this.onError.bind(this);
        
        this.init();
    }
    
    init() {
        console.log('Initializing BitAxe WebSocket Client...');
        this.connect();
    }
    
    connect() {
        if (this.socket && this.isConnected) {
            console.log('WebSocket is already connected');
            return;
        }
        
        try {
            // Initialize Socket.IO connection
            this.socket = io({
                transports: ['websocket', 'polling'],
                timeout: 10000,
                forceNew: true
            });
            
            this.setupEventHandlers();
            console.log('WebSocket connection initiated...');
            
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.scheduleReconnect();
        }
    }
    
    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', this.onConnect);
        this.socket.on('disconnect', this.onDisconnect);
        this.socket.on('connect_error', this.onError);
        
        // Data events
        this.socket.on('initial_data', (data) => {
            console.log('Received initial data:', data);
            this.emit('initialData', data);
        });
        
        this.socket.on('live_update', (data) => {
            console.log('Received live update:', data);
            this.emit('liveUpdate', data);
        });
        
        this.socket.on('miner_update', (data) => {
            console.log('Received miner update:', data);
            this.emit('minerUpdate', data);
        });
        
        this.socket.on('new_alerts', (alerts) => {
            console.log('Received new alerts:', alerts);
            this.emit('newAlerts', alerts);
        });
        
        this.socket.on('alert_acknowledged', (data) => {
            console.log('Alert acknowledged:', data);
            this.emit('alertAcknowledged', data);
        });
        
        // Custom events
        this.socket.on('system_notification', (data) => {
            console.log('System notification:', data);
            this.emit('systemNotification', data);
        });
    }
    
    onConnect() {
        console.log('✅ WebSocket connected successfully');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Subscribe to data streams
        this.updateSubscriptions();
        
        this.emit('connected');
    }
    
    onDisconnect(reason) {
        console.log('❌ WebSocket disconnected:', reason);
        this.isConnected = false;
        
        this.emit('disconnected', reason);
        
        if (this.options.autoReconnect && reason !== 'io client disconnect') {
            this.scheduleReconnect();
        }
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.emit('error', error);
        
        if (!this.isConnected) {
            this.scheduleReconnect();
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('maxReconnectAttemptsReached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.options.reconnectDelay * this.reconnectAttempts;
        
        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            if (!this.isConnected) {
                console.log(`Reconnection attempt ${this.reconnectAttempts}...`);
                this.connect();
            }
        }, delay);
    }
    
    disconnect() {
        if (this.socket) {
            console.log('Disconnecting WebSocket...');
            this.socket.disconnect();
            this.socket = null;
        }
        this.isConnected = false;
    }
    
    // Subscription management
    subscribeToMiners(minerIPs) {
        if (!Array.isArray(minerIPs)) {
            minerIPs = [minerIPs];
        }
        
        this.subscriptions.miners = minerIPs;
        
        if (this.isConnected) {
            this.socket.emit('subscribe', {
                type: 'miners',
                data: { ips: minerIPs }
            });
        }
        
        console.log('Subscribed to miners:', minerIPs);
    }
    
    subscribeToAlerts(enabled = true) {
        this.subscriptions.alerts = enabled;
        
        if (this.isConnected) {
            this.socket.emit('subscribe', {
                type: 'alerts',
                data: { enabled }
            });
        }
        
        console.log('Alert subscription:', enabled ? 'enabled' : 'disabled');
    }
    
    subscribeToSystemStats(enabled = true) {
        this.subscriptions.systemStats = enabled;
        
        if (this.isConnected) {
            this.socket.emit('subscribe', {
                type: 'system_stats',
                data: { enabled }
            });
        }
        
        console.log('System stats subscription:', enabled ? 'enabled' : 'disabled');
    }
    
    updateSubscriptions() {
        if (!this.isConnected) return;
        
        // Subscribe to miners
        if (this.subscriptions.miners.length > 0) {
            this.subscribeToMiners(this.subscriptions.miners);
        }
        
        // Subscribe to alerts
        this.subscribeToAlerts(this.subscriptions.alerts);
        
        // Subscribe to system stats
        this.subscribeToSystemStats(this.subscriptions.systemStats);
    }
    
    // Data requests
    requestHistoricalData(ip, hours = 6) {
        return new Promise((resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('WebSocket not connected'));
                return;
            }
            
            const timeout = setTimeout(() => {
                reject(new Error('Request timeout'));
            }, 10000);
            
            this.socket.emit('get_historical_data', { ip, hours }, (response) => {
                clearTimeout(timeout);
                
                if (response.status === 'success') {
                    resolve(response);
                } else {
                    reject(new Error(response.message || 'Request failed'));
                }
            });
        });
    }
    
    acknowledgeAlert(alertId, user = 'web_user') {
        return new Promise((resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('WebSocket not connected'));
                return;
            }
            
            this.socket.emit('acknowledge_alert', { alert_id: alertId, user }, (response) => {
                if (response.status === 'success') {
                    resolve(response);
                } else {
                    reject(new Error(response.message || 'Acknowledgment failed'));
                }
            });
        });
    }
    
    // Event handling
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }
    
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }
    
    emit(event, ...args) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(...args);
                } catch (error) {
                    console.error('Error in event handler:', error);
                }
            });
        }
    }
    
    // Utility methods
    getConnectionStatus() {
        return {
            connected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            subscriptions: this.subscriptions
        };
    }
    
    ping() {
        if (this.isConnected && this.socket) {
            const startTime = Date.now();
            this.socket.emit('ping', startTime, (response) => {
                const latency = Date.now() - startTime;
                console.log(`WebSocket ping: ${latency}ms`);
                this.emit('ping', latency);
            });
        }
    }
    
    // Start periodic ping
    startPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }
        
        this.pingInterval = setInterval(() => {
            this.ping();
        }, this.options.pingInterval);
    }
    
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    // Cleanup
    destroy() {
        this.stopPing();
        this.disconnect();
        this.eventHandlers = {};
        console.log('WebSocket client destroyed');
    }
}

// Global WebSocket client instance
let bitaxeWS = null;

// Initialize WebSocket client when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing global WebSocket client...');
    
    bitaxeWS = new BitAxeWebSocketClient({
        autoReconnect: true,
        reconnectDelay: 3000,
        maxReconnectAttempts: 10,
        pingInterval: 30000
    });
    
    // Global event handlers
    bitaxeWS.on('connected', () => {
        console.log('🟢 WebSocket connected');
        updateConnectionStatus(true);
    });
    
    bitaxeWS.on('disconnected', (reason) => {
        console.log('🔴 WebSocket disconnected:', reason);
        updateConnectionStatus(false);
    });
    
    bitaxeWS.on('error', (error) => {
        console.error('🚨 WebSocket error:', error);
        showNotification('WebSocket connection error', 'error');
    });
    
    bitaxeWS.on('maxReconnectAttemptsReached', () => {
        console.error('❌ Max reconnection attempts reached');
        showNotification('Unable to connect to real-time updates', 'error');
    });
    
    // Start periodic ping
    bitaxeWS.startPing();
    
    // Make globally available
    window.bitaxeWS = bitaxeWS;
});

// Utility functions
function updateConnectionStatus(connected) {
    const statusElements = document.querySelectorAll('.ws-status');
    statusElements.forEach(element => {
        element.className = `ws-status ${connected ? 'connected' : 'disconnected'}`;
        element.title = connected ? 'Real-time updates active' : 'Real-time updates disconnected';
    });
    
    const statusIcons = document.querySelectorAll('.ws-status-icon');
    statusIcons.forEach(icon => {
        icon.className = `ws-status-icon ${connected ? 'online' : 'offline'}`;
    });
}

function showNotification(message, type = 'info', duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span class="notification-message">${message}</span>
        <button class="notification-close">&times;</button>
    `;
    
    // Add to page
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, duration);
    
    // Close button handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    });
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BitAxeWebSocketClient;
}