/**
 * BitAxe Dashboard Core
 * Main dashboard functionality and state management
 */

window.DashboardCore = (function() {
    'use strict';

    // Dashboard state
    let state = {
        websocket: null,
        reconnectAttempts: 0,
        maxReconnectAttempts: 5,
        reconnectDelay: 1000,
        isConnected: false,
        refreshRate: 5000,
        animationsEnabled: true,
        theme: 'cyber-dark',
        miners: new Map(),
        charts: new Map(),
        settings: {}
    };

    // DOM elements cache
    let elements = {};

    /**
     * Initialize dashboard
     */
    function init() {
        cacheElements();
        setupEventListeners();
        loadSettings();
        initializeWebSocket();
        startDataRefresh();
        
        console.log('Dashboard Core initialized');
    }

    /**
     * Cache DOM elements for performance
     */
    function cacheElements() {
        elements = {
            connectionStatus: document.getElementById('connection-status'),
            mobileMenuToggle: document.getElementById('mobile-menu-toggle'),
            mobileNavPanel: document.getElementById('mobile-nav-panel'),
            settingsToggle: document.getElementById('settings-toggle'),
            settingsPanel: document.getElementById('settings-panel'),
            themeToggle: document.getElementById('theme-toggle'),
            quickActionsFab: document.getElementById('quick-actions-fab'),
            fabMenu: document.getElementById('fab-menu'),
            alertContainer: document.getElementById('alert-container'),
            refreshRateSelect: document.getElementById('refresh-rate'),
            animationsToggle: document.getElementById('animations-toggle'),
            eventsB badge: document.getElementById('events-badge')
        };
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        // Mobile menu toggle
        if (elements.mobileMenuToggle) {
            elements.mobileMenuToggle.addEventListener('click', toggleMobileMenu);
        }

        // Settings panel toggle
        if (elements.settingsToggle) {
            elements.settingsToggle.addEventListener('click', toggleSettings);
        }

        // Theme toggle
        if (elements.themeToggle) {
            elements.themeToggle.addEventListener('click', showThemeSelector);
        }

        // FAB menu
        if (elements.quickActionsFab) {
            elements.quickActionsFab.addEventListener('click', toggleFabMenu);
        }

        // Settings controls
        if (elements.refreshRateSelect) {
            elements.refreshRateSelect.addEventListener('change', updateRefreshRate);
        }

        if (elements.animationsToggle) {
            elements.animationsToggle.addEventListener('change', toggleAnimations);
        }

        // Theme selector buttons
        document.querySelectorAll('.theme-option').forEach(button => {
            button.addEventListener('click', function() {
                setTheme(this.dataset.theme);
            });
        });

        // Close settings panel when clicking outside
        document.addEventListener('click', function(e) {
            if (!elements.settingsPanel?.contains(e.target) && 
                !elements.settingsToggle?.contains(e.target) &&
                elements.settingsPanel?.classList.contains('show')) {
                closeSettings();
            }
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!elements.mobileNavPanel?.contains(e.target) && 
                !elements.mobileMenuToggle?.contains(e.target) &&
                elements.mobileNavPanel?.classList.contains('show')) {
                closeMobileMenu();
            }
        });

        // Handle window resize
        window.addEventListener('resize', debounce(handleResize, 250));

        // Handle visibility change (pause/resume when tab not visible)
        document.addEventListener('visibilitychange', handleVisibilityChange);

        // Keyboard shortcuts
        document.addEventListener('keydown', handleKeyboardShortcuts);
    }

    /**
     * Load user settings from localStorage
     */
    function loadSettings() {
        const saved = localStorage.getItem('bitaxe-dashboard-settings');
        if (saved) {
            try {
                state.settings = JSON.parse(saved);
                applySettings();
            } catch (e) {
                console.warn('Failed to load settings:', e);
            }
        }
    }

    /**
     * Save settings to localStorage
     */
    function saveSettings() {
        localStorage.setItem('bitaxe-dashboard-settings', JSON.stringify(state.settings));
    }

    /**
     * Apply loaded settings
     */
    function applySettings() {
        if (state.settings.theme) {
            setTheme(state.settings.theme);
        }
        if (state.settings.refreshRate) {
            state.refreshRate = state.settings.refreshRate;
            if (elements.refreshRateSelect) {
                elements.refreshRateSelect.value = state.refreshRate;
            }
        }
        if (state.settings.animationsEnabled !== undefined) {
            state.animationsEnabled = state.settings.animationsEnabled;
            if (elements.animationsToggle) {
                elements.animationsToggle.checked = state.animationsEnabled;
            }
            document.body.classList.toggle('animations-disabled', !state.animationsEnabled);
        }
    }

    /**
     * Initialize WebSocket connection for real-time data
     */
    function initializeWebSocket() {
        if (!window.WebSocket) {
            console.warn('WebSocket not supported');
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            state.websocket = new WebSocket(wsUrl);
            
            state.websocket.onopen = function() {
                state.isConnected = true;
                state.reconnectAttempts = 0;
                updateConnectionStatus('online');
                console.log('WebSocket connected');
            };

            state.websocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            state.websocket.onclose = function() {
                state.isConnected = false;
                updateConnectionStatus('offline');
                console.log('WebSocket disconnected');
                attemptReconnect();
            };

            state.websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('error');
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            updateConnectionStatus('error');
        }
    }

    /**
     * Handle WebSocket messages
     */
    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'miner_update':
                updateMinerData(data.payload);
                break;
            case 'system_alert':
                showAlert(data.payload.message, data.payload.type);
                break;
            case 'benchmark_progress':
                updateBenchmarkProgress(data.payload);
                break;
            case 'events_count':
                updateEventsCount(data.payload.count);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    /**
     * Attempt to reconnect WebSocket
     */
    function attemptReconnect() {
        if (state.reconnectAttempts >= state.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        state.reconnectAttempts++;
        const delay = state.reconnectDelay * Math.pow(2, state.reconnectAttempts - 1);
        
        setTimeout(() => {
            console.log(`Attempting to reconnect... (${state.reconnectAttempts}/${state.maxReconnectAttempts})`);
            initializeWebSocket();
        }, delay);
    }

    /**
     * Start data refresh interval
     */
    function startDataRefresh() {
        setInterval(() => {
            if (!document.hidden && !state.isConnected) {
                // Fallback to HTTP polling if WebSocket is not available
                refreshData();
            }
        }, state.refreshRate);
    }

    /**
     * Refresh data via HTTP (fallback)
     */
    async function refreshData() {
        try {
            const response = await fetch('/api/v1/miners/status');
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.data) {
                    data.data.forEach(miner => updateMinerData(miner));
                }
            }
        } catch (error) {
            console.error('Failed to refresh data:', error);
        }
    }

    /**
     * Update miner data in UI
     */
    function updateMinerData(minerData) {
        state.miners.set(minerData.ip, minerData);
        
        // Update miner card if it exists
        const card = document.querySelector(`[data-miner-ip="${minerData.ip}"]`);
        if (card) {
            updateMinerCard(card, minerData);
        }

        // Update charts if they exist
        const chartId = `chart-${minerData.ip.replace(/\./g, '-')}`;
        if (state.charts.has(chartId)) {
            updateChart(chartId, minerData);
        }
    }

    /**
     * Update miner card display
     */
    function updateMinerCard(card, data) {
        // Update status
        card.className = card.className.replace(/status-\w+/, '');
        if (data.temp >= 75) {
            card.classList.add('status-error');
        } else if (data.temp >= 70) {
            card.classList.add('status-warning');
        } else if (data.hashRate > 0) {
            card.classList.add('status-online');
        } else {
            card.classList.add('status-offline');
        }

        // Update values with animation
        const updates = [
            { selector: '.hashrate-value', value: data.hashRate?.toFixed(2) || '0.00', unit: ' GH/s' },
            { selector: '.temperature-value', value: data.temp?.toFixed(1) || '0.0', unit: '°C' },
            { selector: '.power-value', value: data.power?.toFixed(1) || '0.0', unit: 'W' },
            { selector: '.efficiency-value', value: data.power > 0 ? (data.hashRate / data.power).toFixed(2) : '0.00', unit: ' GH/W' },
            { selector: '.frequency-value', value: data.frequency || '0', unit: ' MHz' },
            { selector: '.voltage-value', value: data.coreVoltage || '0', unit: ' mV' }
        ];

        updates.forEach(update => {
            const element = card.querySelector(update.selector);
            if (element) {
                animateValue(element, update.value + update.unit);
            }
        });

        // Update timestamp
        const timestampElement = card.querySelector('.last-update');
        if (timestampElement) {
            timestampElement.textContent = new Date().toLocaleTimeString();
        }
    }

    /**
     * Animate value changes
     */
    function animateValue(element, newValue) {
        if (!state.animationsEnabled) {
            element.textContent = newValue;
            return;
        }

        element.style.opacity = '0.5';
        setTimeout(() => {
            element.textContent = newValue;
            element.style.opacity = '1';
        }, 150);
    }

    /**
     * Update connection status indicator
     */
    function updateConnectionStatus(status) {
        if (!elements.connectionStatus) return;

        elements.connectionStatus.className = 'status-indicator';
        elements.connectionStatus.classList.add(`status-${status}`);

        let title = 'Unknown';
        switch (status) {
            case 'online':
                title = 'Connected - Real-time data';
                break;
            case 'offline':
                title = 'Disconnected - Using cached data';
                break;
            case 'error':
                title = 'Connection error';
                break;
        }
        elements.connectionStatus.title = title;
    }

    /**
     * Show alert notification
     */
    function showAlert(message, type = 'info', duration = 5000) {
        if (!elements.alertContainer) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible animate-slide-down`;
        alert.innerHTML = `
            <div class="alert-content">
                <i data-lucide="${getAlertIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="alert-close" onclick="this.parentElement.remove()">
                <i data-lucide="x"></i>
            </button>
        `;

        elements.alertContainer.appendChild(alert);
        
        // Initialize icons in the new alert
        if (window.lucide) {
            window.lucide.createIcons({ nameAttr: 'data-lucide' });
        }

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (alert.parentElement) {
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 300);
                }
            }, duration);
        }
    }

    /**
     * Get icon for alert type
     */
    function getAlertIcon(type) {
        const icons = {
            success: 'check-circle',
            warning: 'alert-triangle',
            error: 'x-circle',
            danger: 'x-circle',
            info: 'info'
        };
        return icons[type] || 'info';
    }

    /**
     * Toggle mobile menu
     */
    function toggleMobileMenu() {
        if (!elements.mobileNavPanel) return;
        
        const isOpen = elements.mobileNavPanel.classList.contains('show');
        if (isOpen) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    }

    /**
     * Open mobile menu
     */
    function openMobileMenu() {
        if (elements.mobileNavPanel) {
            elements.mobileNavPanel.classList.add('show');
        }
    }

    /**
     * Close mobile menu
     */
    function closeMobileMenu() {
        if (elements.mobileNavPanel) {
            elements.mobileNavPanel.classList.remove('show');
        }
    }

    /**
     * Toggle settings panel
     */
    function toggleSettings() {
        if (!elements.settingsPanel) return;
        
        const isOpen = elements.settingsPanel.classList.contains('show');
        if (isOpen) {
            closeSettings();
        } else {
            openSettings();
        }
    }

    /**
     * Open settings panel
     */
    function openSettings() {
        if (elements.settingsPanel) {
            elements.settingsPanel.classList.add('show');
        }
    }

    /**
     * Close settings panel
     */
    function closeSettings() {
        if (elements.settingsPanel) {
            elements.settingsPanel.classList.remove('show');
        }
    }

    /**
     * Show theme selector
     */
    function showThemeSelector() {
        // For now, just cycle through themes
        const themes = ['cyber-dark', 'neon-blue', 'mining-green'];
        const currentIndex = themes.indexOf(state.theme);
        const nextIndex = (currentIndex + 1) % themes.length;
        setTheme(themes[nextIndex]);
    }

    /**
     * Set dashboard theme
     */
    function setTheme(themeName) {
        // Remove old theme classes
        document.body.className = document.body.className.replace(/theme-\w+/g, '');
        
        // Add new theme class
        if (themeName !== 'cyber-dark') {
            document.body.classList.add(`theme-${themeName}`);
        }

        state.theme = themeName;
        state.settings.theme = themeName;
        saveSettings();

        // Update theme selector UI
        document.querySelectorAll('.theme-option').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === themeName);
        });

        showAlert(`Theme changed to ${themeName.replace('-', ' ')}`, 'success', 2000);
    }

    /**
     * Toggle FAB menu
     */
    function toggleFabMenu() {
        if (!elements.fabMenu) return;
        
        const isOpen = elements.fabMenu.classList.contains('show');
        if (isOpen) {
            closeFabMenu();
        } else {
            openFabMenu();
        }
    }

    /**
     * Open FAB menu
     */
    function openFabMenu() {
        if (elements.fabMenu) {
            elements.fabMenu.classList.add('show');
        }
    }

    /**
     * Close FAB menu
     */
    function closeFabMenu() {
        if (elements.fabMenu) {
            elements.fabMenu.classList.remove('show');
        }
    }

    /**
     * Update refresh rate
     */
    function updateRefreshRate() {
        if (!elements.refreshRateSelect) return;
        
        state.refreshRate = parseInt(elements.refreshRateSelect.value);
        state.settings.refreshRate = state.refreshRate;
        saveSettings();
        
        showAlert(`Refresh rate updated to ${state.refreshRate / 1000}s`, 'success', 2000);
    }

    /**
     * Toggle animations
     */
    function toggleAnimations() {
        if (!elements.animationsToggle) return;
        
        state.animationsEnabled = elements.animationsToggle.checked;
        state.settings.animationsEnabled = state.animationsEnabled;
        saveSettings();
        
        document.body.classList.toggle('animations-disabled', !state.animationsEnabled);
        
        showAlert(`Animations ${state.animationsEnabled ? 'enabled' : 'disabled'}`, 'success', 2000);
    }

    /**
     * Update events count badge
     */
    function updateEventsCount(count) {
        if (elements.eventsBadge) {
            elements.eventsBadge.textContent = count;
            elements.eventsBadge.style.display = count > 0 ? 'flex' : 'none';
        }
    }

    /**
     * Handle window resize
     */
    function handleResize() {
        // Close mobile menu on resize to desktop
        if (window.innerWidth >= 768) {
            closeMobileMenu();
        }
        
        // Refresh charts if they exist
        state.charts.forEach((chart, id) => {
            if (chart && chart.resize) {
                chart.resize();
            }
        });
    }

    /**
     * Handle visibility change (pause/resume when tab not visible)
     */
    function handleVisibilityChange() {
        if (document.hidden) {
            // Tab is not visible, could reduce refresh rate or pause animations
            console.log('Dashboard hidden, reducing activity');
        } else {
            // Tab is visible again, resume normal operation
            console.log('Dashboard visible, resuming normal activity');
            refreshData(); // Get latest data
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    function handleKeyboardShortcuts(e) {
        // Only handle shortcuts when not in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (e.key) {
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    toggleSettings();
                }
                break;
            case 'm':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    toggleMobileMenu();
                }
                break;
            case 't':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    showThemeSelector();
                }
                break;
            case 'Escape':
                closeSettings();
                closeMobileMenu();
                closeFabMenu();
                break;
        }
    }

    /**
     * Debounce function for performance
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Public API
     */
    return {
        init,
        showAlert,
        updateMinerData,
        setTheme,
        getState: () => state,
        refreshData
    };

})();

// Global functions for template usage
window.startQuickBenchmark = function() {
    window.DashboardCore.showAlert('Quick benchmark feature coming soon!', 'info');
};

window.exportData = function() {
    window.DashboardCore.showAlert('Data export feature coming soon!', 'info');
};

window.closeSettings = function() {
    const settingsPanel = document.getElementById('settings-panel');
    if (settingsPanel) {
        settingsPanel.classList.remove('show');
    }
};