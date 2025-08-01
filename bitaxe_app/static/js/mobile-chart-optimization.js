/**
 * BitAxe Mobile Chart Optimization
 * Optimizes Chart.js charts for mobile devices and touch interfaces
 */

class MobileChartOptimizer {
    constructor() {
        this.isMobile = this.detectMobile();
        this.charts = new Map();
        
        this.init();
    }
    
    init() {
        console.log('📱 Initializing Mobile Chart Optimization...');
        
        // Set default mobile optimizations for Chart.js
        this.setDefaultChartOptions();
        
        // Monitor for chart creation
        this.observeChartCreation();
        
        // Handle orientation changes
        this.handleOrientationChange();
        
        console.log('✅ Mobile Chart Optimization initialized');
    }
    
    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
               (window.innerWidth <= 768);
    }
    
    setDefaultChartOptions() {
        if (typeof Chart !== 'undefined') {
            // Mobile-optimized default options
            Chart.defaults.responsive = true;
            Chart.defaults.maintainAspectRatio = false;
            
            // Optimize animations for mobile
            if (this.isMobile) {
                Chart.defaults.animation = {
                    duration: 300, // Faster animations
                    easing: 'easeOutQuart'
                };
                
                // Optimize interaction
                Chart.defaults.interaction = {
                    intersect: false,
                    mode: 'nearest'
                };
            }
        }
    }
    
    observeChartCreation() {
        // Override Chart constructor to apply mobile optimizations
        if (typeof Chart !== 'undefined') {
            const originalChart = Chart;
            const self = this;
            
            Chart = function(ctx, config) {
                // Apply mobile optimizations to config
                config = self.optimizeChartConfig(config);
                
                // Create chart with optimized config
                const chart = new originalChart(ctx, config);
                
                // Store reference and apply post-creation optimizations
                self.charts.set(chart.id, chart);
                self.applyMobileOptimizations(chart);
                
                return chart;
            };
            
            // Copy static properties
            Object.setPrototypeOf(Chart, originalChart);
            Object.assign(Chart, originalChart);
        }
    }
    
    optimizeChartConfig(config) {
        if (!this.isMobile) return config;
        
        // Clone config to avoid mutations
        const optimizedConfig = JSON.parse(JSON.stringify(config));
        
        // Optimize options for mobile
        if (!optimizedConfig.options) {
            optimizedConfig.options = {};
        }
        
        // Responsive settings
        optimizedConfig.options.responsive = true;
        optimizedConfig.options.maintainAspectRatio = false;
        
        // Optimize plugins
        if (!optimizedConfig.options.plugins) {
            optimizedConfig.options.plugins = {};
        }
        
        // Legend optimization
        optimizedConfig.options.plugins.legend = {
            ...optimizedConfig.options.plugins.legend,
            display: true,
            position: 'bottom', // Better for mobile
            labels: {
                ...optimizedConfig.options.plugins.legend?.labels,
                usePointStyle: true,
                padding: 15,
                boxWidth: 12,
                font: {
                    size: 12
                }
            }
        };
        
        // Tooltip optimization
        optimizedConfig.options.plugins.tooltip = {
            ...optimizedConfig.options.plugins.tooltip,
            enabled: true,
            mode: 'nearest',
            intersect: false,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleFont: { size: 14 },
            bodyFont: { size: 12 },
            padding: 10,
            cornerRadius: 6,
            displayColors: true,
            callbacks: {
                ...optimizedConfig.options.plugins.tooltip?.callbacks,
                // Custom callback for better mobile formatting
                label: function(context) {
                    let label = context.dataset.label || '';
                    if (label) {
                        label += ': ';
                    }
                    
                    // Format number based on type
                    const value = context.parsed.y;
                    if (typeof value === 'number') {
                        if (value > 1000) {
                            label += (value / 1000).toFixed(2) + 'k';
                        } else {
                            label += value.toFixed(2);
                        }
                    } else {
                        label += value;
                    }
                    
                    return label;
                }
            }
        };
        
        // Scale optimization
        if (!optimizedConfig.options.scales) {
            optimizedConfig.options.scales = {};
        }
        
        // X-axis optimization
        optimizedConfig.options.scales.x = {
            ...optimizedConfig.options.scales.x,
            grid: {
                display: false // Cleaner look on mobile
            },
            ticks: {
                ...optimizedConfig.options.scales.x?.ticks,
                maxTicksLimit: 6, // Limit ticks on mobile
                font: { size: 10 },
                maxRotation: 45,
                minRotation: 0
            }
        };
        
        // Y-axis optimization
        optimizedConfig.options.scales.y = {
            ...optimizedConfig.options.scales.y,
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                ...optimizedConfig.options.scales.y?.ticks,
                maxTicksLimit: 5,
                font: { size: 10 },
                callback: function(value) {
                    // Format large numbers
                    if (value >= 1000000) {
                        return (value / 1000000).toFixed(1) + 'M';
                    } else if (value >= 1000) {
                        return (value / 1000).toFixed(1) + 'k';
                    }
                    return value;
                }
            }
        };
        
        // Animation optimization
        optimizedConfig.options.animation = {
            duration: 300,
            easing: 'easeOutQuart'
        };
        
        // Interaction optimization
        optimizedConfig.options.interaction = {
            intersect: false,
            mode: 'nearest'
        };
        
        return optimizedConfig;
    }
    
    applyMobileOptimizations(chart) {
        if (!this.isMobile) return;
        
        const canvas = chart.canvas;
        const container = canvas.parentElement;
        
        // Add touch-friendly styles
        canvas.style.touchAction = 'pan-y';
        
        // Optimize container
        if (container) {
            container.style.position = 'relative';
            container.style.height = this.getOptimalChartHeight(chart);
            container.style.userSelect = 'none';
        }
        
        // Add touch event handlers
        this.addTouchHandlers(chart);
        
        // Optimize for retina displays
        this.optimizeForRetina(chart);
    }
    
    getOptimalChartHeight(chart) {
        const screenHeight = window.innerHeight;
        const isLandscape = window.innerWidth > window.innerHeight;
        
        // Adjust height based on chart type and screen size
        let height = '250px'; // Default mobile height
        
        if (chart.config.type === 'line' || chart.config.type === 'bar') {
            height = isLandscape ? '200px' : '250px';
        } else if (chart.config.type === 'doughnut' || chart.config.type === 'pie') {
            height = '200px';
        }
        
        // Adjust for very small screens
        if (screenHeight < 600) {
            height = '180px';
        }
        
        return height;
    }
    
    addTouchHandlers(chart) {
        const canvas = chart.canvas;
        
        // Prevent default touch behaviors that might interfere
        canvas.addEventListener('touchstart', (e) => {
            // Allow single touch for interaction
            if (e.touches.length === 1) {
                return;
            }
            e.preventDefault();
        }, { passive: false });
        
        // Handle touch move for better interaction
        canvas.addEventListener('touchmove', (e) => {
            if (e.touches.length > 1) {
                e.preventDefault();
            }
        }, { passive: false });
        
        // Add tap feedback
        let tapTimeout;
        canvas.addEventListener('touchstart', () => {
            canvas.style.opacity = '0.8';
            clearTimeout(tapTimeout);
        });
        
        canvas.addEventListener('touchend', () => {
            tapTimeout = setTimeout(() => {
                canvas.style.opacity = '1';
            }, 100);
        });
    }
    
    optimizeForRetina(chart) {
        const canvas = chart.canvas;
        const ctx = canvas.getContext('2d');
        const devicePixelRatio = window.devicePixelRatio || 1;
        
        if (devicePixelRatio > 1) {
            // Get the size the canvas should be displayed at
            const rect = canvas.getBoundingClientRect();
            
            // Set the internal size to the device pixel ratio
            canvas.width = rect.width * devicePixelRatio;
            canvas.height = rect.height * devicePixelRatio;
            
            // Scale the canvas back down using CSS
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            // Scale the drawing context so everything draws at the correct size
            ctx.scale(devicePixelRatio, devicePixelRatio);
        }
    }
    
    handleOrientationChange() {
        window.addEventListener('orientationchange', () => {
            // Delay to allow orientation change to complete
            setTimeout(() => {
                this.resizeAllCharts();
            }, 300);
        });
        
        // Also handle window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.resizeAllCharts();
            }, 150);
        });
    }
    
    resizeAllCharts() {
        this.charts.forEach((chart) => {
            try {
                // Update container height
                const container = chart.canvas.parentElement;
                if (container) {
                    container.style.height = this.getOptimalChartHeight(chart);
                }
                
                // Trigger chart resize
                chart.resize();
            } catch (error) {
                console.warn('Error resizing chart:', error);
            }
        });
    }
    
    // Utility methods for external use
    optimizeExistingChart(chart) {
        if (this.isMobile && chart) {
            this.applyMobileOptimizations(chart);
            this.charts.set(chart.id, chart);
        }
    }
    
    createMobileOptimizedChart(ctx, config) {
        if (typeof Chart !== 'undefined') {
            return new Chart(ctx, this.optimizeChartConfig(config));
        }
        return null;
    }
    
    destroyChart(chartId) {
        if (this.charts.has(chartId)) {
            const chart = this.charts.get(chartId);
            chart.destroy();
            this.charts.delete(chartId);
        }
    }
    
    // Get chart-specific mobile options
    getMobileChartOptions(chartType) {
        const baseOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: this.isMobile ? 300 : 1000
            }
        };
        
        switch (chartType) {
            case 'line':
                return {
                    ...baseOptions,
                    elements: {
                        point: {
                            radius: 2,
                            hoverRadius: 4
                        },
                        line: {
                            tension: 0.1
                        }
                    }
                };
                
            case 'bar':
                return {
                    ...baseOptions,
                    barPercentage: 0.8,
                    categoryPercentage: 0.9
                };
                
            case 'doughnut':
            case 'pie':
                return {
                    ...baseOptions,
                    cutout: chartType === 'doughnut' ? '60%' : 0,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 15,
                                usePointStyle: true
                            }
                        }
                    }
                };
                
            default:
                return baseOptions;
        }
    }
}

// Global instance
window.mobileChartOptimizer = new MobileChartOptimizer();

// Utility functions for external use
window.createMobileChart = function(ctx, config) {
    return window.mobileChartOptimizer.createMobileOptimizedChart(ctx, config);
};

window.optimizeChart = function(chart) {
    return window.mobileChartOptimizer.optimizeExistingChart(chart);
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Apply optimizations to any existing charts
    if (typeof Chart !== 'undefined' && Chart.instances) {
        Object.values(Chart.instances).forEach(chart => {
            window.mobileChartOptimizer.optimizeExistingChart(chart);
        });
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MobileChartOptimizer;
}