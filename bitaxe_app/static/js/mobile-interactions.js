/**
 * BitAxe Mobile Interactions
 * Touch-optimized interactions and mobile-specific features
 */

class MobileInteractions {
    constructor() {
        this.isMobile = this.detectMobile();
        this.isTouch = 'ontouchstart' in window;
        this.fabMenu = null;
        this.lastScrollY = 0;
        this.ticking = false;
        
        this.init();
    }
    
    init() {
        console.log('🔥 Initializing Mobile Interactions...');
        
        // Initialize mobile-specific features
        this.initTouchGestures();
        this.initFabMenu();
        this.initPullToRefresh();
        this.initScrollOptimization();
        this.initMobileNavigation();
        this.initToastNotifications();
        this.initKeyboardHandling();
        
        // Add mobile-specific CSS classes
        document.body.classList.toggle('is-mobile', this.isMobile);
        document.body.classList.toggle('is-touch', this.isTouch);
        
        console.log('✅ Mobile Interactions initialized');
    }
    
    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
               (window.innerWidth <= 768);
    }
    
    initTouchGestures() {
        // Add touch feedback to interactive elements
        const touchElements = document.querySelectorAll('.btn, .nav-link, .tab-item, .metric-card, .card');
        
        touchElements.forEach(element => {
            element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
            element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });
        });
        
        // Swipe gestures for navigation
        let startX, startY, startTime;
        
        document.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            startX = touch.clientX;
            startY = touch.clientY;
            startTime = Date.now();
        }, { passive: true });
        
        document.addEventListener('touchend', (e) => {
            if (!startX || !startY) return;
            
            const touch = e.changedTouches[0];
            const endX = touch.clientX;
            const endY = touch.clientY;
            const endTime = Date.now();
            
            const deltaX = endX - startX;
            const deltaY = endY - startY;
            const deltaTime = endTime - startTime;
            
            // Only consider fast swipes
            if (deltaTime > 300) return;
            
            // Minimum swipe distance
            if (Math.abs(deltaX) < 50 && Math.abs(deltaY) < 50) return;
            
            // Horizontal swipe
            if (Math.abs(deltaX) > Math.abs(deltaY)) {
                if (deltaX > 0) {
                    this.handleSwipeRight();
                } else {
                    this.handleSwipeLeft();
                }
            }
            // Vertical swipe
            else {
                if (deltaY > 0) {
                    this.handleSwipeDown();
                } else {
                    this.handleSwipeUp();
                }
            }
            
            startX = null;
            startY = null;
        }, { passive: true });
    }
    
    handleTouchStart(e) {
        e.currentTarget.classList.add('touch-active');
    }
    
    handleTouchEnd(e) {
        setTimeout(() => {
            e.currentTarget.classList.remove('touch-active');
        }, 150);
    }
    
    handleSwipeLeft() {
        // Could navigate to next page or open menu
        console.log('Swipe left detected');
    }
    
    handleSwipeRight() {
        // Could navigate to previous page or close menu
        const mobileMenu = document.getElementById('mobile-nav-panel');
        if (mobileMenu && !mobileMenu.classList.contains('active')) {
            this.toggleMobileMenu();
        }
    }
    
    handleSwipeDown() {
        // Pull to refresh or show notifications
        if (window.scrollY === 0) {
            this.triggerPullToRefresh();
        }
    }
    
    handleSwipeUp() {
        // Could hide UI elements or show quick actions
        this.hideFabMenu();
    }
    
    initFabMenu() {
        const fab = document.getElementById('quick-actions-fab');
        const fabMenu = document.getElementById('fab-menu');
        
        if (!fab || !fabMenu) return;
        
        this.fabMenu = fabMenu;
        
        fab.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleFabMenu();
        });
        
        // Close FAB menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!fab.contains(e.target) && !fabMenu.contains(e.target)) {
                this.hideFabMenu();
            }
        });
        
        // Close FAB menu on scroll
        window.addEventListener('scroll', () => {
            this.hideFabMenu();
        }, { passive: true });
    }
    
    toggleFabMenu() {
        if (this.fabMenu) {
            const isActive = this.fabMenu.classList.contains('active');
            this.fabMenu.classList.toggle('active', !isActive);
            
            // Rotate FAB icon
            const fabIcon = document.querySelector('#quick-actions-fab i');
            if (fabIcon) {
                fabIcon.style.transform = isActive ? 'rotate(0deg)' : 'rotate(45deg)';
            }
        }
    }
    
    hideFabMenu() {
        if (this.fabMenu && this.fabMenu.classList.contains('active')) {
            this.fabMenu.classList.remove('active');
            
            // Reset FAB icon
            const fabIcon = document.querySelector('#quick-actions-fab i');
            if (fabIcon) {
                fabIcon.style.transform = 'rotate(0deg)';
            }
        }
    }
    
    initPullToRefresh() {
        let startY = 0;
        let currentY = 0;
        let pulling = false;
        
        const refreshThreshold = 100;
        const refreshElement = this.createPullToRefreshElement();
        
        document.addEventListener('touchstart', (e) => {
            if (window.scrollY === 0) {
                startY = e.touches[0].clientY;
                pulling = true;
            }
        }, { passive: true });
        
        document.addEventListener('touchmove', (e) => {
            if (!pulling) return;
            
            currentY = e.touches[0].clientY;
            const pullDistance = currentY - startY;
            
            if (pullDistance > 0 && window.scrollY === 0) {
                e.preventDefault();
                
                const pullRatio = Math.min(pullDistance / refreshThreshold, 1);
                refreshElement.style.transform = `translateY(${pullDistance * 0.5}px)`;
                refreshElement.style.opacity = pullRatio;
                
                if (pullDistance > refreshThreshold) {
                    refreshElement.classList.add('ready');
                } else {
                    refreshElement.classList.remove('ready');
                }
            }
        });
        
        document.addEventListener('touchend', () => {
            if (!pulling) return;
            
            const pullDistance = currentY - startY;
            
            if (pullDistance > refreshThreshold) {
                this.triggerPullToRefresh();
            }
            
            // Reset
            refreshElement.style.transform = 'translateY(-100%)';
            refreshElement.style.opacity = '0';
            refreshElement.classList.remove('ready');
            
            pulling = false;
            startY = 0;
            currentY = 0;
        });
    }
    
    createPullToRefreshElement() {
        const refreshElement = document.createElement('div');
        refreshElement.className = 'pull-to-refresh';
        refreshElement.innerHTML = `
            <div class="refresh-icon">
                <i data-lucide="refresh-cw"></i>
            </div>
            <div class="refresh-text">Pull to refresh</div>
        `;
        
        // Add styles
        refreshElement.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: var(--card-bg);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            color: var(--text-secondary);
            transform: translateY(-100%);
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 999;
            border-bottom: 1px solid var(--border-color);
        `;
        
        document.body.appendChild(refreshElement);
        
        // Initialize lucide icons for the refresh element
        if (window.lucide) {
            window.lucide.createIcons();
        }
        
        return refreshElement;
    }
    
    triggerPullToRefresh() {
        console.log('🔄 Pull to refresh triggered');
        
        // Show loading state
        this.showToast('Refreshing...', 'info');
        
        // Trigger page refresh after a short delay
        setTimeout(() => {
            window.location.reload();
        }, 500);
    }
    
    initScrollOptimization() {
        // Hide/show navigation on scroll
        window.addEventListener('scroll', () => {
            if (!this.ticking) {
                requestAnimationFrame(() => {
                    this.updateScrollPosition();
                    this.ticking = false;
                });
                this.ticking = true;
            }
        }, { passive: true });
    }
    
    updateScrollPosition() {
        const currentScrollY = window.scrollY;
        const navbar = document.getElementById('main-navbar');
        const bottomTabBar = document.getElementById('bottom-tab-bar');
        
        if (currentScrollY > this.lastScrollY && currentScrollY > 100) {
            // Scrolling down - hide navigation
            if (navbar) navbar.style.transform = 'translateY(-100%)';
            if (bottomTabBar) bottomTabBar.style.transform = 'translateY(100%)';
        } else {
            // Scrolling up - show navigation
            if (navbar) navbar.style.transform = 'translateY(0)';
            if (bottomTabBar) bottomTabBar.style.transform = 'translateY(0)';
        }
        
        this.lastScrollY = currentScrollY;
    }
    
    initMobileNavigation() {
        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const mobileNavPanel = document.getElementById('mobile-nav-panel');
        
        if (mobileMenuToggle && mobileNavPanel) {
            mobileMenuToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!mobileNavPanel.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
                    this.closeMobileMenu();
                }
            });
        }
        
        // Handle navbar toggle for realtime template
        const navbarToggle = document.getElementById('navbar-toggle');
        const navbarMenu = document.getElementById('navbar-menu');
        
        if (navbarToggle && navbarMenu) {
            navbarToggle.addEventListener('click', () => {
                navbarMenu.classList.toggle('active');
            });
        }
    }
    
    toggleMobileMenu() {
        const mobileNavPanel = document.getElementById('mobile-nav-panel');
        if (mobileNavPanel) {
            mobileNavPanel.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        }
    }
    
    closeMobileMenu() {
        const mobileNavPanel = document.getElementById('mobile-nav-panel');
        if (mobileNavPanel) {
            mobileNavPanel.classList.remove('active');
            document.body.classList.remove('menu-open');
        }
    }
    
    initToastNotifications() {
        // Create toast container if it doesn't exist
        if (!document.getElementById('toast-container')) {
            const toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.cssText = `
                position: fixed;
                top: 80px;
                left: 1rem;
                right: 1rem;
                z-index: 10001;
                pointer-events: none;
            `;
            document.body.appendChild(toastContainer);
        }
    }
    
    showToast(message, type = 'info', duration = 3000) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i data-lucide="${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add styles
        toast.style.cssText = `
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-left: 4px solid ${this.getToastColor(type)};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transform: translateX(100%);
            transition: transform 0.3s ease;
            pointer-events: all;
        `;
        
        toast.querySelector('.toast-content').style.cssText = `
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-primary);
        `;
        
        toastContainer.appendChild(toast);
        
        // Initialize lucide icons
        if (window.lucide) {
            window.lucide.createIcons();
        }
        
        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(0)';
        });
        
        // Auto remove
        setTimeout(() => {
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
        
        return toast;
    }
    
    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'x-circle',
            warning: 'alert-triangle',
            info: 'info'
        };
        return icons[type] || 'info';
    }
    
    getToastColor(type) {
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#3b82f6'
        };
        return colors[type] || '#3b82f6';
    }
    
    initKeyboardHandling() {
        // Handle virtual keyboard on mobile
        if (this.isMobile) {
            let initialViewportHeight = window.innerHeight;
            
            window.addEventListener('resize', () => {
                const currentViewportHeight = window.innerHeight;
                const heightDifference = initialViewportHeight - currentViewportHeight;
                
                // Keyboard is likely open if height difference is significant
                if (heightDifference > 150) {
                    document.body.classList.add('keyboard-open');
                    
                    // Adjust bottom tab bar
                    const bottomTabBar = document.getElementById('bottom-tab-bar');
                    if (bottomTabBar) {
                        bottomTabBar.style.display = 'none';
                    }
                } else {
                    document.body.classList.remove('keyboard-open');
                    
                    // Restore bottom tab bar
                    const bottomTabBar = document.getElementById('bottom-tab-bar');
                    if (bottomTabBar) {
                        bottomTabBar.style.display = 'flex';
                    }
                }
            });
        }
        
        // Prevent zoom on input focus (iOS Safari)
        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('focus', () => {
                if (this.isMobile) {
                    // Scroll input into view
                    setTimeout(() => {
                        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 300);
                }
            });
        });
    }
    
    // Utility methods for external use
    vibrate(pattern = [100]) {
        if ('vibrate' in navigator) {
            navigator.vibrate(pattern);
        }
    }
    
    showSuccessToast(message) {
        this.showToast(message, 'success');
        this.vibrate([50]);
    }
    
    showErrorToast(message) {
        this.showToast(message, 'error');
        this.vibrate([100, 50, 100]);
    }
    
    isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }
    
    smoothScrollTo(element, offset = 0) {
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition = elementPosition - offset;
        
        window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
        });
    }
}

// Global functions for external use
window.showNotification = function(message, type = 'info') {
    if (window.mobileInteractions) {
        window.mobileInteractions.showToast(message, type);
    }
};

window.startQuickBenchmark = function() {
    window.showNotification('Quick benchmark feature coming soon!', 'info');
};

window.exportData = function() {
    window.showNotification('Data export feature coming soon!', 'info');
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.mobileInteractions = new MobileInteractions();
});

// Add CSS for touch states and mobile-specific styles
const mobileStyles = document.createElement('style');
mobileStyles.textContent = `
    .touch-active {
        transform: scale(0.98);
        opacity: 0.8;
        transition: all 0.1s ease;
    }
    
    .is-mobile .navbar {
        transition: transform 0.3s ease;
    }
    
    .is-mobile .bottom-tab-bar {
        transition: transform 0.3s ease;
    }
    
    .menu-open {
        overflow: hidden;
    }
    
    .keyboard-open .fab-container {
        display: none;
    }
    
    .pull-to-refresh.ready .refresh-icon i {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;

document.head.appendChild(mobileStyles);