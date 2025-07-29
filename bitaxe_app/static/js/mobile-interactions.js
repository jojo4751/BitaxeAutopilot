/**
 * BitAxe Mobile Interactions
 * Touch gestures and mobile-specific interactions
 */

window.MobileInteractions = (function() {
    'use strict';

    let isTouch = false;
    let touchStartPos = { x: 0, y: 0 };
    let touchEndPos = { x: 0, y: 0 };
    let swipeThreshold = 50;
    let tapTimeout = null;
    let doubleTapTimeout = null;
    let longPressTimeout = null;
    let lastTap = 0;

    /**
     * Initialize mobile interactions
     */
    function init() {
        detectTouchDevice();
        setupTouchEvents();
        setupGestureRecognition();
        setupHapticFeedback();
        setupViewportHandling();
        
        console.log('Mobile Interactions initialized');
    }

    /**
     * Detect if device supports touch
     */
    function detectTouchDevice() {
        isTouch = 'ontouchstart' in window || 
                  navigator.maxTouchPoints > 0 || 
                  navigator.msMaxTouchPoints > 0;
        
        if (isTouch) {
            document.body.classList.add('touch-device');
        }
    }

    /**
     * Setup basic touch events
     */
    function setupTouchEvents() {
        // Prevent default touch behavior on specific elements
        document.addEventListener('touchstart', handleTouchStart, { passive: false });
        document.addEventListener('touchmove', handleTouchMove, { passive: false });
        document.addEventListener('touchend', handleTouchEnd, { passive: false });

        // Add touch feedback to interactive elements
        addTouchFeedback();
    }

    /**
     * Handle touch start
     */
    function handleTouchStart(e) {
        const touch = e.touches[0];
        touchStartPos = { x: touch.clientX, y: touch.clientY };

        // Long press detection
        const element = e.target.closest('[data-long-press]');
        if (element) {
            longPressTimeout = setTimeout(() => {
                handleLongPress(element, e);
            }, 500);
        }

        // Prevent unwanted behaviors
        if (e.target.closest('.no-touch-action')) {
            e.preventDefault();
        }
    }

    /**
     * Handle touch move
     */
    function handleTouchMove(e) {
        // Clear long press if user moves finger
        if (longPressTimeout) {
            clearTimeout(longPressTimeout);
            longPressTimeout = null;
        }

        const touch = e.touches[0];
        const deltaX = Math.abs(touch.clientX - touchStartPos.x);
        const deltaY = Math.abs(touch.clientY - touchStartPos.y);

        // Handle swipe cards
        const swipeCard = e.target.closest('.swipe-card');
        if (swipeCard && (deltaX > 10 || deltaY > 10)) {
            handleCardSwipe(swipeCard, touch.clientX - touchStartPos.x, touch.clientY - touchStartPos.y);
        }

        // Handle chart interactions
        const chart = e.target.closest('.chart-container');
        if (chart && deltaX > 10) {
            e.preventDefault(); // Prevent page scroll while interacting with chart
        }
    }

    /**
     * Handle touch end
     */
    function handleTouchEnd(e) {
        // Clear long press timeout
        if (longPressTimeout) {
            clearTimeout(longPressTimeout);
            longPressTimeout = null;
        }

        const touch = e.changedTouches[0];
        touchEndPos = { x: touch.clientX, y: touch.clientY };

        // Calculate swipe direction and distance
        const deltaX = touchEndPos.x - touchStartPos.x;
        const deltaY = touchEndPos.y - touchStartPos.y;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

        // Handle swipes
        if (distance > swipeThreshold) {
            handleSwipe(deltaX, deltaY, e.target);
        } else {
            // Handle taps
            handleTap(e);
        }
    }

    /**
     * Handle swipe gestures
     */
    function handleSwipe(deltaX, deltaY, target) {
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        // Determine swipe direction
        let direction = '';
        if (absX > absY) {
            direction = deltaX > 0 ? 'right' : 'left';
        } else {
            direction = deltaY > 0 ? 'down' : 'up';
        }

        // Handle specific swipe contexts
        const context = target.closest('[data-swipe]') || target.closest('.swipe-container');
        if (context) {
            handleContextualSwipe(context, direction, deltaX, deltaY);
        }

        // Global swipe handlers
        handleGlobalSwipe(direction, target);
    }

    /**
     * Handle contextual swipes
     */
    function handleContextualSwipe(context, direction, deltaX, deltaY) {
        const swipeType = context.dataset.swipe || context.classList.contains('swipe-container');

        switch (swipeType) {
            case 'miner-card':
                handleMinerCardSwipe(context, direction);
                break;
            case 'tab-navigation':
                handleTabSwipe(context, direction);
                break;
            case 'chart':
                handleChartSwipe(context, direction, deltaX);
                break;
            default:
                // Custom swipe handler
                if (context.dataset.swipeHandler) {
                    const handler = window[context.dataset.swipeHandler];
                    if (typeof handler === 'function') {
                        handler(direction, deltaX, deltaY);
                    }
                }
        }
    }

    /**
     * Handle global swipes
     */
    function handleGlobalSwipe(direction, target) {
        // Don't handle global swipes if target has specific handler
        if (target.closest('[data-swipe]') || 
            target.closest('.swipe-container') ||
            target.closest('input') ||
            target.closest('textarea')) {
            return;
        }

        switch (direction) {
            case 'right':
                // Swipe right to open mobile menu (if on left edge)
                if (touchStartPos.x < 50) {
                    openMobileMenu();
                }
                break;
            case 'left':
                // Swipe left to close mobile menu or settings
                closePanels();
                break;
            case 'down':
                // Swipe down to refresh (if at top of page)
                if (window.scrollY < 50) {
                    handlePullToRefresh();
                }
                break;
            case 'up':
                // Swipe up to show quick actions (if at bottom)
                if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50) {
                    showQuickActions();
                }
                break;
        }
    }

    /**
     * Handle miner card swipes
     */
    function handleMinerCardSwipe(card, direction) {
        switch (direction) {
            case 'left':
                // Swipe left to show actions
                showCardActions(card);
                break;
            case 'right':
                // Swipe right to hide actions or expand details
                if (card.classList.contains('actions-visible')) {
                    hideCardActions(card);
                } else {
                    expandCardDetails(card);
                }
                break;
            case 'up':
                // Swipe up to bookmark/favorite
                toggleCardBookmark(card);
                break;
            case 'down':
                // Swipe down to minimize/collapse
                collapseCardDetails(card);
                break;
        }
    }

    /**
     * Handle tab swipes
     */
    function handleTabSwipe(container, direction) {
        const tabs = container.querySelectorAll('.tab-item');
        const activeTab = container.querySelector('.tab-item.active');
        const currentIndex = Array.from(tabs).indexOf(activeTab);

        let newIndex = currentIndex;
        if (direction === 'left' && currentIndex < tabs.length - 1) {
            newIndex = currentIndex + 1;
        } else if (direction === 'right' && currentIndex > 0) {
            newIndex = currentIndex - 1;
        }

        if (newIndex !== currentIndex) {
            tabs[newIndex].click();
            vibrate(50); // Short vibration feedback
        }
    }

    /**
     * Handle chart swipes
     */
    function handleChartSwipe(chart, direction, deltaX) {
        // Implement chart panning/zooming
        const chartInstance = chart.chartInstance || chart._chart;
        if (chartInstance && chartInstance.pan) {
            chartInstance.pan(deltaX);
        }
    }

    /**
     * Handle tap events
     */
    function handleTap(e) {
        const now = Date.now();
        const timeDiff = now - lastTap;

        if (timeDiff < 300 && timeDiff > 0) {
            // Double tap
            handleDoubleTap(e);
            lastTap = 0; // Reset to prevent triple tap
        } else {
            // Single tap
            if (tapTimeout) {
                clearTimeout(tapTimeout);
            }
            
            tapTimeout = setTimeout(() => {
                handleSingleTap(e);
                tapTimeout = null;
            }, 300);
            
            lastTap = now;
        }
    }

    /**
     * Handle single tap
     */
    function handleSingleTap(e) {
        const target = e.target;
        
        // Add ripple effect to buttons
        if (target.closest('.btn') || target.closest('.card') || target.closest('[data-ripple]')) {
            addRippleEffect(target, e);
        }
    }

    /**
     * Handle double tap
     */
    function handleDoubleTap(e) {
        const target = e.target;
        
        // Double tap to zoom on charts
        const chart = target.closest('.chart-container');
        if (chart) {
            toggleChartZoom(chart);
            return;
        }

        // Double tap to expand cards
        const card = target.closest('.card');
        if (card && !target.closest('button') && !target.closest('input')) {
            toggleCardExpansion(card);
            return;
        }

        // Double tap to toggle theme
        if (target.closest('.page-header')) {
            cycleTheme();
        }
    }

    /**
     * Handle long press
     */
    function handleLongPress(element, e) {
        vibrate(100); // Long vibration for long press
        
        const action = element.dataset.longPress;
        switch (action) {
            case 'context-menu':
                showContextMenu(element, e);
                break;
            case 'quick-edit':
                showQuickEdit(element);
                break;
            case 'bookmark':
                toggleBookmark(element);
                break;
            default:
                // Custom long press handler
                if (window[action] && typeof window[action] === 'function') {
                    window[action](element, e);
                }
        }
    }

    /**
     * Add touch feedback to interactive elements
     */
    function addTouchFeedback() {
        const interactiveElements = document.querySelectorAll('button, .btn, .card, [data-touch-feedback]');
        
        interactiveElements.forEach(element => {
            element.addEventListener('touchstart', function() {
                this.classList.add('touch-active');
            }, { passive: true });

            element.addEventListener('touchend', function() {
                setTimeout(() => {
                    this.classList.remove('touch-active');
                }, 150);
            }, { passive: true });

            element.addEventListener('touchcancel', function() {
                this.classList.remove('touch-active');
            }, { passive: true });
        });
    }

    /**
     * Add ripple effect
     */
    function addRippleEffect(element, event) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple-effect';
        
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            left: ${x}px;
            top: ${y}px;
            width: ${size}px;
            height: ${size}px;
            pointer-events: none;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    /**
     * Setup gesture recognition
     */
    function setupGestureRecognition() {
        // Add CSS for touch states
        const style = document.createElement('style');
        style.textContent = `
            .touch-active {
                transform: scale(0.98);
                opacity: 0.8;
                transition: all 0.1s ease;
            }

            @keyframes ripple {
                to {
                    transform: scale(2);
                    opacity: 0;
                }
            }

            .touch-device .hover-only {
                display: none !important;
            }

            .touch-device .btn {
                min-height: 48px;
                min-width: 48px;
            }

            .touch-device .form-input,
            .touch-device .form-select {
                min-height: 48px;
                font-size: 16px; /* Prevent zoom on iOS */
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Setup haptic feedback
     */
    function setupHapticFeedback() {
        // Check if device supports vibration
        if (!navigator.vibrate) {
            console.log('Vibration API not supported');
        }
    }

    /**
     * Vibrate device
     */
    function vibrate(duration = 50) {
        if (navigator.vibrate) {
            navigator.vibrate(duration);
        }
    }

    /**
     * Setup viewport handling
     */
    function setupViewportHandling() {
        // Handle viewport changes (keyboard show/hide on mobile)
        let initialViewportHeight = window.innerHeight;
        
        window.addEventListener('resize', () => {
            const currentViewportHeight = window.innerHeight;
            const heightDiff = initialViewportHeight - currentViewportHeight;
            
            if (heightDiff > 150) {
                // Keyboard is likely open
                document.body.classList.add('keyboard-open');
            } else {
                // Keyboard is likely closed
                document.body.classList.remove('keyboard-open');
            }
        });

        // Prevent zoom on double-tap for most elements
        document.addEventListener('touchend', (e) => {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
            }
        });
    }

    /**
     * Utility functions for swipe actions
     */
    function openMobileMenu() {
        const mobilePanel = document.getElementById('mobile-nav-panel');
        if (mobilePanel) {
            mobilePanel.classList.add('show');
            vibrate(25);
        }
    }

    function closePanels() {
        const panels = document.querySelectorAll('.mobile-nav-panel.show, .settings-panel.show, .fab-menu.show');
        panels.forEach(panel => panel.classList.remove('show'));
        if (panels.length > 0) {
            vibrate(25);
        }
    }

    function handlePullToRefresh() {
        if (window.DashboardCore && window.DashboardCore.refreshData) {
            window.DashboardCore.refreshData();
            window.DashboardCore.showAlert('Refreshing data...', 'info', 2000);
            vibrate(50);
        }
    }

    function showQuickActions() {
        const fab = document.getElementById('quick-actions-fab');
        if (fab) {
            fab.click();
            vibrate(50);
        }
    }

    function showCardActions(card) {
        card.classList.add('actions-visible');
        vibrate(25);
    }

    function hideCardActions(card) {
        card.classList.remove('actions-visible');
    }

    function expandCardDetails(card) {
        card.classList.add('expanded');
        vibrate(25);
    }

    function collapseCardDetails(card) {
        card.classList.remove('expanded');
    }

    function toggleCardBookmark(card) {
        card.classList.toggle('bookmarked');
        vibrate(50);
    }

    function toggleChartZoom(chart) {
        chart.classList.toggle('zoomed');
        vibrate(25);
    }

    function toggleCardExpansion(card) {
        card.classList.toggle('expanded');
        vibrate(25);
    }

    function cycleTheme() {
        if (window.ThemeSystem) {
            const themes = ['cyber-dark', 'neon-blue', 'mining-green', 'retro-amber', 'arctic-blue'];
            const current = window.ThemeSystem.getCurrentTheme();
            const currentIndex = themes.indexOf(current.id);
            const nextIndex = (currentIndex + 1) % themes.length;
            window.ThemeSystem.applyTheme(themes[nextIndex]);
            vibrate(100);
        }
    }

    function showContextMenu(element, event) {
        // Create context menu
        const menu = document.createElement('div');
        menu.className = 'context-menu';
        menu.innerHTML = `
            <div class="context-menu-item" onclick="copyToClipboard('${element.textContent}')">
                <i data-lucide="copy"></i> Copy
            </div>
            <div class="context-menu-item" onclick="shareElement('${element.id}')">
                <i data-lucide="share"></i> Share
            </div>
            <div class="context-menu-item" onclick="bookmarkElement('${element.id}')">
                <i data-lucide="bookmark"></i> Bookmark
            </div>
        `;
        
        menu.style.cssText = `
            position: fixed;
            background: var(--color-card-bg);
            border: 1px solid var(--color-border);
            border-radius: var(--border-radius-md);
            box-shadow: var(--shadow-lg);
            z-index: 9999;
            min-width: 150px;
        `;
        
        document.body.appendChild(menu);
        
        // Position menu
        const rect = element.getBoundingClientRect();
        menu.style.left = rect.left + 'px';
        menu.style.top = (rect.bottom + 10) + 'px';
        
        // Remove menu after delay or on next tap
        setTimeout(() => {
            if (menu.parentElement) {
                menu.remove();
            }
        }, 3000);
        
        document.addEventListener('touchstart', function removeMenu() {
            menu.remove();
            document.removeEventListener('touchstart', removeMenu);
        });
    }

    /**
     * Handle card swipe with visual feedback
     */
    function handleCardSwipe(card, deltaX, deltaY) {
        const threshold = 50;
        
        if (Math.abs(deltaX) > threshold) {
            // Horizontal swipe
            card.style.transform = `translateX(${deltaX * 0.3}px)`;
            
            if (deltaX > threshold) {
                card.classList.add('swipe-right');
            } else if (deltaX < -threshold) {
                card.classList.add('swipe-left');
            }
        }
        
        // Reset after animation
        setTimeout(() => {
            card.style.transform = '';
            card.classList.remove('swipe-left', 'swipe-right');
        }, 300);
    }

    /**
     * Public API
     */
    return {
        init,
        vibrate,
        isTouch: () => isTouch,
        addRippleEffect
    };

})();

// Global utility functions
window.copyToClipboard = function(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
        if (window.DashboardCore) {
            window.DashboardCore.showAlert('Copied to clipboard', 'success', 2000);
        }
    }
};

window.shareElement = function(elementId) {
    if (navigator.share) {
        navigator.share({
            title: 'BitAxe Dashboard',
            url: window.location.href
        });
    }
};

window.bookmarkElement = function(elementId) {
    // Implement bookmarking logic
    if (window.DashboardCore) {
        window.DashboardCore.showAlert('Bookmark added', 'success', 2000);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.MobileInteractions !== 'undefined') {
        window.MobileInteractions.init();
    }
});