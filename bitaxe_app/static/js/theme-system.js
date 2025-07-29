/**
 * BitAxe Theme System
 * Advanced theme management with customization options
 */

window.ThemeSystem = (function() {
    'use strict';

    // Theme configurations
    const themes = {
        'cyber-dark': {
            name: 'Cyber Dark',
            description: 'Classic cyberpunk dark theme with neon accents',
            colors: {
                primary: '#00d4ff',
                secondary: '#ff6b35',
                success: '#00ff88',
                warning: '#ffab00',
                danger: '#ff3366',
                background: '#0a0a0b',
                surface: '#1a1a1f',
                card: '#252530'
            },
            effects: {
                glow: true,
                particles: false,
                scanlines: false
            }
        },
        'neon-blue': {
            name: 'Neon Blue',
            description: 'Electric blue theme with purple accents',
            colors: {
                primary: '#4d9fff',
                secondary: '#bf5fff',
                success: '#00ff88',
                warning: '#ffab00',
                danger: '#ff3366',
                background: '#0a0a0b',
                surface: '#1a1a1f',
                card: '#252530'
            },
            effects: {
                glow: true,
                particles: true,
                scanlines: false
            }
        },
        'mining-green': {
            name: 'Mining Green',
            description: 'Green mining theme with gold accents',
            colors: {
                primary: '#39ff14',
                secondary: '#ffd700',
                success: '#00ff88',
                warning: '#ffab00',
                danger: '#ff3366',
                background: '#0a0a0b',
                surface: '#1a1a1f',
                card: '#252530'
            },
            effects: {
                glow: true,
                particles: false,
                scanlines: true
            }
        },
        'retro-amber': {
            name: 'Retro Amber',
            description: 'Vintage terminal amber theme',
            colors: {
                primary: '#ffb000',
                secondary: '#ff8c00',
                success: '#00ff88',
                warning: '#ffab00',
                danger: '#ff3366',
                background: '#1a1a0a',
                surface: '#2a2a1a',
                card: '#353520'
            },
            effects: {
                glow: false,
                particles: false,
                scanlines: true
            }
        },
        'arctic-blue': {
            name: 'Arctic Blue',
            description: 'Cool blue theme for cold mining',
            colors: {
                primary: '#00bfff',
                secondary: '#87ceeb',
                success: '#00ff88',
                warning: '#ffab00',
                danger: '#ff3366',
                background: '#0a0a1a',
                surface: '#1a1a2a',
                card: '#252535'
            },
            effects: {
                glow: true,
                particles: true,
                scanlines: false
            }
        }
    };

    let currentTheme = 'cyber-dark';
    let customizations = {};

    /**
     * Initialize theme system
     */
    function init() {
        loadThemeSettings();
        applyTheme(currentTheme);
        setupThemeControls();
        console.log('Theme System initialized');
    }

    /**
     * Load theme settings from localStorage
     */
    function loadThemeSettings() {
        const saved = localStorage.getItem('bitaxe-theme-settings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                currentTheme = settings.theme || 'cyber-dark';
                customizations = settings.customizations || {};
            } catch (e) {
                console.warn('Failed to load theme settings:', e);
            }
        }
    }

    /**
     * Save theme settings to localStorage
     */
    function saveThemeSettings() {
        const settings = {
            theme: currentTheme,
            customizations: customizations
        };
        localStorage.setItem('bitaxe-theme-settings', JSON.stringify(settings));
    }

    /**
     * Apply theme to the page
     */
    function applyTheme(themeId, customOptions = {}) {
        const theme = themes[themeId];
        if (!theme) {
            console.error('Theme not found:', themeId);
            return false;
        }

        currentTheme = themeId;
        
        // Remove old theme classes
        document.body.className = document.body.className.replace(/theme-\w+/g, '');
        
        // Add new theme class
        if (themeId !== 'cyber-dark') {
            document.body.classList.add(`theme-${themeId}`);
        }

        // Apply CSS custom properties
        const root = document.documentElement;
        Object.entries(theme.colors).forEach(([key, value]) => {
            root.style.setProperty(`--theme-${key}`, value);
        });

        // Apply effects
        applyThemeEffects(theme.effects, customOptions);

        // Update theme selector UI
        updateThemeSelector(themeId);

        // Save settings
        saveThemeSettings();

        // Notify other systems
        document.dispatchEvent(new CustomEvent('themeChanged', {
            detail: { theme: themeId, config: theme }
        }));

        return true;
    }

    /**
     * Apply theme effects
     */
    function applyThemeEffects(effects, customOptions = {}) {
        const mergedEffects = { ...effects, ...customOptions };

        // Glow effects
        document.body.classList.toggle('glow-effects', mergedEffects.glow);

        // Particle effects
        if (mergedEffects.particles) {
            createParticleEffect();
        } else {
            removeParticleEffect();
        }

        // Scanline effects
        document.body.classList.toggle('scanlines', mergedEffects.scanlines);
    }

    /**
     * Create particle background effect
     */
    function createParticleEffect() {
        // Remove existing particles
        removeParticleEffect();

        const canvas = document.createElement('canvas');
        canvas.id = 'particle-canvas';
        canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.3;
        `;

        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const particles = [];
        const particleCount = 50;

        // Resize canvas
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Create particles
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2
            });
        }

        // Animation loop
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach(particle => {
                // Update position
                particle.x += particle.vx;
                particle.y += particle.vy;

                // Bounce off edges
                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;

                // Draw particle
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 212, 255, ${particle.opacity})`;
                ctx.fill();
            });

            requestAnimationFrame(animate);
        }

        animate();
    }

    /**
     * Remove particle effect
     */
    function removeParticleEffect() {
        const canvas = document.getElementById('particle-canvas');
        if (canvas) {
            canvas.remove();
        }
    }

    /**
     * Setup theme control UI
     */
    function setupThemeControls() {
        // Create theme selector modal if it doesn't exist
        if (!document.getElementById('theme-selector-modal')) {
            createThemeSelector();
        }

        // Update existing theme buttons
        updateThemeSelector(currentTheme);
    }

    /**
     * Create advanced theme selector modal
     */
    function createThemeSelector() {
        const modal = document.createElement('div');
        modal.id = 'theme-selector-modal';
        modal.className = 'theme-modal';
        modal.innerHTML = `
            <div class="theme-modal-backdrop" onclick="closeThemeSelector()"></div>
            <div class="theme-modal-content">
                <div class="theme-modal-header">
                    <h3>Choose Theme</h3>
                    <button class="btn-icon" onclick="closeThemeSelector()">
                        <i data-lucide="x"></i>
                    </button>
                </div>
                <div class="theme-modal-body">
                    <div class="theme-grid">
                        ${Object.entries(themes).map(([id, theme]) => `
                            <div class="theme-card" data-theme="${id}" onclick="selectTheme('${id}')">
                                <div class="theme-preview" style="background: ${theme.colors.background}">
                                    <div class="theme-preview-bar" style="background: ${theme.colors.primary}"></div>
                                    <div class="theme-preview-card" style="background: ${theme.colors.card}"></div>
                                </div>
                                <div class="theme-info">
                                    <h4>${theme.name}</h4>
                                    <p>${theme.description}</p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="theme-customization">
                        <h4>Effects</h4>
                        <div class="effect-controls">
                            <label class="effect-control">
                                <input type="checkbox" id="glow-effect" checked>
                                <span>Glow Effects</span>
                            </label>
                            <label class="effect-control">
                                <input type="checkbox" id="particle-effect">
                                <span>Particle Background</span>
                            </label>
                            <label class="effect-control">
                                <input type="checkbox" id="scanline-effect">
                                <span>Scanline Overlay</span>
                            </label>
                        </div>
                    </div>
                </div>
                <div class="theme-modal-footer">
                    <button class="btn btn-secondary" onclick="resetTheme()">Reset to Default</button>
                    <button class="btn btn-primary" onclick="closeThemeSelector()">Apply</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add CSS for theme modal
        const style = document.createElement('style');
        style.textContent = `
            .theme-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 9999;
                display: none;
                align-items: center;
                justify-content: center;
            }

            .theme-modal.show {
                display: flex;
                animation: fadeIn 0.3s ease-out;
            }

            .theme-modal-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(10px);
            }

            .theme-modal-content {
                position: relative;
                width: 90%;
                max-width: 800px;
                max-height: 80vh;
                background: var(--color-card-bg);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-lg);
                box-shadow: var(--shadow-xl);
                overflow: hidden;
                animation: scaleIn 0.3s ease-out;
            }

            .theme-modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-lg);
                border-bottom: 1px solid var(--color-border);
            }

            .theme-modal-body {
                padding: var(--spacing-lg);
                max-height: 60vh;
                overflow-y: auto;
            }

            .theme-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: var(--spacing-md);
                margin-bottom: var(--spacing-xl);
            }

            .theme-card {
                border: 2px solid var(--color-border);
                border-radius: var(--border-radius-md);
                overflow: hidden;
                cursor: pointer;
                transition: all var(--transition-fast);
            }

            .theme-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }

            .theme-card.active {
                border-color: var(--color-accent-primary);
                box-shadow: var(--glow-primary);
            }

            .theme-preview {
                height: 80px;
                position: relative;
                overflow: hidden;
            }

            .theme-preview-bar {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 20px;
            }

            .theme-preview-card {
                position: absolute;
                bottom: 10px;
                left: 10px;
                right: 10px;
                height: 40px;
                border-radius: 4px;
            }

            .theme-info {
                padding: var(--spacing-md);
            }

            .theme-info h4 {
                margin-bottom: var(--spacing-xs);
                color: var(--color-text-primary);
            }

            .theme-info p {
                font-size: 0.8rem;
                color: var(--color-text-secondary);
            }

            .theme-customization h4 {
                margin-bottom: var(--spacing-md);
                color: var(--color-text-primary);
            }

            .effect-controls {
                display: flex;
                flex-wrap: wrap;
                gap: var(--spacing-md);
            }

            .effect-control {
                display: flex;
                align-items: center;
                gap: var(--spacing-sm);
                cursor: pointer;
                color: var(--color-text-primary);
            }

            .theme-modal-footer {
                display: flex;
                justify-content: space-between;
                padding: var(--spacing-lg);
                border-top: 1px solid var(--color-border);
            }
        `;
        document.head.appendChild(style);

        // Initialize icons
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    /**
     * Show theme selector modal
     */
    function showThemeSelector() {
        const modal = document.getElementById('theme-selector-modal');
        if (modal) {
            modal.classList.add('show');
            updateThemeSelector(currentTheme);
        }
    }

    /**
     * Close theme selector modal
     */
    function closeThemeSelector() {
        const modal = document.getElementById('theme-selector-modal');
        if (modal) {
            modal.classList.remove('show');
        }
    }

    /**
     * Update theme selector UI
     */
    function updateThemeSelector(activeTheme) {
        // Update theme cards
        document.querySelectorAll('.theme-card').forEach(card => {
            card.classList.toggle('active', card.dataset.theme === activeTheme);
        });

        // Update theme option buttons
        document.querySelectorAll('.theme-option').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === activeTheme);
        });

        // Update effect checkboxes based on current theme
        const theme = themes[activeTheme];
        if (theme) {
            const glowCheckbox = document.getElementById('glow-effect');
            const particleCheckbox = document.getElementById('particle-effect');
            const scanlineCheckbox = document.getElementById('scanline-effect');

            if (glowCheckbox) glowCheckbox.checked = theme.effects.glow;
            if (particleCheckbox) particleCheckbox.checked = theme.effects.particles;
            if (scanlineCheckbox) scanlineCheckbox.checked = theme.effects.scanlines;
        }
    }

    /**
     * Select theme from modal
     */
    function selectTheme(themeId) {
        const customEffects = getCustomEffects();
        applyTheme(themeId, customEffects);
        
        if (window.DashboardCore) {
            window.DashboardCore.showAlert(`Theme changed to ${themes[themeId].name}`, 'success', 2000);
        }
    }

    /**
     * Get custom effect settings
     */
    function getCustomEffects() {
        const glowCheckbox = document.getElementById('glow-effect');
        const particleCheckbox = document.getElementById('particle-effect');
        const scanlineCheckbox = document.getElementById('scanline-effect');

        return {
            glow: glowCheckbox?.checked,
            particles: particleCheckbox?.checked,
            scanlines: scanlineCheckbox?.checked
        };
    }

    /**
     * Reset theme to default
     */
    function resetTheme() {
        applyTheme('cyber-dark');
        updateThemeSelector('cyber-dark');
        
        if (window.DashboardCore) {
            window.DashboardCore.showAlert('Theme reset to default', 'success', 2000);
        }
    }

    /**
     * Get current theme info
     */
    function getCurrentTheme() {
        return {
            id: currentTheme,
            config: themes[currentTheme],
            customizations: customizations
        };
    }

    /**
     * Get all available themes
     */
    function getAvailableThemes() {
        return themes;
    }

    /**
     * Auto-detect preferred theme based on system settings
     */
    function autoDetectTheme() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            // Light mode preference - use a lighter dark theme
            return 'arctic-blue';
        } else {
            // Dark mode preference (default)
            return 'cyber-dark';
        }
    }

    // Global functions for template usage
    window.showThemeSelector = showThemeSelector;
    window.closeThemeSelector = closeThemeSelector;
    window.selectTheme = selectTheme;
    window.resetTheme = resetTheme;

    /**
     * Public API
     */
    return {
        init,
        applyTheme,
        showThemeSelector,
        closeThemeSelector,
        getCurrentTheme,
        getAvailableThemes,
        autoDetectTheme
    };

})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.ThemeSystem !== 'undefined') {
        window.ThemeSystem.init();
    }
});