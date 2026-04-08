// Enterprise Spam Detection Application JavaScript

// Global variables
let isProcessing = false;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize animations
    initializeAnimations();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize form validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// Initialize animations
function initializeAnimations() {
    // Fade in elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements
    const animateElements = document.querySelectorAll('.card, .stat-box, .metric-card');
    animateElements.forEach(el => observer.observe(el));
}

// Initialize real-time updates
function initializeRealTimeUpdates() {
    // Auto-refresh dashboard data every 30 seconds
    if (window.location.pathname === '/analytics') {
        setInterval(updateDashboardData, 30000);
    }
}

// Update dashboard data
function updateDashboardData() {
    // Simulate data refresh with loading states
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach(card => {
        card.classList.add('loading');
    });
    
    // Simulate API call
    setTimeout(() => {
        cards.forEach(card => {
            card.classList.remove('loading');
        });
        showNotification('Dashboard data refreshed', 'success');
    }, 1000);
}

// Show notification
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    const container = document.getElementById('toastContainer') || createToastContainer();
    container.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Create toast container
function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1050';
    document.body.appendChild(container);
    return container;
}

// Loading state management
function setLoading(element, loading = true) {
    if (loading) {
        element.classList.add('loading');
        element.disabled = true;
        
        // Add spinner if button
        if (element.tagName === 'BUTTON') {
            const originalText = element.innerHTML;
            element.dataset.originalText = originalText;
            element.innerHTML = '<span class="spinner me-2"></span>Processing...';
        }
    } else {
        element.classList.remove('loading');
        element.disabled = false;
        
        // Restore button text
        if (element.tagName === 'BUTTON' && element.dataset.originalText) {
            element.innerHTML = element.dataset.originalText;
            delete element.dataset.originalText;
        }
    }
}

// API helper functions
class APIHelper {
    static async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }
    
    static async predictMessage(text) {
        return this.request('/api/predict', {
            method: 'POST',
            body: JSON.stringify({ text: text })
        });
    }
    
    static async batchPredict(texts) {
        return this.request('/api/batch_predict', {
            method: 'POST',
            body: JSON.stringify({ texts: texts })
        });
    }
}

// Text analysis utilities
class TextAnalyzer {
    static async analyzeText(text) {
        if (!text || text.trim().length === 0) {
            throw new Error('Text cannot be empty');
        }
        
        try {
            setLoading(document.querySelector('button[type="submit"]'), true);
            
            const result = await APIHelper.predictMessage(text);
            
            // Update UI with results
            updateAnalysisResults(text, result);
            
            return result;
        } catch (error) {
            showNotification('Analysis failed: ' + error.message, 'danger');
            throw error;
        } finally {
            setLoading(document.querySelector('button[type="submit"]'), false);
        }
    }
    
    static async analyzeBatch(texts) {
        if (!Array.isArray(texts) || texts.length === 0) {
            throw new Error('Texts array cannot be empty');
        }
        
        try {
            const result = await APIHelper.batchPredict(texts);
            return result;
        } catch (error) {
            showNotification('Batch analysis failed: ' + error.message, 'danger');
            throw error;
        }
    }
}

// Update analysis results in UI
function updateAnalysisResults(text, result) {
    // This would be implemented based on the specific page structure
    console.log('Analysis results:', result);
}

// File upload utilities
class FileUploader {
    static validateFile(file) {
        const allowedTypes = ['text/plain', 'text/csv'];
        const maxSize = 16 * 1024 * 1024; // 16MB
        
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload .txt or .csv files only.');
        }
        
        if (file.size > maxSize) {
            throw new Error('File too large. Maximum size is 16MB.');
        }
        
        return true;
    }
    
    static async uploadFile(file) {
        this.validateFile(file);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            return await response.text();
        } catch (error) {
            console.error('File upload failed:', error);
            throw error;
        }
    }
}

// Chart utilities
class ChartManager {
    static charts = {};
    
    static createChart(elementId, type, data, options = {}) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
        }
        
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        this.charts[elementId] = new Chart(ctx, {
            type: type,
            data: data,
            options: finalOptions
        });
        
        return this.charts[elementId];
    }
    
    static updateChart(elementId, data) {
        if (this.charts[elementId]) {
            this.charts[elementId].data = data;
            this.charts[elementId].update();
        }
    }
    
    static destroyChart(elementId) {
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
            delete this.charts[elementId];
        }
    }
}

// Utility functions
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

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

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter to submit forms
    if (e.ctrlKey && e.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.tagName === 'TEXTAREA') {
            const form = activeElement.closest('form');
            if (form) {
                form.dispatchEvent(new Event('submit', { cancelable: true }));
            }
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const openModal = document.querySelector('.modal.show');
        if (openModal) {
            const modal = bootstrap.Modal.getInstance(openModal);
            if (modal) {
                modal.hide();
            }
        }
    }
});

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred', 'danger');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showNotification('An unexpected error occurred', 'danger');
});

// Export classes for use in other scripts
window.SpamDetectionApp = {
    APIHelper,
    TextAnalyzer,
    FileUploader,
    ChartManager,
    showNotification,
    setLoading,
    formatBytes,
    formatNumber,
    debounce
};
