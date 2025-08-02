/**
 * Enhanced RAG Pipeline Dashboard - Complete JavaScript Application
 * Features: Live Theater, Domain Intelligence, Database Explorer, ML Gallery, Portfolio Scanner
 */

// Global State
const AppState = {
    currentPage: 'live-theater',
    currentDomain: 'news',
    currentTable: null,
    systemHealth: null,
    websocket: null,
    isTheaterRunning: false
};

// Enhanced API Client
class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: { 'Content-Type': 'application/json', ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, config);
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            showToast(`API Error: ${error.message}`, 'error');
            throw error;
        }
    }

    // Core endpoints
    async checkHealth() { return this.request('/api/health'); }
    async getSetups(params = {}) { return this.request(`/api/setups?${new URLSearchParams(params)}`); }
    async getDomainIntelligence(domain) { return this.request(`/api/domain/${domain}/intelligence`); }
    async generateTSNE(domain) { return this.request('/api/generate-tsne', { method: 'POST', body: JSON.stringify({ domain }) }); }
    async getDatabaseTables() { return this.request('/api/database/tables'); }
    async getTableData(tableName, params = {}) { return this.request(`/api/database/table/${tableName}?${new URLSearchParams(params)}`); }
    async getTopMLPredictions(limit = 50) { return this.request(`/api/ml-results/top-predictions?limit=${limit}`); }
    async getModelPerformance() { return this.request('/api/model-performance'); }
    async getVisualizations() { return this.request('/api/visualizations'); }
}

const api = new APIClient();

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced RAG Pipeline Dashboard initializing...');
    initializeEventListeners();
    initializeThemeSelector();
    checkSystemHealth();
    showPage('live-theater');
    console.log('✅ Dashboard initialized successfully');
});

// Event Listeners
function initializeEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-button').forEach(button => {
        button.addEventListener('click', (e) => showPage(e.target.getAttribute('data-page')));
    });

    // Live Theater
    const startTheaterBtn = document.getElementById('start-theater');
    if (startTheaterBtn) startTheaterBtn.addEventListener('click', startLivePredictionTheater);

    // Domain Intelligence
    document.querySelectorAll('.domain-tab').forEach(tab => {
        tab.addEventListener('click', (e) => showDomainIntelligence(e.target.getAttribute('data-domain')));
    });

    // Visualization generation
    document.querySelectorAll('.generate-viz-btn').forEach(btn => {
        btn.addEventListener('click', (e) => generateVisualization(e.target.getAttribute('data-viz')));
    });

    // Database Explorer
    const loadTableBtn = document.getElementById('load-table');
    if (loadTableBtn) loadTableBtn.addEventListener('click', loadSelectedTable);

    // ML Gallery
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.addEventListener('click', (e) => filterMLResults(e.target.getAttribute('data-category')));
    });

    // Portfolio Scanner
    const startScanBtn = document.getElementById('start-scan');
    if (startScanBtn) startScanBtn.addEventListener('click', startPortfolioScan);

    // Confidence slider
    const confidenceSlider = document.getElementById('min-confidence');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', (e) => {
            document.getElementById('confidence-value').textContent = Math.round(e.target.value * 100) + '%';
        });
    }
}

// Theme Management
function initializeThemeSelector() {
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
        themeSelect.addEventListener('change', (e) => setTheme(e.target.value));
        const savedTheme = localStorage.getItem('dashboard-theme') || 'professional';
        themeSelect.value = savedTheme;
        setTheme(savedTheme);
    }
}

function setTheme(themeName) {
    document.body.className = `theme-${themeName}`;
    localStorage.setItem('dashboard-theme', themeName);
    showToast(`Theme changed to ${themeName}`, 'info');
}

// Page Navigation
function showPage(pageId) {
    document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-page="${pageId}"]`)?.classList.add('active');
    
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        AppState.currentPage = pageId;
        loadPageContent(pageId);
    }
}

function loadPageContent(pageId) {
    switch(pageId) {
        case 'domain-intelligence': loadDomainIntelligence(); break;
        case 'database-explorer': loadDatabaseExplorer(); break;
        case 'ml-gallery': loadMLGallery(); break;
        case 'portfolio-scanner': initializePortfolioScanner(); break;
    }
}

// System Health
async function checkSystemHealth() {
    try {
        const health = await api.checkHealth();
        AppState.systemHealth = health;
        updateSystemStatus(health);
        setTimeout(checkSystemHealth, 30000);
    } catch (error) {
        updateSystemStatus({ status: 'unhealthy', error: error.message });
        setTimeout(checkSystemHealth, 10000);
    }
}

function updateSystemStatus(health) {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    
    if (statusIndicator && statusText) {
        if (health.status === 'healthy') {
            statusIndicator.style.color = '#22c55e';
            statusText.textContent = `Healthy (${health.total_setups} setups)`;
        } else {
            statusIndicator.style.color = '#ef4444';
            statusText.textContent = 'System Error';
        }
    }
}

// Live Prediction Theater
async function startLivePredictionTheater() {
    const setupCount = parseInt(document.getElementById('setup-count').value) || 5;
    
    if (AppState.isTheaterRunning) {
        showToast('Prediction theater is already running', 'warning');
        return;
    }

    AppState.isTheaterRunning = true;
    showTheaterElements();
    resetAgentCards();
    
    try {
        await connectToTheaterWebSocket(setupCount);
    } catch (error) {
        console.error('Theater error:', error);
        showToast(`Theater error: ${error.message}`, 'error');
        AppState.isTheaterRunning = false;
    }
}

function showTheaterElements() {
    ['theater-progress', 'agent-status-grid', 'theater-log', 'theater-results'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = id === 'agent-status-grid' ? 'grid' : 'block';
    });
}

async function connectToTheaterWebSocket(count) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live-prediction`;
    
    AppState.websocket = new WebSocket(wsUrl);
    
    AppState.websocket.onopen = () => {
        console.log('WebSocket connected');
        AppState.websocket.send(JSON.stringify({ action: 'start_prediction', count: count }));
    };
    
    AppState.websocket.onmessage = (event) => handleTheaterMessage(JSON.parse(event.data));
    AppState.websocket.onclose = () => { console.log('WebSocket disconnected'); AppState.isTheaterRunning = false; };
    AppState.websocket.onerror = (error) => { console.error('WebSocket error:', error); showToast('Connection error', 'error'); AppState.isTheaterRunning = false; };
}

function handleTheaterMessage(data) {
    switch(data.type) {
        case 'theater_start':
            logTheaterMessage(data.message, 'info');
            updateTheaterProgress(0, data.message);
            break;
        case 'setups_selected':
            logTheaterMessage(data.message, 'success');
            break;
        case 'setup_start':
            logTheaterMessage(data.message, 'info');
            updateTheaterProgress((data.setup_index / data.total_setups) * 100, data.message);
            break;
        case 'step_progress':
            updateAgentStatus(data.agent, data.step, data.message);
            logTheaterMessage(data.message, 'progress');
            break;
        case 'agent_prediction_complete':
            updateAgentComplete(data.agent, data.result);
            logTheaterMessage(data.message, 'success');
            break;
        case 'setup_complete':
            displaySetupResult(data);
            break;
        case 'theater_complete':
            logTheaterMessage(data.message, 'success');
            updateTheaterProgress(100, 'Theater Complete!');
            AppState.isTheaterRunning = false;
            break;
    }
}

function logTheaterMessage(message, type = 'info') {
    const logContent = document.getElementById('log-content');
    if (logContent) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
        logContent.appendChild(logEntry);
        logContent.scrollTop = logContent.scrollHeight;
    }
}

function updateTheaterProgress(percentage, message) {
    const progressFill = document.getElementById('progress-fill');
    const progressTitle = document.getElementById('progress-title');
    
    if (progressFill) progressFill.style.width = `${percentage}%`;
    if (progressTitle) progressTitle.textContent = message;
}

function resetAgentCards() {
    ['fundamentals', 'news', 'analyst_recommendations', 'userposts'].forEach(agent => {
        const statusEl = document.getElementById(`${agent}-status`);
        const progressEl = document.getElementById(`${agent}-progress`);
        
        if (statusEl) statusEl.textContent = 'Waiting...';
        if (progressEl) progressEl.innerHTML = '';
    });
}

function updateAgentStatus(agent, step, message) {
    const statusEl = document.getElementById(`${agent}-status`);
    if (statusEl) {
        statusEl.textContent = step === 'embedding_generation' ? 'Generating embeddings...' : 'Making prediction...';
    }
}

function updateAgentComplete(agent, result) {
    const statusEl = document.getElementById(`${agent}-status`);
    const progressEl = document.getElementById(`${agent}-progress`);
    
    if (statusEl) statusEl.textContent = 'Complete ✅';
    if (progressEl) {
        const confidence = Math.round(result.confidence_score * 100);
        progressEl.innerHTML = `
            <div class="agent-result">
                <div class="prediction-value ${result.prediction_class.toLowerCase()}">
                    ${result.predicted_outperformance_10d.toFixed(2)}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                    <span class="confidence-text">${confidence}%</span>
                </div>
            </div>
        `;
    }
}

function displaySetupResult(data) {
    const resultsGrid = document.getElementById('results-grid');
    if (!resultsGrid) return;
    
    const actualClass = data.actual_label > 2 ? 'positive' : data.actual_label < -2 ? 'negative' : 'neutral';
    const isAccurate = data.ensemble_prediction.prediction_class.toLowerCase() === actualClass;
    
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card';
    resultCard.innerHTML = `
        <div class="result-header">
            <h4>${data.setup_id}</h4>
            <span class="accuracy-badge ${isAccurate ? 'accurate' : 'inaccurate'}">
                ${isAccurate ? '✅ Accurate' : '❌ Inaccurate'}
            </span>
        </div>
        <div class="result-metrics">
            <div class="metric">
                <label>Predicted:</label>
                <span class="value ${data.ensemble_prediction.prediction_class.toLowerCase()}">
                    ${data.ensemble_prediction.predicted_outperformance_10d.toFixed(2)}%
                </span>
            </div>
            <div class="metric">
                <label>Actual:</label>
                <span class="value ${actualClass}">
                    ${data.actual_label ? data.actual_label.toFixed(2) : 'N/A'}%
                </span>
            </div>
            <div class="metric">
                <label>Confidence:</label>
                <span class="value">${Math.round(data.ensemble_prediction.confidence_score * 100)}%</span>
            </div>
        </div>
        <div class="agent-breakdown">
            ${Object.entries(data.agent_predictions).map(([agent, pred]) => `
                <div class="agent-pred">
                    <span class="agent-name">${agent}</span>
                    <span class="agent-value ${pred.prediction_class.toLowerCase()}">
                        ${pred.predicted_outperformance_10d.toFixed(1)}%
                    </span>
                </div>
            `).join('')}
        </div>
    `;
    
    resultsGrid.appendChild(resultCard);
}

// Domain Intelligence
async function loadDomainIntelligence() {
    showDomainIntelligence('news');
}

async function showDomainIntelligence(domain) {
    AppState.currentDomain = domain;
    
    document.querySelectorAll('.domain-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`[data-domain="${domain}"]`)?.classList.add('active');
    
    document.querySelectorAll('.domain-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`domain-${domain}`)?.classList.add('active');
    
    try {
        const intelligence = await api.getDomainIntelligence(domain);
        displayDomainIntelligence(domain, intelligence);
    } catch (error) {
        console.error(`Error loading ${domain} intelligence:`, error);
    }
}

function displayDomainIntelligence(domain, data) {
    if (domain === 'news' && data.intelligence) {
        const bullishEl = document.getElementById('bullish-terms');
        if (bullishEl && data.intelligence.bullish_terms) {
            bullishEl.innerHTML = data.intelligence.bullish_terms
                .map(term => `<span class="term-tag positive">${term}</span>`)
                .join('');
        }
        
        const bearishEl = document.getElementById('bearish-terms');
        if (bearishEl && data.intelligence.bearish_terms) {
            bearishEl.innerHTML = data.intelligence.bearish_terms
                .map(term => `<span class="term-tag negative">${term}</span>`)
                .join('');
        }
        
        const chartEl = document.getElementById('sentiment-chart');
        if (chartEl && data.intelligence.sentiment_distribution) {
            const dist = data.intelligence.sentiment_distribution;
            const total = dist.positive + dist.negative + dist.neutral;
            chartEl.innerHTML = `
                <div class="sentiment-bars">
                    <div class="sentiment-bar">
                        <label>Positive</label>
                        <div class="bar-container">
                            <div class="bar positive" style="width: ${(dist.positive / total) * 100}%"></div>
                        </div>
                        <span>${dist.positive}</span>
                    </div>
                    <div class="sentiment-bar">
                        <label>Negative</label>
                        <div class="bar-container">
                            <div class="bar negative" style="width: ${(dist.negative / total) * 100}%"></div>
                        </div>
                        <span>${dist.negative}</span>
                    </div>
                    <div class="sentiment-bar">
                        <label>Neutral</label>
                        <div class="bar-container">
                            <div class="bar neutral" style="width: ${(dist.neutral / total) * 100}%"></div>
                        </div>
                        <span>${dist.neutral}</span>
                    </div>
                </div>
            `;
        }
    }
}

async function generateVisualization(vizType) {
    const btn = document.querySelector(`[data-viz="${vizType}"]`);
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Generating...';
    }
    
    try {
        if (vizType === 'news-tsne') {
            const result = await api.generateTSNE('news');
            if (result.success) {
                const container = btn.parentElement;
                container.innerHTML = `
                    <img src="${result.image}" alt="News t-SNE Clustering" style="max-width: 100%; height: auto;">
                    <div class="viz-stats">
                        <p>📊 ${result.terms_count} terms clustered by performance class</p>
                        <p>🟢 ${result.clusters.positive} positive • 🔴 ${result.clusters.negative} negative • ⚪ ${result.clusters.neutral} neutral</p>
                    </div>
                `;
                showToast('t-SNE visualization generated!', 'success');
            }
        }
    } catch (error) {
        console.error('Visualization error:', error);
        showToast('Failed to generate visualization', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Generate t-SNE Visualization';
        }
    }
}

// Database Explorer
async function loadDatabaseExplorer() {
    try {
        const tables = await api.getDatabaseTables();
        populateTableSelector(tables.tables);
    } catch (error) {
        console.error('Error loading database explorer:', error);
    }
}

function populateTableSelector(tables) {
    const tableSelect = document.getElementById('table-select');
    if (tableSelect) {
        tableSelect.innerHTML = '<option value="">Choose a table...</option>';
        tables.forEach(table => {
            const option = document.createElement('option');
            option.value = table.name;
            option.textContent = `${table.display_name} (${table.rows} rows)`;
            tableSelect.appendChild(option);
        });
    }
}

async function loadSelectedTable() {
    const tableSelect = document.getElementById('table-select');
    const tableName = tableSelect.value;
    
    if (!tableName) {
        showToast('Please select a table first', 'warning');
        return;
    }
    
    AppState.currentTable = tableName;
    
    try {
        showLoading(true, 'Loading table data...');
        
        const searchTerm = document.getElementById('table-search').value;
        const performanceFilter = document.getElementById('performance-filter').value;
        
        const params = { limit: 100, offset: 0 };
        if (searchTerm) params.search = searchTerm;
        if (performanceFilter) params.performance_filter = performanceFilter;
        
        const tableData = await api.getTableData(tableName, params);
        displayTableData(tableData);
        updateTableStats(tableData);
        
    } catch (error) {
        console.error('Error loading table:', error);
        showToast('Failed to load table data', 'error');
    } finally {
        showLoading(false);
    }
}

function displayTableData(tableData) {
    const container = document.getElementById('data-table-container');
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    
    if (!container || !tableHead || !tableBody) return;
    
    container.style.display = 'block';
    
    tableHead.innerHTML = '';
    const headerRow = document.createElement('tr');
    tableData.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.replace('_', ' ').toUpperCase();
        headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);
    
    tableBody.innerHTML = '';
    tableData.data.forEach(row => {
        const tr = document.createElement('tr');
        tableData.columns.forEach(col => {
            const td = document.createElement('td');
            let value = row[col];
            
            if (typeof value === 'number') {
                if (col.includes('outperformance') || col.includes('percent')) {
                    value = value.toFixed(2) + '%';
                } else if (value % 1 !== 0) {
                    value = value.toFixed(3);
                }
            }
            
            td.textContent = value || '';
            tr.appendChild(td);
        });
        tableBody.appendChild(tr);
    });
}

function updateTableStats(tableData) {
    const statsContainer = document.getElementById('table-stats');
    if (statsContainer) {
        statsContainer.style.display = 'flex';
        
        document.getElementById('total-rows').textContent = tableData.total_count.toLocaleString();
        document.getElementById('total-columns').textContent = tableData.columns.length;
        document.getElementById('data-completeness').textContent = 
            Math.round((tableData.returned_count / tableData.total_count) * 100) + '%';
        document.getElementById('last-updated').textContent = 'Live Data';
    }
}

// ML Gallery
async function loadMLGallery() {
    try {
        showLoading(true, 'Loading ML results...');
        
        const performance = await api.getModelPerformance();
        displayModelPerformance(performance);
        
        const predictions = await api.getTopMLPredictions(50);
        displayTopPredictions(predictions);
        
        const visualizations = await api.getVisualizations();
        displayVisualizationGallery(visualizations);
        
    } catch (error) {
        console.error('Error loading ML gallery:', error);
        showToast('Failed to load ML results', 'error');
    } finally {
        showLoading(false);
    }
}

function displayModelPerformance(performance) {
    const container = document.getElementById('performance-cards');
    if (!container) return;
    
    container.innerHTML = performance.models.map(model => `
        <div class="performance-card">
            <h4>${model.name}</h4>
            <div class="performance-metrics">
                <div class="metric">
                    <label>Precision</label>
                    <span class="value">${model.precision.toFixed(3)}</span>
                </div>
                <div class="metric">
                    <label>Recall</label>
                    <span class="value">${model.recall.toFixed(3)}</span>
                </div>
                <div class="metric">
                    <label>F1-Score</label>
                    <span class="value">${model.f1_score.toFixed(3)}</span>
                </div>
                <div class="metric">
                    <label>AUC</label>
                    <span class="value">${model.auc.toFixed(3)}</span>
                </div>
            </div>
            ${model.name === performance.best_model ? 
                '<div class="best-model-badge">🏆 Best Model</div>' : ''
            }
        </div>
    `).join('');
}

function displayTopPredictions(data) {
    console.log('Top predictions loaded:', data.summary);
}

function displayVisualizationGallery(data) {
    const galleryGrid = document.getElementById('gallery-grid');
    if (!galleryGrid) return;
    
    if (!data.categories || Object.keys(data.categories).length === 0) {
        galleryGrid.innerHTML = '<p class="no-data">No visualizations found. Run your ML pipeline to generate charts.</p>';
        return;
    }
    
    galleryGrid.innerHTML = Object.entries(data.categories).map(([category, files]) => `
        <div class="gallery-category">
            <h4>${category.replace('_', ' ').toUpperCase()}</h4>
            <div class="file-grid">
                ${files.map(file => `
                    <div class="file-card">
                        <div class="file-icon">
                            ${file.type === 'image' ? '🖼️' : '📄'}
                        </div>
                        <div class="file-info">
                            <div class="file-name">${file.filename}</div>
                            <div class="file-size">${formatFileSize(file.size)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function filterMLResults(category) {
    document.querySelectorAll('.category-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-category="${category}"]`)?.classList.add('active');
    console.log('Filtering ML results by:', category);
}

// Portfolio Scanner
function initializePortfolioScanner() {
    const startDate = document.getElementById('scan-start-date');
    const endDate = document.getElementById('scan-end-date');
    
    if (startDate && endDate) {
        const today = new Date();
        const monthAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
        
        endDate.value = today.toISOString().split('T')[0];
        startDate.value = monthAgo.toISOString().split('T')[0];
    }
}

async function startPortfolioScan() {
    const startDate = document.getElementById('scan-start-date').value;
    const endDate = document.getElementById('scan-end-date').value;
    const minConfidence = parseFloat(document.getElementById('min-confidence').value);
    const predictionClass = document.getElementById('prediction-class').value;
    
    if (!startDate || !endDate) {
        showToast('Please select start and end dates', 'warning');
        return;
    }
    
    try {
        showLoading(true, 'Scanning portfolio opportunities...');
        
        const predictions = await api.getTopMLPredictions(50);
        
        let filteredPredictions = predictions.predictions.filter(pred => 
            pred.confidence_score >= minConfidence
        );
        
        if (predictionClass !== 'all') {
            filteredPredictions = filteredPredictions.filter(pred => 
                pred.prediction_class.toLowerCase() === predictionClass
            );
        }
        
        displayScanResults(filteredPredictions, predictions.summary);
        
    } catch (error) {
        console.error('Error during portfolio scan:', error);
        showToast('Portfolio scan failed', 'error');
    } finally {
        showLoading(false);
    }
}

function displayScanResults(predictions, summary) {
    const resultsContainer = document.getElementById('scan-results');
    const summaryContainer = document.getElementById('results-summary');
    const tbody = document.getElementById('results-tbody');
    
    if (!resultsContainer || !summaryContainer || !tbody) return;
    
    resultsContainer.style.display = 'block';
    
    summaryContainer.innerHTML = `
        <div class="scan-summary">
            <div class="summary-card">
                <h4>Found Opportunities</h4>
                <span class="big-number">${predictions.length}</span>
            </div>
            <div class="summary-card">
                <h4>Avg Confidence</h4>
                <span class="big-number">${(predictions.reduce((sum, p) => sum + p.confidence_score, 0) / predictions.length * 100).toFixed(1)}%</span>
            </div>
            <div class="summary-card">
                <h4>Positive Predictions</h4>
                <span class="big-number positive">${predictions.filter(p => p.prediction_class === 'POSITIVE').length}</span>
            </div>
            <div class="summary-card">
                <h4>Overall Accuracy</h4>
                <span class="big-number">${(summary.accuracy_rate * 100).toFixed(1)}%</span>
            </div>
        </div>
    `;
    
    tbody.innerHTML = predictions.map(pred => `
        <tr>
            <td>${pred.setup_id}</td>
            <td>${pred.ticker}</td>
            <td class="value ${pred.prediction_class.toLowerCase()}">${pred.predicted_performance}%</td>
            <td>${Math.round(pred.confidence_score * 100)}%</td>
            <td class="value ${pred.actual_class.toLowerCase()}">${pred.actual_performance}%</td>
            <td>
                <span class="class-badge ${pred.prediction_class.toLowerCase()}">
                    ${pred.prediction_class}
                </span>
            </td>
            <td>
                <button class="action-btn" onclick="viewSetupDetails('${pred.setup_id}')">
                    👁️ View
                </button>
            </td>
        </tr>
    `).join('');
}

// Utility Functions
function showLoading(show, message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
        if (text) text.textContent = message;
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function viewSetupDetails(setupId) {
    showToast(`Viewing details for ${setupId}`, 'info');
}

// Debug helpers
window.AppState = AppState;
window.api = api;