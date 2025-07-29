// Enhanced RAG Pipeline Frontend Application
// ==========================================

// Global State Management
let state = {
    selectedSetup: null,
    setups: [],
    healthStatus: 'connecting',
    currentPage: 'main',
    currentTheme: 'default',
    isTheaterRunning: false,
    websocket: null,
    theaterResults: [],
    similarSetups: []
};

// API Configuration
const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace('http', 'ws');

// API Client Class
class APIClient {
    constructor(baseURL = API_BASE) {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
}

// Initialize API client
const api = new APIClient();

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Enhanced RAG Pipeline Dashboard...');
    
    // Initialize theme
    initializeTheme();
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial data
    await loadInitialData();
    
    // Start health monitoring
    startHealthMonitoring();
    
    console.log('✅ Dashboard initialized successfully');
});

// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('rag-pipeline-theme') || 'default';
    const themeSelector = document.getElementById('theme-selector');
    
    if (themeSelector) {
        themeSelector.value = savedTheme;
        changeTheme(savedTheme);
    }
}

function changeTheme(themeName) {
    document.documentElement.setAttribute('data-theme', themeName);
    state.currentTheme = themeName;
    localStorage.setItem('rag-pipeline-theme', themeName);
    
    // Add smooth transition effect
    document.body.style.transition = 'all 0.3s ease';
    setTimeout(() => {
        document.body.style.transition = '';
    }, 300);
}

// Event Listeners Setup
function setupEventListeners() {
    // Theme selector
    const themeSelector = document.getElementById('theme-selector');
    if (themeSelector) {
        themeSelector.addEventListener('change', (e) => changeTheme(e.target.value));
    }

    // Live Theater Controls
    const startTheaterBtn = document.getElementById('start-theater-btn');
    if (startTheaterBtn) {
        startTheaterBtn.addEventListener('click', startLivePredictionTheater);
    }

    // Setup controls
    const setupSelect = document.getElementById('setup-select');
    if (setupSelect) {
        setupSelect.addEventListener('change', (e) => {
            state.selectedSetup = e.target.value;
            updateSetupButtons();
        });
    }

    const randomSetupBtn = document.getElementById('random-setup-btn');
    if (randomSetupBtn) {
        randomSetupBtn.addEventListener('click', selectRandomSetup);
    }

    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', generatePrediction);
    }

    const similarBtn = document.getElementById('similar-btn');
    if (similarBtn) {
        similarBtn.addEventListener('click', findSimilarSetups);
    }

    // Portfolio Scanner
    const scanBtn = document.getElementById('scan-btn');
    if (scanBtn) {
        scanBtn.addEventListener('click', scanPortfolio);
    }

    const probabilitySlider = document.getElementById('min-probability');
    if (probabilitySlider) {
        probabilitySlider.addEventListener('input', (e) => {
            const display = document.getElementById('probability-display');
            if (display) {
                display.textContent = `${Math.round(e.target.value * 100)}%`;
            }
        });
    }

    // Initialize date inputs
    initializeDateInputs();
}

function initializeDateInputs() {
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 30);
    
    const startInput = document.getElementById('start-date');
    const endInput = document.getElementById('end-date');
    
    if (startInput) startInput.value = startDate.toISOString().split('T')[0];
    if (endInput) endInput.value = today.toISOString().split('T')[0];
}

// Page Navigation
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(`${pageId}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
        state.currentPage = pageId;
    }
    
    // Activate corresponding nav button
    const navBtn = Array.from(document.querySelectorAll('.nav-btn')).find(btn => 
        btn.getAttribute('onclick')?.includes(pageId)
    );
    if (navBtn) {
        navBtn.classList.add('active');
    }

    // Load page-specific data
    loadPageData(pageId);
}

async function loadPageData(pageId) {
    switch (pageId) {
        case 'models':
            await loadModelPerformance();
            break;
        case 'visualizations':
            await loadVisualizationGallery();
            break;
        case 'data':
            await loadDataExplorer();
            break;
    }
}

// Data Loading Functions
async function loadInitialData() {
    try {
        await Promise.all([
            loadSetups(),
            checkHealth()
        ]);
    } catch (error) {
        console.error('Failed to load initial data:', error);
        showError('Failed to load initial data. Please refresh the page.');
    }
}

async function loadSetups() {
    try {
        showLoading('setup-select', 'Loading setups...');
        
        const response = await api.get('/api/setups?limit=200&random_sample=false');
        state.setups = response.setups || [];
        
        const setupSelect = document.getElementById('setup-select');
        if (setupSelect) {
            setupSelect.innerHTML = '<option value="">Select a setup...</option>';
            
            state.setups.forEach(setup => {
                const option = document.createElement('option');
                option.value = setup.setup_id;
                option.textContent = `${setup.setup_id} (${setup.ticker})`;
                setupSelect.appendChild(option);
            });
        }
        
        console.log(`📊 Loaded ${state.setups.length} setups`);
    } catch (error) {
        console.error('Failed to load setups:', error);
        showError('Failed to load setups');
    }
}

async function checkHealth() {
    try {
        const health = await api.get('/api/health');
        updateHealthStatus(health);
        updateSystemMetrics(health);
        return health;
    } catch (error) {
        console.error('Health check failed:', error);
        updateHealthStatus({ status: 'unhealthy', error: error.message });
        return null;
    }
}

function updateHealthStatus(health) {
    const indicator = document.getElementById('health-indicator');
    const statusText = document.getElementById('health-status');
    const statusDot = indicator?.querySelector('.status-dot');
    
    if (statusText) {
        statusText.textContent = health.status === 'healthy' ? 'System Online' : 'System Issues';
    }
    
    if (statusDot) {
        statusDot.className = 'status-dot';
        if (health.status === 'healthy') {
            statusDot.classList.add('healthy');
        } else {
            statusDot.classList.add('error');
        }
    }
    
    state.healthStatus = health.status;
}

function updateSystemMetrics(health) {
    const updates = {
        'total-setups': health.total_setups || 0,
        'available-models': health.components ? Object.keys(health.components).length : 0,
        'db-status': health.database === 'connected' ? '✅' : '❌',
    };
    
    Object.entries(updates).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
}

// Live Prediction Theater
async function startLivePredictionTheater() {
    if (state.isTheaterRunning) return;
    
    const setupCount = parseInt(document.getElementById('setup-count').value) || 5;
    const startBtn = document.getElementById('start-theater-btn');
    const theaterStage = document.getElementById('theater-stage');
    const theaterLog = document.getElementById('theater-log');
    
    // Validate input
    if (setupCount < 1 || setupCount > 20) {
        showError('Please enter a number between 1 and 20');
        return;
    }
    
    // Initialize theater
    state.isTheaterRunning = true;
    state.theaterResults = [];
    
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = '<span class="spinner"></span> Theater Running...';
    }
    
    if (theaterStage) theaterStage.style.display = 'block';
    if (theaterLog) theaterLog.innerHTML = '';
    
    // Reset agent cards
    resetAgentCards();
    
    try {
        // Connect to WebSocket
        await connectToTheaterWebSocket(setupCount);
    } catch (error) {
        console.error('Theater failed:', error);
        showError(`Theater failed: ${error.message}`);
        stopLivePredictionTheater();
    }
}

async function connectToTheaterWebSocket(setupCount) {
    return new Promise((resolve, reject) => {
        const wsUrl = `${WS_BASE}/ws/live-prediction`;
        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
        
        state.websocket = new WebSocket(wsUrl);
        
        state.websocket.onopen = () => {
            console.log('✅ WebSocket connected');
            
            // Send start command
            state.websocket.send(JSON.stringify({
                action: 'start_prediction',
                count: setupCount
            }));
        };
        
        state.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleTheaterMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        
        state.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(new Error('WebSocket connection failed'));
        };
        
        state.websocket.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            if (state.isTheaterRunning) {
                stopLivePredictionTheater();
            }
        };
    });
}

function handleTheaterMessage(data) {
    console.log('🎭 Theater message:', data.type, data.message);
    
    switch (data.type) {
        case 'theater_start':
            logTheaterMessage(data.message, 'setup-start');
            updateTheaterProgress(0);
            break;
            
        case 'setups_selected':
            logTheaterMessage(data.message, 'info');
            break;
            
        case 'setup_start':
            logTheaterMessage(data.message, 'setup-start');
            updateTheaterProgress((data.setup_index / data.total_setups) * 100);
            break;
            
        case 'step_progress':
            logTheaterMessage(data.message, 'progress');
            if (data.agent) {
                updateAgentStatus(data.agent, data.step);
            }
            break;
            
        case 'agent_prediction_complete':
            logTheaterMessage(data.message, 'agent-complete');
            updateAgentComplete(data.agent, data.result);
            break;
            
        case 'setup_complete':
            logTheaterMessage(data.message, 'setup-complete');
            addTheaterResult(data);
            break;
            
        case 'theater_complete':
            logTheaterMessage(data.message, 'theater-complete');
            completeTheater();
            break;
            
        case 'error':
            logTheaterMessage(`❌ ${data.message}`, 'error');
            showError(data.message);
            stopLivePredictionTheater();
            break;
    }
}

function logTheaterMessage(message, type = 'info') {
    const theaterLog = document.getElementById('theater-log');
    if (!theaterLog) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    entry.innerHTML = `
        <div class="log-timestamp">${timestamp}</div>
        <div class="log-message">${message}</div>
    `;
    
    theaterLog.appendChild(entry);
    theaterLog.scrollTop = theaterLog.scrollHeight;
}

function updateTheaterProgress(percentage) {
    const progressFill = document.getElementById('theater-progress');
    if (progressFill) {
        progressFill.style.width = `${percentage}%`;
    }
}

function resetAgentCards() {
    const agentCards = document.querySelectorAll('.agent-card');
    agentCards.forEach(card => {
        card.classList.remove('active', 'complete');
        const status = card.querySelector('.agent-status');
        if (status) status.textContent = 'Ready';
    });
}

function updateAgentStatus(agentName, status) {
    const agentCard = document.querySelector(`.agent-card.${agentName}`);
    if (agentCard) {
        agentCard.classList.add('active');
        agentCard.classList.remove('complete');
        
        const statusElement = agentCard.querySelector('.agent-status');
        if (statusElement) {
            statusElement.textContent = status === 'embedding_generation' ? 'Generating...' : 'Predicting...';
        }
    }
}

function updateAgentComplete(agentName, result) {
    const agentCard = document.querySelector(`.agent-card.${agentName}`);
    if (agentCard) {
        agentCard.classList.remove('active');
        agentCard.classList.add('complete');
        
        const statusElement = agentCard.querySelector('.agent-status');
        if (statusElement) {
            const predictionClass = result.prediction_class || 'NEUTRAL';
            statusElement.textContent = `✅ ${predictionClass}`;
        }
    }
}

function addTheaterResult(setupData) {
    state.theaterResults.push(setupData);
    
    // Could add visual summary of results here
    const resultSummary = document.createElement('div');
    resultSummary.className = 'theater-result-summary';
    resultSummary.innerHTML = `
        <strong>${setupData.setup_id}</strong>: 
        Ensemble: ${setupData.ensemble_prediction?.prediction_class || 'N/A'} 
        (Actual: ${setupData.actual_label !== null ? setupData.actual_label.toFixed(2) + '%' : 'Unknown'})
    `;
    
    const theaterLog = document.getElementById('theater-log');
    if (theaterLog) {
        theaterLog.appendChild(resultSummary);
    }
}

function completeTheater() {
    setTimeout(() => {
        stopLivePredictionTheater();
        showSuccess(`🎭 Theater complete! Processed ${state.theaterResults.length} setups.`);
    }, 1000);
}

function stopLivePredictionTheater() {
    state.isTheaterRunning = false;
    
    if (state.websocket) {
        state.websocket.close();
        state.websocket = null;
    }
    
    const startBtn = document.getElementById('start-theater-btn');
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>🎬</span> Start Live Prediction Theater';
    }
    
    // Reset agent cards
    resetAgentCards();
    
    console.log('🎭 Live Prediction Theater stopped');
}

// Setup Analysis Functions
function updateSetupButtons() {
    const predictBtn = document.getElementById('predict-btn');
    const similarBtn = document.getElementById('similar-btn');
    
    const hasSetup = state.selectedSetup && state.selectedSetup !== '';
    
    if (predictBtn) predictBtn.disabled = !hasSetup;
    if (similarBtn) similarBtn.disabled = !hasSetup;
}

function selectRandomSetup() {
    if (state.setups.length === 0) return;
    
    const randomSetup = state.setups[Math.floor(Math.random() * state.setups.length)];
    const setupSelect = document.getElementById('setup-select');
    
    if (setupSelect) {
        setupSelect.value = randomSetup.setup_id;
        state.selectedSetup = randomSetup.setup_id;
        updateSetupButtons();
    }
}

async function generatePrediction() {
    if (!state.selectedSetup) return;
    
    const predictBtn = document.getElementById('predict-btn');
    const resultsDiv = document.getElementById('prediction-results');
    
    try {
        if (predictBtn) {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<span class="spinner"></span> Generating Prediction...';
        }
        
        const prediction = await api.post('/api/predict', {
            setup_id: state.selectedSetup,
            include_similar: true,
            similarity_limit: 5
        });
        
        displayPredictionResults(prediction);
        
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showError(`Prediction failed: ${error.message}`);
    } finally {
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<span>🔮</span> Generate Enhanced Prediction';
        }
    }
}

function displayPredictionResults(prediction) {
    const pred = prediction.prediction;
    
    // Update metric cards
    updateElement('prediction-value', pred.prediction_class || 'N/A');
    updateElement('probability-value', pred.predicted_outperformance_10d ? 
        `${pred.predicted_outperformance_10d.toFixed(2)}%` : 'N/A');
    updateElement('actual-value', prediction.actual_label !== null ? 
        `${prediction.actual_label.toFixed(2)}%` : 'Unknown');
    updateElement('similar-count', prediction.similar_setups?.length || 0);
    
    // Update confidence
    const confidenceElement = document.getElementById('prediction-confidence');
    if (confidenceElement && pred.confidence_score) {
        confidenceElement.textContent = `${(pred.confidence_score * 100).toFixed(1)}% confidence`;
        confidenceElement.className = 'metric-change';
        if (pred.confidence_score > 0.7) confidenceElement.classList.add('positive');
        else if (pred.confidence_score < 0.4) confidenceElement.classList.add('negative');
    }
    
    // Update accuracy indicator
    const accuracyElement = document.getElementById('accuracy-indicator');
    if (accuracyElement && prediction.actual_label !== null) {
        const predicted = pred.predicted_outperformance_10d || 0;
        const actual = prediction.actual_label;
        const error = Math.abs(predicted - actual);
        
        accuracyElement.textContent = `±${error.toFixed(2)}% error`;
        accuracyElement.className = 'metric-change';
        if (error < 2) accuracyElement.classList.add('positive');
        else if (error > 5) accuracyElement.classList.add('negative');
    }
    
    // Update reasoning
    updateElement('ai-reasoning', pred.reasoning || 'No reasoning provided');
    
    // Store similar setups for potential display
    if (prediction.similar_setups) {
        state.similarSetups = prediction.similar_setups;
    }
}

async function findSimilarSetups() {
    if (!state.selectedSetup) return;
    
    const similarBtn = document.getElementById('similar-btn');
    const resultsDiv = document.getElementById('similar-results');
    
    try {
        if (similarBtn) {
            similarBtn.disabled = true;
            similarBtn.innerHTML = '<span class="spinner"></span> Finding Similar...';
        }
        
        const response = await api.get(`/api/setup/${state.selectedSetup}/similar?limit=10`);
        
        displaySimilarResults(response.similar_setups);
        
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (error) {
        console.error('Similar setups search failed:', error);
        showError(`Similar setups search failed: ${error.message}`);
    } finally {
        if (similarBtn) {
            similarBtn.disabled = false;
            similarBtn.innerHTML = '<span>🔍</span> Find Similar Setups';
        }
    }
}

function displaySimilarResults(similarSetups) {
    const tableBody = document.querySelector('#similar-table tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    similarSetups.forEach(setup => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${setup.setup_id}</td>
            <td class="number">${(setup.similarity_score * 100).toFixed(1)}%</td>
            <td class="number ${setup.outperformance_10d > 0 ? 'positive' : setup.outperformance_10d < 0 ? 'negative' : ''}">
                ${setup.outperformance_10d?.toFixed(2)}%
            </td>
            <td>
                <span class="prediction-class ${setup.prediction_class.toLowerCase()}">
                    ${setup.prediction_class}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
    
    state.similarSetups = similarSetups;
}

// Portfolio Scanner
async function scanPortfolio() {
    const scanBtn = document.getElementById('scan-btn');
    const resultsDiv = document.getElementById('scanner-results');
    
    try {
        if (scanBtn) {
            scanBtn.disabled = true;
            scanBtn.innerHTML = '<span class="spinner"></span> Scanning...';
        }
        
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const minProbability = parseFloat(document.getElementById('min-probability').value);
        
        // Mock scan results for now
        const mockResults = generateMockScanResults();
        
        displayScanResults(mockResults);
        
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (error) {
        console.error('Portfolio scan failed:', error);
        showError(`Portfolio scan failed: ${error.message}`);
    } finally {
        if (scanBtn) {
            scanBtn.disabled = false;
            scanBtn.innerHTML = '<span>🔍</span> Scan Portfolio';
        }
    }
}

function generateMockScanResults() {
    const results = [];
    const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'CRM'];
    
    for (let i = 0; i < 8; i++) {
        results.push({
            setup_id: `${tickers[i]}_2024-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`,
            probability: 0.5 + Math.random() * 0.4,
            confidence: 0.3 + Math.random() * 0.6,
            predicted_return: (Math.random() - 0.5) * 20,
            risk_level: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)]
        });
    }
    
    return results.sort((a, b) => b.probability - a.probability);
}

function displayScanResults(results) {
    const tableBody = document.querySelector('#scan-table tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    results.forEach(result => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${result.setup_id}</td>
            <td class="number">${(result.probability * 100).toFixed(1)}%</td>
            <td class="number">${(result.confidence * 100).toFixed(1)}%</td>
            <td class="number ${result.predicted_return > 0 ? 'positive' : 'negative'}">
                ${result.predicted_return.toFixed(2)}%
            </td>
            <td>
                <span class="risk-level ${result.risk_level.toLowerCase()}">
                    ${result.risk_level}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Model Performance Page
async function loadModelPerformance() {
    try {
        const performance = await api.get('/api/model-performance');
        displayModelPerformance(performance);
    } catch (error) {
        console.error('Failed to load model performance:', error);
        showError('Failed to load model performance data');
    }
}

function displayModelPerformance(performance) {
    const tableBody = document.querySelector('#model-performance-table tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    performance.models.forEach(model => {
        const row = document.createElement('tr');
        const isBest = model.name === performance.best_model;
        
        row.innerHTML = `
            <td><strong>${model.name}</strong> ${isBest ? '👑' : ''}</td>
            <td class="number">${model.precision.toFixed(3)}</td>
            <td class="number">${model.recall.toFixed(3)}</td>
            <td class="number">${model.f1_score.toFixed(3)}</td>
            <td class="number">${model.auc.toFixed(3)}</td>
            <td>
                <span class="status ${isBest ? 'best' : 'active'}">
                    ${isBest ? 'Best Model' : 'Active'}
                </span>
            </td>
        `;
        
        if (isBest) {
            row.style.background = 'var(--hover-bg)';
        }
        
        tableBody.appendChild(row);
    });
    
    // Update model metrics
    updateModelMetrics(performance);
}

function updateModelMetrics(performance) {
    const metricsContainer = document.getElementById('model-metrics');
    if (!metricsContainer) return;
    
    const bestModel = performance.models.find(m => m.name === performance.best_model);
    if (!bestModel) return;
    
    metricsContainer.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">👑</div>
            <div class="metric-change">${bestModel.name}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Precision</div>
            <div class="metric-value">${bestModel.precision.toFixed(3)}</div>
            <div class="metric-change positive">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best F1-Score</div>
            <div class="metric-value">${bestModel.f1_score.toFixed(3)}</div>
            <div class="metric-change positive">Balance</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Evaluation Date</div>
            <div class="metric-value">📅</div>
            <div class="metric-change">${new Date(performance.evaluation_date).toLocaleDateString()}</div>
        </div>
    `;
}

// Visualization Gallery
async function loadVisualizationGallery() {
    try {
        const vizData = await api.get('/api/visualizations');
        displayVisualizationGallery(vizData.categories);
    } catch (error) {
        console.error('Failed to load visualizations:', error);
        
        // Show placeholder content
        const gallery = document.getElementById('viz-gallery');
        if (gallery) {
            gallery.innerHTML = `
                <div class="text-center p-4">
                    <h3>📊 Analysis Visualizations</h3>
                    <p>Visualization gallery will display your ML analysis charts and reports.</p>
                    <p>Upload your analysis results to the <code>visualizations/</code> or <code>ml/analysis/</code> directory.</p>
                </div>
            `;
        }
    }
}

function displayVisualizationGallery(categories) {
    const gallery = document.getElementById('viz-gallery');
    const categoriesDiv = document.getElementById('viz-categories');
    
    if (!gallery || !categoriesDiv) return;
    
    // Display categories
    categoriesDiv.innerHTML = '';
    Object.keys(categories).forEach(category => {
        const btn = document.createElement('button');
        btn.className = 'secondary-btn';
        btn.textContent = `📊 ${category}`;
        btn.onclick = () => showVisualizationCategory(category, categories[category]);
        categoriesDiv.appendChild(btn);
    });
    
    // Display first category by default
    const firstCategory = Object.keys(categories)[0];
    if (firstCategory) {
        showVisualizationCategory(firstCategory, categories[firstCategory]);
    }
}

function showVisualizationCategory(category, files) {
    const gallery = document.getElementById('viz-gallery');
    if (!gallery) return;
    
    gallery.innerHTML = `
        <h3>📊 ${category} Analysis</h3>
        <div class="viz-grid">
            ${files.map(file => `
                <div class="viz-item">
                    <div class="viz-thumbnail">
                        ${file.type === 'image' ? 
                            `<img src="/api/visualization/${category}/${file.filename}" alt="${file.filename}">` :
                            `<div class="viz-text-icon">📄</div>`
                        }
                    </div>
                    <div class="viz-info">
                        <div class="viz-name">${file.filename}</div>
                        <div class="viz-size">${formatFileSize(file.size)}</div>
                        <div class="viz-date">${new Date(file.modified).toLocaleDateString()}</div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Data Explorer
async function loadDataExplorer() {
    // Initialize with fundamentals data
    showDataTab('fundamentals');
}

async function showDataTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const activeTab = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
        btn.getAttribute('onclick')?.includes(tabName)
    );
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    // Load data for the selected tab
    const tableTitle = document.querySelector('#data-content .table-title');
    const placeholder = document.getElementById('data-table-placeholder');
    
    if (tableTitle) {
        const titles = {
            'fundamentals': '🏛️ Fundamentals Features',
            'news': '📰 News Analysis Features',
            'userposts': '💬 Community Sentiment Features',
            'analyst': '📈 Analyst Coverage Features'
        };
        tableTitle.textContent = titles[tabName] || 'Data Features';
    }
    
    if (placeholder) {
        placeholder.innerHTML = `
            <div class="text-center p-4">
                <h3>${tableTitle?.textContent || 'Data Features'}</h3>
                <p>Interactive data exploration for ${tabName} coming soon...</p>
                <p>This will show comprehensive AI-extracted features from your pipeline.</p>
            </div>
        `;
    }
}

// Knowledge Graph Functions
function initializeKnowledgeGraph() {
    const container = document.getElementById('knowledge-graph-container');
    if (!container) return;
    
    container.innerHTML = `
        <div class="text-center p-4">
            <h3>🧠 AI Knowledge Graph</h3>
            <p>Knowledge graph visualization coming soon...</p>
            <p>This will show relationships between setups, features, and predictions.</p>
            <div class="mt-4">
                <div class="loading">
                    <span class="spinner"></span>
                    Initializing graph engine...
                </div>
            </div>
        </div>
    `;
    
    // TODO: Implement actual Cytoscape.js graph
}

function findSimilarNodes() {
    showInfo('Finding similar nodes in the knowledge graph...');
}

function exportGraph() {
    showInfo('Graph export functionality coming soon...');
}

// Health Monitoring
function startHealthMonitoring() {
    // Check health every 30 seconds
    setInterval(async () => {
        try {
            await checkHealth();
        } catch (error) {
            console.error('Health monitoring failed:', error);
        }
    }, 30000);
}

// Utility Functions
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<span class="spinner"></span> ${message}`;
    }
}

function showError(message) {
    console.error('Error:', message);
    
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'toast error';
    toast.innerHTML = `
        <span>❌</span>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function showSuccess(message) {
    console.log('Success:', message);
    
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'toast success';
    toast.innerHTML = `
        <span>✅</span>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function showInfo(message) {
    console.log('Info:', message);
    
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'toast info';
    toast.innerHTML = `
        <span>ℹ️</span>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// CSS for toast notifications
const toastStyles = `
<style>
.toast {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--card-background);
    color: var(--text-primary);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 8px 32px var(--shadow-color-lg);
    border: 2px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    z-index: 10000;
    animation: slideInRight 0.3s ease;
    max-width: 400px;
}

.toast.error {
    border-color: var(--error-color);
    background: rgba(239, 68, 68, 0.1);
}

.toast.success {
    border-color: var(--success-color);
    background: rgba(16, 185, 129, 0.1);
}

.toast.info {
    border-color: var(--info-color);
    background: rgba(6, 182, 212, 0.1);
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(100%);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', toastStyles);

console.log('🚀 Enhanced RAG Pipeline Frontend Application Loaded');