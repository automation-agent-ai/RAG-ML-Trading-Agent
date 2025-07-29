// Register Cytoscape extensions
if (typeof cytoscape !== 'undefined') {
    // Register cose-bilkent layout
    if (typeof cytoscapeCoseBilkent !== 'undefined') {
        cytoscape.use(cytoscapeCoseBilkent);
        console.log('✅ Registered cose-bilkent layout');
    } else if (typeof window.cytoscapeCoseBilkent !== 'undefined') {
        cytoscape.use(window.cytoscapeCoseBilkent);
        console.log('✅ Registered cose-bilkent layout (from window)');
    } else {
        console.warn('⚠️ cose-bilkent layout not found');
    }
    
    // Register dagre layout  
    if (typeof cytoscapeDagre !== 'undefined') {
        cytoscape.use(cytoscapeDagre);
        console.log('✅ Registered dagre layout');
    } else if (typeof window.cytoscapeDagre !== 'undefined') {
        cytoscape.use(window.cytoscapeDagre);
        console.log('✅ Registered dagre layout (from window)');
    } else {
        console.warn('⚠️ dagre layout not found');
    }
    
    // Register cola layout
    if (typeof cytoscapeCola !== 'undefined') {
        cytoscape.use(cytoscapeCola);
        console.log('✅ Registered cola layout');
    } else if (typeof window.cytoscapeCola !== 'undefined') {
        cytoscape.use(window.cytoscapeCola);
        console.log('✅ Registered cola layout (from window)');
    } else {
        console.warn('⚠️ cola layout not found');
    }
} else {
    console.error('❌ Cytoscape not loaded');
}

// Function to get available layout with fallback
function getAvailableLayout(preferredLayout = 'cose-bilkent') {
    if (typeof cytoscape === 'undefined') return 'grid';
    
    // Test if layout is available by trying to create a dummy instance
    try {
        const testCy = cytoscape({
            elements: [],
            layout: { name: preferredLayout }
        });
        testCy.destroy();
        console.log(`✅ Layout '${preferredLayout}' is available`);
        return preferredLayout;
    } catch (error) {
        console.warn(`⚠️ Layout '${preferredLayout}' not available, trying fallback...`);
        
        // Try fallback layouts in order of preference
        const fallbacks = ['dagre', 'cola', 'cose', 'circle', 'grid'];
        for (const layout of fallbacks) {
            try {
                const testCy = cytoscape({
                    elements: [],
                    layout: { name: layout }
                });
                testCy.destroy();
                console.log(`✅ Using fallback layout '${layout}'`);
                return layout;
            } catch (e) {
                console.warn(`⚠️ Layout '${layout}' also not available`);
            }
        }
        
        console.error('❌ No layouts available, using grid as last resort');
        return 'grid';
    }
}

// Global state management
let state = {
    selectedSetup: null,
    setups: [],
    healthStatus: 'connecting',
    currentPage: 'main',
    currentTheme: 'default'
};

// API configuration
const API_BASE = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing RAG Pipeline Frontend...');
    
    // Initialize theme system
    initializeTheme();
    
    // Initialize navigation
    setupNavigation();
    
    // Check health status
    await checkHealth();
    
    // Load initial data
    await loadSetups();
    await loadPerfectSetup();
    
    // Setup event listeners
    setupEventListeners();
    
    console.log('✅ Frontend initialization complete');
});

// Theme system functions
function initializeTheme() {
    const themeSelector = document.getElementById('theme-selector');
    const savedTheme = localStorage.getItem('selected-theme') || 'default';
    
    // Apply saved theme
    applyTheme(savedTheme);
    themeSelector.value = savedTheme;
    
    // Theme selector event listener
    themeSelector.addEventListener('change', (e) => {
        const newTheme = e.target.value;
        applyTheme(newTheme);
        localStorage.setItem('selected-theme', newTheme);
    });
}

function applyTheme(theme) {
    state.currentTheme = theme;
    
    if (theme === 'default') {
        document.body.removeAttribute('data-theme');
    } else {
        document.body.setAttribute('data-theme', theme);
    }
    
    console.log(`🎨 Applied theme: ${theme}`);
}

// Navigation functions
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const page = e.target.getAttribute('onclick').match(/'(.+)'/)[1];
            showPage(page);
        });
    });
}

function showPage(pageId) {
    // Update navigation state
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[onclick="showPage('${pageId}')"]`).classList.add('active');
    
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(page => page.classList.remove('active'));
    
    // Show selected page
    const targetPage = document.getElementById(`${pageId}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
        state.currentPage = pageId;
        
        // Load page-specific data
        switch(pageId) {
            case 'data':
                if (state.selectedSetup) {
                    loadDataForSelectedSetup();
                }
                break;
            case 'metrics':
                loadModelEvaluationTables();
                break;
            case 'visualizations':
                loadVisualizationsForPage();
                break;
            case 'knowledge-graph':
                console.log('🚀 Knowledge Graph tab clicked, initializing page...');
                // Add visual debugging to the page
                const kgStatusDiv = document.getElementById('kg-status-text');
                if (kgStatusDiv) {
                    kgStatusDiv.textContent = 'Initializing page...';
                }
                initializeKnowledgeGraphPage();
                break;
        }
    }
}

// Health check function
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const health = await response.json();
        
        const indicator = document.getElementById('health-indicator');
        const statusText = document.getElementById('health-status');
        const statusDot = indicator.querySelector('.status-dot');
        
        if (health.status === 'healthy') {
            statusText.textContent = 'System Healthy';
            statusDot.className = 'status-dot healthy';
            state.healthStatus = 'healthy';
        } else {
            statusText.textContent = 'System Warning';
            statusDot.className = 'status-dot error';
            state.healthStatus = 'warning';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const indicator = document.getElementById('health-indicator');
        const statusText = document.getElementById('health-status');
        const statusDot = indicator.querySelector('.status-dot');
        
        statusText.textContent = 'Connection Error';
        statusDot.className = 'status-dot error';
        state.healthStatus = 'error';
    }
}

// Setup selection functions
async function loadSetups() {
    try {
        const response = await fetch(`${API_BASE}/api/setups?limit=100`);
        const data = await response.json();
        
        state.setups = data.setups || [];
        
        const select = document.getElementById('setup-select');
        select.innerHTML = '<option value="">Select a setup...</option>';
        
        state.setups.forEach(setup => {
            const option = document.createElement('option');
            option.value = setup.setup_id;
            option.textContent = `${setup.setup_id} (${setup.company_name || 'Unknown'})`;
            select.appendChild(option);
        });
        
        console.log(`✅ Loaded ${state.setups.length} setups`);
    } catch (error) {
        console.error('Failed to load setups:', error);
        document.getElementById('setup-select').innerHTML = '<option value="">Error loading setups</option>';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Setup selection
    const setupSelect = document.getElementById('setup-select');
    setupSelect.addEventListener('change', (e) => {
        const setupId = e.target.value;
        if (setupId) {
            state.selectedSetup = setupId;
            document.getElementById('predict-btn').disabled = false;
            document.getElementById('explain-btn').disabled = false;
            
            // Load data for current page if on data page
            if (state.currentPage === 'data') {
                loadDataForSelectedSetup();
            }
        } else {
            state.selectedSetup = null;
            document.getElementById('predict-btn').disabled = true;
            document.getElementById('explain-btn').disabled = true;
        }
    });
    
    // Prediction button
    document.getElementById('predict-btn').addEventListener('click', generatePrediction);
    
    // Explanation button
    document.getElementById('explain-btn').addEventListener('click', generateExplanation);
    
    // Scanner buttons
    document.getElementById('scan-btn').addEventListener('click', runScanner);
    document.getElementById('scan-today-btn').addEventListener('click', scanToday);
}

// Prediction functions
async function generatePrediction() {
    if (!state.selectedSetup) return;
    
    const button = document.getElementById('predict-btn');
    const results = document.getElementById('prediction-results');
    
    try {
        button.disabled = true;
        button.textContent = '🔄 Generating...';
        
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ setup_id: state.selectedSetup })
        });
        
        const prediction = await response.json();
        
        if (response.ok) {
            displayPredictionResults(prediction);
            results.style.display = 'block';
        } else {
            throw new Error(prediction.detail || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Prediction failed: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = '🔮 Generate Prediction';
    }
}

function displayPredictionResults(prediction) {
    // Update prediction metrics
    const predictionValue = document.getElementById('prediction-value');
    const probabilityValue = document.getElementById('probability-value');
    const confidenceValue = document.getElementById('confidence-value');
    
    predictionValue.textContent = prediction.prediction === 1 ? 'Outperform' : 'Underperform';
    predictionValue.className = `metric-value ${prediction.prediction === 1 ? 'positive' : 'negative'}`;
    
    probabilityValue.textContent = (prediction.probability * 100).toFixed(1) + '%';
    confidenceValue.textContent = (prediction.confidence * 100).toFixed(1) + '%';
    
    // Add model information and threshold
    const resultsContainer = document.getElementById('prediction-results');
    
    // Create or update ground truth section
    let groundTruthSection = resultsContainer.querySelector('.ground-truth-section');
    if (!groundTruthSection) {
        groundTruthSection = document.createElement('div');
        groundTruthSection.className = 'ground-truth-section';
        groundTruthSection.style.cssText = `
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 8px;
            background: var(--surface-color);
            border: 1px solid var(--border-color);
        `;
        resultsContainer.appendChild(groundTruthSection);
    }
    
    // Display model information
    let modelInfo = `
        <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">
            🤖 Model Details
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin-bottom: 1rem;">
            <div><strong>Model:</strong> ${prediction.model_type || 'ensemble_lr_calibrated'}</div>
            <div><strong>Threshold:</strong> ${((prediction.threshold_used || 0.5) * 100).toFixed(1)}%</div>
        </div>
    `;
    
    // Display ground truth if available
    if (prediction.ground_truth !== null && prediction.ground_truth !== undefined) {
        const actualPerf = prediction.outperformance_10d;
        const isCorrect = prediction.prediction_correct;
        const correctIcon = isCorrect ? '✅' : '❌';
        const correctColor = isCorrect ? 'var(--success-color)' : 'var(--error-color)';
        
        modelInfo += `
            <h4 style="margin: 1rem 0 0.5rem 0; color: var(--text-primary);">
                📊 Ground Truth Validation
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem;">
                <div><strong>Actual Performance:</strong> ${prediction.ground_truth}</div>
                <div><strong>Actual Return:</strong> ${actualPerf !== null ? (actualPerf * 100).toFixed(2) + '%' : 'N/A'}</div>
                <div style="color: ${correctColor};">
                    <strong>Prediction Result:</strong> ${correctIcon} ${isCorrect ? 'Correct' : 'Incorrect'}
                </div>
            </div>
        `;
    } else {
        modelInfo += `
            <div style="color: var(--text-secondary); font-style: italic; margin-top: 1rem;">
                📊 Ground truth data not available for this setup
            </div>
        `;
    }
    
    groundTruthSection.innerHTML = modelInfo;
    
    // Update factors
    const keyFactors = prediction.key_factors || ['Market position', 'Financial stability'];
    const riskFactors = prediction.risk_factors || ['Market volatility', 'Sector risks'];
    
    document.getElementById('key-factors').innerHTML = 
        keyFactors.map(factor => `<p>✅ ${factor}</p>`).join('');
    
    document.getElementById('risk-factors').innerHTML = 
        riskFactors.map(factor => `<p>⚠️ ${factor}</p>`).join('');
}

// Data loading functions for Data Analysis page
async function loadDataForSelectedSetup() {
    if (!state.selectedSetup) return;
    
    console.log(`📊 Loading data for setup: ${state.selectedSetup}`);
    
    try {
        // Load fundamentals data
        await loadFundamentalsData();
        // Load news data
        await loadNewsData();
        // Load userposts data
        await loadUserPostsData();
    } catch (error) {
        console.error('Error loading data for setup:', error);
    }
}

async function loadFundamentalsData() {
    try {
        console.log(`📈 Loading fundamentals data for ${state.selectedSetup}`);
        const response = await fetch(`${API_BASE}/api/setup/${state.selectedSetup}/fundamentals`);
        const data = await response.json();
        
        const container = document.getElementById('fundamentals-data');
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load fundamentals data');
        }
        
        if (data.length === 0) {
            container.innerHTML = '<p>No fundamentals data available for this setup.</p>';
            return;
        }
        
        const table = createDataTable(data, 'Fundamentals Data');
        container.innerHTML = '';
        container.appendChild(table);
        console.log(`✅ Loaded ${data.length} fundamentals records`);
    } catch (error) {
        console.error('Error loading fundamentals data:', error);
        document.getElementById('fundamentals-data').innerHTML = `<p>Error loading fundamentals data: ${error.message}</p>`;
    }
}

async function loadNewsData() {
    try {
        console.log(`📰 Loading news data for ${state.selectedSetup}`);
        const response = await fetch(`${API_BASE}/api/setup/${state.selectedSetup}/news`);
        const data = await response.json();
        
        const container = document.getElementById('news-data');
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load news data');
        }
        
        if (data.length === 0) {
            container.innerHTML = '<p>No news data available for this setup.</p>';
            return;
        }
        
        const table = createDataTable(data, 'RNS News Data');
        container.innerHTML = '';
        container.appendChild(table);
        console.log(`✅ Loaded ${data.length} news records`);
    } catch (error) {
        console.error('Error loading news data:', error);
        document.getElementById('news-data').innerHTML = `<p>Error loading news data: ${error.message}</p>`;
    }
}

async function loadUserPostsData() {
    try {
        console.log(`👥 Loading user posts data for ${state.selectedSetup}`);
        const response = await fetch(`${API_BASE}/api/setup/${state.selectedSetup}/userposts`);
        const data = await response.json();
        
        const container = document.getElementById('userposts-data');
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load userposts data');
        }
        
        if (data.length === 0) {
            container.innerHTML = '<p>No user posts data available for this setup.</p>';
            return;
        }
        
        const table = createDataTable(data, 'User Posts Data');
        container.innerHTML = '';
        container.appendChild(table);
        console.log(`✅ Loaded ${data.length} user posts records`);
    } catch (error) {
        console.error('Error loading user posts data:', error);
        document.getElementById('userposts-data').innerHTML = `<p>Error loading user posts data: ${error.message}</p>`;
    }
}

function createDataTable(data, title) {
    if (!data || data.length === 0) return document.createElement('div');
    
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');
    
    // Create header
    const headerRow = document.createElement('tr');
    const keys = Object.keys(data[0]);
    keys.forEach(key => {
        const th = document.createElement('th');
        th.textContent = key.replace(/_/g, ' ').toUpperCase();
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    
    // Create rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        keys.forEach(key => {
            const td = document.createElement('td');
            const value = row[key];
            
            if (typeof value === 'number') {
                td.textContent = value.toFixed(4);
            } else if (value === null || value === undefined) {
                td.textContent = '-';
                td.style.color = '#9ca3af';
            } else {
                td.textContent = value;
            }
            
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    
    table.appendChild(thead);
    table.appendChild(tbody);
    
    return table;
}

// Model evaluation functions
async function loadModelEvaluationTables() {
    try {
        await loadModelPerformanceTable();
        await loadFeatureImportanceTable();
    } catch (error) {
        console.error('Error loading model evaluation tables:', error);
    }
}

async function loadModelPerformanceTable() {
    try {
        // Load reports from all three categories
        const [fundamentalsReport, textReport, ensembleReport] = await Promise.all([
            fetch(`${API_BASE}/api/reports/analysis_ml_fundamentals`).then(r => r.json()),
            fetch(`${API_BASE}/api/reports/analysis_ml_text`).then(r => r.json()),
            fetch(`${API_BASE}/api/reports/analysis_ml_ensemble`).then(r => r.json())
        ]);
        
        const container = document.getElementById('model-performance-table');
        const table = parseModelPerformanceFromReports(
            fundamentalsReport.content, 
            textReport.content, 
            ensembleReport.content
        );
        container.innerHTML = '';
        container.appendChild(table);
    } catch (error) {
        console.error('Error loading model performance table:', error);
        document.getElementById('model-performance-table').innerHTML = '<p>Error loading model performance data.</p>';
    }
}

function parseModelPerformanceFromReports(fundamentalsReport, textReport, ensembleReport) {
    const models = [];
    
    // Parse fundamentals models
    const fundModels = extractModelsFromReport(fundamentalsReport, 'Fundamentals');
    models.push(...fundModels);
    
    // Parse text models  
    const textModels = extractModelsFromReport(textReport, 'Text');
    models.push(...textModels);
    
    // Parse ensemble models
    const ensembleModels = extractModelsFromReport(ensembleReport, 'Ensemble');
    models.push(...ensembleModels);
    
    return createModelPerformanceTable(models);
}

function extractModelsFromReport(reportText, category) {
    const models = [];
    const lines = reportText.split('\n');
    
    const modelTypes = ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression'];
    
    modelTypes.forEach(modelType => {
        const modelData = { name: `${modelType} (${category})`, category };
        
        // Find the model section
        const modelStart = lines.findIndex(line => line.includes(modelType));
        if (modelStart === -1) return;
        
        // Extract metrics
        for (let i = modelStart; i < modelStart + 15 && i < lines.length; i++) {
            const line = lines[i];
            
            if (line.includes('Precision:')) {
                const match = line.match(/(\d+\.\d+)\s*±\s*(\d+\.\d+)/);
                if (match) {
                    modelData.precision = { value: parseFloat(match[1]), std: parseFloat(match[2]) };
                }
            } else if (line.includes('Recall:')) {
                const match = line.match(/(\d+\.\d+)\s*±\s*(\d+\.\d+)/);
                if (match) {
                    modelData.recall = { value: parseFloat(match[1]), std: parseFloat(match[2]) };
                }
            } else if (line.includes('F1-Score:')) {
                const match = line.match(/(\d+\.\d+)\s*±\s*(\d+\.\d+)/);
                if (match) {
                    modelData.f1 = { value: parseFloat(match[1]), std: parseFloat(match[2]) };
                }
            } else if (line.includes('AUC:')) {
                const match = line.match(/(\d+\.\d+)\s*±\s*(\d+\.\d+)/);
                if (match) {
                    modelData.auc = { value: parseFloat(match[1]), std: parseFloat(match[2]) };
                }
            }
        }
        
        if (Object.keys(modelData).length > 2) {
            models.push(modelData);
        }
    });
    
    return models;
}

function createModelPerformanceTable(models) {
    const table = document.createElement('table');
    table.className = 'performance-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    ['Model', 'Precision', 'Recall', 'F1-Score', 'AUC'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    
    models.forEach(model => {
        const row = document.createElement('tr');
        
        // Model name
        const nameCell = document.createElement('td');
        nameCell.textContent = model.name;
        nameCell.style.fontWeight = '600';
        row.appendChild(nameCell);
        
        // Metrics
        ['precision', 'recall', 'f1', 'auc'].forEach(metric => {
            const cell = document.createElement('td');
            cell.className = 'metric-cell';
            
            if (model[metric]) {
                const mainValue = document.createElement('div');
                mainValue.className = 'metric-main';
                mainValue.textContent = model[metric].value.toFixed(3);
                
                const stdValue = document.createElement('div');
                stdValue.className = 'metric-std';
                stdValue.textContent = `±${model[metric].std.toFixed(3)}`;
                
                cell.appendChild(mainValue);
                cell.appendChild(stdValue);
            } else {
                cell.textContent = '-';
            }
            
            row.appendChild(cell);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    return table;
}

async function loadFeatureImportanceTable() {
    try {
        // For now, create a mock feature importance table
        // In a real implementation, you'd parse this from the reports or get it from an API
        const container = document.getElementById('feature-importance-table');
        const table = createFeatureImportanceTable();
        container.innerHTML = '';
        container.appendChild(table);
    } catch (error) {
        document.getElementById('feature-importance-table').innerHTML = '<p>Error loading feature importance data.</p>';
    }
}

function createFeatureImportanceTable() {
    const features = [
        { name: 'Market Cap', importance: 0.15, stability: 0.92 },
        { name: 'P/E Ratio', importance: 0.13, stability: 0.88 },
        { name: 'Revenue Growth', importance: 0.12, stability: 0.85 },
        { name: 'Debt to Equity', importance: 0.11, stability: 0.89 },
        { name: 'ROE', importance: 0.10, stability: 0.87 },
        { name: 'Current Ratio', importance: 0.09, stability: 0.91 },
        { name: 'Gross Margin', importance: 0.08, stability: 0.84 },
        { name: 'EBITDA Margin', importance: 0.07, stability: 0.86 },
        { name: 'Price to Book', importance: 0.08, stability: 0.82 },
        { name: 'Free Cash Flow', importance: 0.07, stability: 0.90 }
    ];
    
    const table = document.createElement('table');
    table.className = 'feature-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    ['Feature', 'Importance', 'Stability Score'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    
    features.forEach(feature => {
        const row = document.createElement('tr');
        
        // Feature name
        const nameCell = document.createElement('td');
        nameCell.textContent = feature.name;
        nameCell.style.fontWeight = '600';
        row.appendChild(nameCell);
        
        // Importance with bar
        const importanceCell = document.createElement('td');
        const importanceContainer = document.createElement('div');
        importanceContainer.style.display = 'flex';
        importanceContainer.style.alignItems = 'center';
        importanceContainer.style.gap = '0.5rem';
        
        const importanceText = document.createElement('span');
        importanceText.textContent = (feature.importance * 100).toFixed(1) + '%';
        importanceText.style.minWidth = '3rem';
        
        const importanceBar = document.createElement('div');
        importanceBar.className = 'importance-bar';
        importanceBar.style.flex = '1';
        
        const importanceFill = document.createElement('div');
        importanceFill.className = 'importance-fill';
        importanceFill.style.width = (feature.importance * 100) + '%';
        
        importanceBar.appendChild(importanceFill);
        importanceContainer.appendChild(importanceText);
        importanceContainer.appendChild(importanceBar);
        importanceCell.appendChild(importanceContainer);
        row.appendChild(importanceCell);
        
        // Stability score
        const stabilityCell = document.createElement('td');
        stabilityCell.textContent = (feature.stability * 100).toFixed(1) + '%';
        stabilityCell.style.textAlign = 'center';
        row.appendChild(stabilityCell);
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    return table;
}

// Perfect setup functions
async function loadPerfectSetup() {
    try {
        const response = await fetch(`${API_BASE}/api/perfect_setup`);
        const data = await response.json();
        
        const container = document.getElementById('perfect-setup');
        container.innerHTML = '';
        
        if (data.perfect_features && Object.keys(data.perfect_features).length > 0) {
            Object.entries(data.perfect_features).forEach(([feature, value]) => {
                const metricDiv = document.createElement('div');
                metricDiv.className = 'perfect-metric';
                
                const label = document.createElement('span');
                label.className = 'label';
                label.textContent = feature.replace(/_/g, ' ').toUpperCase();
                
                const valueSpan = document.createElement('span');
                valueSpan.className = 'value';
                valueSpan.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                
                metricDiv.appendChild(label);
                metricDiv.appendChild(valueSpan);
                container.appendChild(metricDiv);
            });
        } else {
            container.innerHTML = '<p>No perfect setup data available.</p>';
        }
    } catch (error) {
        document.getElementById('perfect-setup').innerHTML = '<p>Error loading perfect setup data.</p>';
    }
}

// Knowledge Graph explanation
async function generateExplanation() {
    if (!state.selectedSetup) return;
    
    const button = document.getElementById('explain-btn');
    const results = document.getElementById('explanation-results');
    const content = document.getElementById('explanation-content');
    
    try {
        button.disabled = true;
        button.textContent = '🔄 Generating...';
        
        const response = await fetch(`${API_BASE}/api/explain/${state.selectedSetup}`, {
            method: 'POST'
        });
        
        const explanation = await response.json();
        
        if (response.ok) {
            content.innerHTML = formatExplanation(explanation.explanation);
            results.style.display = 'block';
        } else {
            throw new Error(explanation.detail || 'Explanation failed');
        }
    } catch (error) {
        console.error('Explanation error:', error);
        content.innerHTML = `<p>Error generating explanation: ${error.message}</p>`;
        results.style.display = 'block';
    } finally {
        button.disabled = false;
        button.textContent = '💡 Generate Explanation';
    }
}

function formatExplanation(explanation) {
    return explanation
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n- /g, '</p><ul><li>')
        .replace(/\n([^-])/g, '</li></ul><p>$1')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

// Scanner functions
async function runScanner() {
    const dateFrom = document.getElementById('scanner-date-from').value;
    const dateTo = document.getElementById('scanner-date-to').value;
    const minProb = parseFloat(document.getElementById('scanner-min-prob').value);
    const button = document.getElementById('scan-btn');
    const results = document.getElementById('scanner-results');
    const output = document.getElementById('scanner-output');
    
    if (!dateFrom || !dateTo) {
        alert('Please select both start and end dates');
        return;
    }
    
    if (new Date(dateFrom) > new Date(dateTo)) {
        alert('Start date must be before end date');
        return;
    }
    
    try {
        button.disabled = true;
        button.textContent = '🔄 Scanning...';
        
        const response = await fetch(`${API_BASE}/api/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                date_from: dateFrom, 
                date_to: dateTo, 
                min_probability: minProb 
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayScannerResults(result);
            results.style.display = 'block';
        } else {
            throw new Error(result.detail || 'Scanner failed');
        }
    } catch (error) {
        console.error('Scanner error:', error);
        output.innerHTML = `<p>Scanner failed: ${error.message}</p>`;
        results.style.display = 'block';
    } finally {
        button.disabled = false;
        button.textContent = '🔍 Scan Date Range';
    }
}

async function scanToday() {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('scanner-date-from').value = today;
    document.getElementById('scanner-date-to').value = today;
    await runScanner();
}

function displayScannerResults(result) {
    const output = document.getElementById('scanner-output');
    
    if (result.potential_setups && result.potential_setups.length > 0) {
        let html = `
            <div class="scanner-summary">
                <h3>📊 Scanner Results</h3>
                <p><strong>Scanned:</strong> ${result.scanned_setups} setups</p>
                <p><strong>Found:</strong> ${result.potential_setups.length} high-probability setups</p>
                <p><strong>Threshold:</strong> ${(result.threshold * 100).toFixed(1)}%</p>
            </div>
            <div class="scanner-table">
                <table>
                    <thead>
                        <tr>
                            <th>Setup ID</th>
                            <th>Ticker</th>
                            <th>Date</th>
                            <th>Probability</th>
                            <th>Prediction</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        result.potential_setups.forEach(setup => {
            const probPercent = (setup.probability * 100).toFixed(1);
            const statusIcon = setup.confirmed ? '✅' : '⏳';
            const statusText = setup.confirmed ? 'Confirmed' : 'Pending';
            
            html += `
                <tr>
                    <td>${setup.setup_id}</td>
                    <td>${setup.stock_ticker}</td>
                    <td>${setup.setup_date}</td>
                    <td>${probPercent}%</td>
                    <td>${setup.prediction ? 'Outperform' : 'Underperform'}</td>
                    <td>${statusIcon} ${statusText}</td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        output.innerHTML = html;
    } else {
        output.innerHTML = `
            <div class="scanner-summary">
                <h3>🔍 Scanner Results</h3>
                <p>No high-probability setups found in the specified date range.</p>
                <p><strong>Scanned:</strong> ${result.scanned_setups || 0} setups</p>
                <p><strong>Threshold:</strong> ${(result.threshold * 100).toFixed(1)}%</p>
                <p><em>Try lowering the minimum probability threshold or expanding the date range.</em></p>
            </div>
        `;
    }
}

// Visualizations functions
async function loadVisualizationsForPage() {
    try {
        const response = await fetch(`${API_BASE}/api/visualizations`);
        const data = await response.json();
        
        // Load fundamentals visualizations
        loadCategoryVisualizations('analysis_ml_fundamentals', data, 'fundamentals-visualizations');
        
        // Load text visualizations
        loadCategoryVisualizations('analysis_ml_text', data, 'text-visualizations');
        
        // Load ensemble visualizations
        loadCategoryVisualizations('analysis_ml_ensemble', data, 'ensemble-visualizations');
        
        console.log('✅ Loaded visualizations for all categories');
    } catch (error) {
        console.error('Error loading visualizations:', error);
        ['fundamentals-visualizations', 'text-visualizations', 'ensemble-visualizations'].forEach(containerId => {
            document.getElementById(containerId).innerHTML = '<p>Error loading visualizations.</p>';
        });
    }
}

function loadCategoryVisualizations(categoryName, data, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (data[categoryName] && data[categoryName].png_files) {
        const pngFiles = data[categoryName].png_files;
        
        pngFiles.forEach(file => {
            const vizItem = document.createElement('div');
            vizItem.className = 'viz-item';
            
            const img = document.createElement('img');
            img.src = `${API_BASE}/api/visualization/${categoryName}/${file}`;
            img.alt = file;
            img.loading = 'lazy';
            img.onerror = () => {
                img.style.display = 'none';
                vizItem.innerHTML = '<p>Error loading visualization</p>';
            };
            
            const title = document.createElement('h4');
            title.textContent = file.replace('.png', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            vizItem.appendChild(img);
            vizItem.appendChild(title);
            container.appendChild(vizItem);
        });
        
        console.log(`✅ Loaded ${pngFiles.length} visualizations for ${categoryName}`);
    } else {
        container.innerHTML = '<p>No visualizations available for this category.</p>';
    }
}

// Utility functions
function showError(message) {
    alert(`Error: ${message}`);
}

function showLoading(element) {
    element.classList.add('loading');
}

function hideLoading(element) {
    element.classList.remove('loading');
}

// ====== KNOWLEDGE GRAPH FUNCTIONS ======

// Knowledge Graph state
let kgState = {
    initialized: false,
    selectedSetup: null,
    graphData: null,
    currentVisualization: null,
    status: 'initializing'
};

// Initialize Knowledge Graph page
async function initializeKnowledgeGraphPage() {
    console.log('🧠 Initializing Knowledge Graph page...');
    
    // Add alert for debugging
    alert('🧠 initializeKnowledgeGraphPage called!');
    
    // Load setups for KG selector
    alert('📋 About to load KG setups...');
    await loadKGSetups();
    alert('✅ KG setups loaded!');
    
    // Check KG status with timeout
    alert('🔍 About to check KG status...');
    try {
        await Promise.race([
            checkKGStatus(),
            new Promise((_, reject) => setTimeout(() => reject(new Error('KG status check timeout')), 5000))
        ]);
        alert('✅ KG status checked!');
    } catch (error) {
        alert(`⚠️ KG status check failed/timeout: ${error.message} - continuing anyway...`);
    }
    
    // Setup event listeners
    alert('🔧 About to setup KG event listeners...');
    console.log('🔧 About to setup KG event listeners...');
    try {
        setupKGEventListeners();
        alert('✅ Event listeners setup complete!');
        console.log('✅ KG event listeners setup complete');
    } catch (error) {
        alert(`❌ Error setting up event listeners: ${error.message}`);
        console.error('❌ Error setting up event listeners:', error);
    }
    
    // Show loading state
    showKGLoading();
    
    kgState.initialized = true;
    console.log('✅ Knowledge Graph page initialized');
}

// Load setups for KG selector
async function loadKGSetups() {
    try {
        const response = await fetch(`${API_BASE}/api/setups?limit=100`);
        const data = await response.json();
        
        const kgSetupSelect = document.getElementById('kg-setup-select');
        kgSetupSelect.innerHTML = '<option value="">Choose a setup to explore...</option>';
        
        data.setups.forEach(setup => {
            const option = document.createElement('option');
            option.value = setup.setup_id;
            option.textContent = `${setup.setup_id} (${setup.company_name || 'Unknown'})`;
            kgSetupSelect.appendChild(option);
        });
        
        console.log(`✅ Loaded ${data.setups.length} setups for KG`);
    } catch (error) {
        console.error('Failed to load KG setups:', error);
    }
}

// Check Knowledge Graph status
async function checkKGStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/kg/status`);
        const data = await response.json();
        
        alert(`🔍 KG Status Response: ${data.initialization_status}`);
        
        updateKGStatus(data);
        
        // If not initialized, start initialization
        if (data.initialization_status === 'not_initialized') {
            alert('🚀 KG not initialized, starting initialization...');
            await initializeKG();
        } else if (data.initialization_status === 'completed') {
            // KG is already ready, hide loading
            alert('✅ KG already completed, hiding loading...');
            hideKGLoading();
        } else {
            alert(`⏳ KG Status: ${data.initialization_status} - no action taken`);
        }
        
    } catch (error) {
        console.error('Failed to check KG status:', error);
        alert(`❌ Error checking KG status: ${error.message}`);
        updateKGStatus({
            initialization_status: 'error',
            total_nodes: 0,
            total_edges: 0
        });
        hideKGLoading();
    }
}

// Initialize Knowledge Graph
async function initializeKG() {
    try {
        updateKGStatus({ initialization_status: 'initializing' });
        
        const response = await fetch(`${API_BASE}/api/kg/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ force_rebuild: false })
        });
        
        const data = await response.json();
        console.log('KG initialization started:', data);
        
        // Poll for completion
        pollKGStatus();
        
    } catch (error) {
        console.error('Failed to initialize KG:', error);
        updateKGStatus({ initialization_status: 'error' });
    }
}

// Poll KG status until ready
async function pollKGStatus() {
    const maxAttempts = 30;
    let attempts = 0;
    
    const poll = async () => {
        attempts++;
        
        try {
            const response = await fetch(`${API_BASE}/api/kg/status`);
            const data = await response.json();
            
            updateKGStatus(data);
            
            if (data.initialization_status === 'completed') {
                console.log('✅ Knowledge Graph ready!');
                hideKGLoading();
                return;
            }
            
            if (data.initialization_status === 'error' || attempts >= maxAttempts) {
                console.error('❌ Knowledge Graph initialization failed or timed out');
                hideKGLoading();
                return;
            }
            
            // Continue polling
            setTimeout(poll, 2000);
            
        } catch (error) {
            console.error('Error polling KG status:', error);
        }
    };
    
    poll();
}

// Update KG status display
function updateKGStatus(statusData) {
    const statusText = document.getElementById('kg-status-text');
    const statusDot = document.querySelector('.status-dot');
    
    kgState.status = statusData.initialization_status;
    
    switch (statusData.initialization_status) {
        case 'completed':
            statusText.textContent = `Ready (${statusData.total_nodes} nodes, ${statusData.total_edges} edges)`;
            statusDot.style.background = 'var(--success-color)';
            statusDot.style.animation = 'none';
            break;
        case 'initializing':
            statusText.textContent = 'Initializing knowledge graph...';
            statusDot.style.background = 'var(--warning-color)';
            statusDot.style.animation = 'pulse 2s infinite';
            break;
        case 'error':
            statusText.textContent = 'Initialization failed';
            statusDot.style.background = 'var(--error-color)';
            statusDot.style.animation = 'none';
            break;
        default:
            statusText.textContent = 'Unknown status';
            statusDot.style.background = 'var(--text-secondary)';
            break;
    }
}

// Setup KG event listeners
function setupKGEventListeners() {
    console.log('🔧 Setting up KG event listeners...');
    
    // Check if elements exist
    const setupSelect = document.getElementById('kg-setup-select');
    const exploreBtn = document.getElementById('kg-explore-btn');
    
    console.log('Setup select element:', setupSelect);
    console.log('Explore button element:', exploreBtn);
    
    // Add visual debugging to the page
    const kgStatusDiv = document.getElementById('kg-status-text');
    if (kgStatusDiv) {
        kgStatusDiv.textContent = 'Setting up event listeners...';
        kgStatusDiv.style.color = 'orange';
        kgStatusDiv.style.fontWeight = 'bold';
    }
    
    // Add alert for debugging
    console.log('🚨 ALERT: setupKGEventListeners called!');
    
    if (!setupSelect) {
        console.error('❌ kg-setup-select element not found!');
        if (kgStatusDiv) {
            kgStatusDiv.textContent = 'ERROR: Setup select not found!';
            kgStatusDiv.style.color = 'red';
        }
        alert('ERROR: Setup select element not found!');
        return;
    }
    
    if (!exploreBtn) {
        console.error('❌ kg-explore-btn element not found!');
        if (kgStatusDiv) {
            kgStatusDiv.textContent = 'ERROR: Explore button not found!';
            kgStatusDiv.style.color = 'red';
        }
        alert('ERROR: Explore button element not found!');
        return;
    }
    
    // Show success message
    if (kgStatusDiv) {
        kgStatusDiv.textContent = 'Event listeners attached! Select a setup.';
        kgStatusDiv.style.color = 'green';
    }
    
    // Setup selection
    setupSelect.addEventListener('change', (e) => {
        const setupId = e.target.value;
        console.log('🔍 KG Setup selected:', setupId);
        
        // Add alert for debugging
        alert(`Setup selected: ${setupId}`);
        
        if (setupId) {
            kgState.selectedSetup = setupId;
            exploreBtn.disabled = false;
            exploreBtn.style.backgroundColor = '#6366f1';
            exploreBtn.style.cursor = 'pointer';
            console.log('✅ Explore Graph button enabled');
            // Visual feedback
            if (kgStatusDiv) {
                kgStatusDiv.textContent = `Setup ${setupId} selected. Button enabled!`;
                kgStatusDiv.style.color = 'green';
            }
        } else {
            kgState.selectedSetup = null;
            exploreBtn.disabled = true;
            exploreBtn.style.backgroundColor = '#6b7280';
            exploreBtn.style.cursor = 'not-allowed';
            console.log('❌ Explore Graph button disabled');
            // Visual feedback
            if (kgStatusDiv) {
                kgStatusDiv.textContent = 'No setup selected. Button disabled.';
                kgStatusDiv.style.color = 'orange';
            }
        }
    });
    
    // Explore button
    exploreBtn.addEventListener('click', exploreKGSetup);
    
    // Reset button
    document.getElementById('kg-reset-btn').addEventListener('click', resetKGView);
    
    // Visual confirmation that event listeners are attached
    if (kgStatusDiv) {
        kgStatusDiv.textContent = 'Event listeners attached! Select a setup.';
    }
    console.log('✅ All KG event listeners attached successfully');
}

// Explore KG setup
async function exploreKGSetup() {
    console.log('🔍 Starting KG exploration for:', kgState.selectedSetup);
    
    if (!kgState.selectedSetup) {
        console.error('❌ No setup selected');
        alert('Please select a setup first');
        return;
    }
    
    const button = document.getElementById('kg-explore-btn');
    const originalText = button.textContent;
    
    try {
        button.disabled = true;
        button.textContent = '🔄 Exploring...';
        
        showKGLoading();
        
        // Perform traversal
        const analysisDepth = document.getElementById('kg-analysis-depth').value;
        console.log('📊 Request params:', {
            setup_id: kgState.selectedSetup,
            analysis_depth: analysisDepth,
            include_similar: true,
            include_reasoning: true,
            max_similar: 10
        });
        
        const response = await fetch(`${API_BASE}/api/kg/traverse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                setup_id: kgState.selectedSetup,
                analysis_depth: analysisDepth,
                include_similar: true,
                include_reasoning: true,
                max_similar: 10
            })
        });
        
        console.log('📡 Response status:', response.status);
        const result = await response.json();
        console.log('📊 Response data:', result);
        
        if (response.ok) {
            // Check if setup has any connections
            if (result.subgraph_nodes.length <= 1 && result.subgraph_edges.length === 0) {
                throw new Error(`Setup ${kgState.selectedSetup} has no feature connections. Try a setup with rich feature data like BLND_2025-03-21, WTB_2025-01-06, or HWDN_2024-08-16.`);
            }
            
            // Display results
            displayKGResults(result);
            
            // Load and display visualization
            await loadKGVisualization(kgState.selectedSetup);
            
        } else {
            throw new Error(result.detail || 'Traversal failed');
        }
        
    } catch (error) {
        console.error('❌ KG exploration error:', error);
        alert(`Exploration failed: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
        hideKGLoading();
    }
}

// Display KG results
function displayKGResults(result) {
    console.log('📊 Displaying KG results:', result);
    
    // Display similar setups
    displaySimilarSetups(result.similar_setups);
    
    // Display reasoning paths
    displayReasoningPaths(result.reasoning_paths);
    
    // Display insights and recommendations
    displayKGInsights(result.key_insights, result.recommendations);
}

// Display similar setups
function displaySimilarSetups(similarSetups) {
    const container = document.getElementById('kg-similar-setups');
    
    if (!similarSetups || similarSetups.length === 0) {
        container.innerHTML = '<p>No similar setups found.</p>';
        return;
    }
    
    let html = '<div class="kg-similar-setups">';
    
    similarSetups.forEach(setup => {
        const outcomeClass = setup.outcome ? 'kg-outcome-success' : 'kg-outcome-failure';
        const outcomeText = setup.outcome ? 'Successful' : 'Failed';
        
        html += `
            <div class="kg-similar-setup">
                <div class="kg-setup-header">
                    <div class="kg-setup-id">
                        <span class="kg-outcome-indicator ${outcomeClass}"></span>
                        ${setup.setup_id}
                    </div>
                    <div class="kg-similarity-score">${(setup.similarity_score * 100).toFixed(1)}%</div>
                </div>
                <div class="kg-setup-details">
                    <p><strong>Outcome:</strong> ${outcomeText}</p>
                    <p><strong>Explanation:</strong> ${setup.explanation}</p>
                </div>
                <div class="kg-common-features">
                    <strong>Common Features:</strong>
                    ${setup.common_features.map(feature => 
                        `<span class="kg-feature-tag">${feature}</span>`
                    ).join('')}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// Display reasoning paths
function displayReasoningPaths(reasoningPaths) {
    const container = document.getElementById('kg-reasoning-paths');
    
    if (!reasoningPaths || reasoningPaths.length === 0) {
        container.innerHTML = '<p>No reasoning paths found.</p>';
        return;
    }
    
    let html = '<div class="kg-reasoning-paths">';
    
    reasoningPaths.forEach(path => {
        html += `
            <div class="kg-reasoning-path">
                <div class="kg-path-header">
                    <div class="kg-path-type">${path.path_type}</div>
                    <div class="kg-path-confidence">${(path.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="kg-path-explanation">
                    <p>${path.explanation}</p>
                </div>
                <div class="kg-path-nodes">
                    ${path.nodes.map((node, index) => {
                        let html = `<span class="kg-path-node">${node}</span>`;
                        if (index < path.nodes.length - 1) {
                            html += '<span class="kg-path-arrow">→</span>';
                        }
                        return html;
                    }).join('')}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// Display KG insights
function displayKGInsights(insights, recommendations) {
    const container = document.getElementById('kg-insights');
    
    let html = '';
    
    if (insights && insights.length > 0) {
        html += '<div class="kg-insights">';
        html += '<h3>💡 Key Insights</h3>';
        insights.forEach(insight => {
            html += `
                <div class="kg-insight-item">
                    <div class="kg-insight-icon">💡</div>
                    <div>${insight}</div>
                </div>
            `;
        });
        html += '</div>';
    }
    
    if (recommendations && recommendations.length > 0) {
        html += '<div class="kg-recommendations">';
        html += '<h3>🎯 Recommendations</h3>';
        recommendations.forEach(recommendation => {
            html += `
                <div class="kg-recommendation-item">
                    <div class="kg-recommendation-icon">🎯</div>
                    <div>${recommendation}</div>
                </div>
            `;
        });
        html += '</div>';
    }
    
    if (!html) {
        html = '<p>No insights or recommendations available.</p>';
    }
    
    container.innerHTML = html;
}

// Load and display KG visualization
async function loadKGVisualization(setupId) {
    console.log('📊 Loading KG visualization for:', setupId);
    
    try {
        // Add timeout to prevent infinite loading
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Visualization loading timed out')), 10000);
        });
        
        // Get subgraph data
        const fetchPromise = fetch(`${API_BASE}/api/kg/subgraph`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                setup_id: setupId,
                max_hops: 2,
                include_similar: true,
                include_reasoning: false
            })
        });
        
        const response = await Promise.race([fetchPromise, timeoutPromise]);
        const subgraphData = await response.json();
        
        console.log('📊 Subgraph data loaded:', subgraphData.subgraph_stats);
        
        if (response.ok) {
            // Check if we have meaningful data to visualize
            if (subgraphData.nodes.length <= 1 && subgraphData.edges.length === 0) {
                throw new Error(`Setup ${setupId} has no feature connections to visualize. Try a setup with rich feature data like BLND_2025-03-21, WTB_2025-01-06, or HWDN_2024-08-16.`);
            }
            
            renderKGVisualization(subgraphData);
        } else {
            throw new Error(subgraphData.detail || 'Subgraph loading failed');
        }
        
    } catch (error) {
        console.error('❌ Error loading KG visualization:', error);
        
        // Show error message
        const graphContainer = document.getElementById('kg-graph-cytoscape');
        if (graphContainer) {
            graphContainer.innerHTML = `
                <div style="display: flex; justify-content: center; align-items: center; height: 100%; color: var(--error-color); text-align: center; padding: 2rem;">
                    <div>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">⚠️ Visualization Failed</p>
                        <p style="font-size: 0.9rem; line-height: 1.4;">${error.message}</p>
                    </div>
                </div>
            `;
        }
    }
}

// Global Cytoscape.js instance
let kgCytoscapeInstance = null;

// Render KG visualization using Cytoscape.js
function renderKGVisualization(data) {
    const container = document.getElementById('kg-graph-cytoscape');
    
    if (!container) {
        console.error('❌ Cytoscape container not found');
        return;
    }
    
    // Clear existing visualization
    if (kgCytoscapeInstance) {
        kgCytoscapeInstance.destroy();
        kgCytoscapeInstance = null;
    }
    
    // Prepare nodes and edges for Cytoscape.js
    const cytoscapeElements = prepareCytoscapeData(data);
    
    // Initialize Cytoscape.js
    kgCytoscapeInstance = cytoscape({
        container: container,
        elements: cytoscapeElements,
        
        style: getCytoscapeStyles(),
        
        layout: {
            name: getAvailableLayout('cose-bilkent'),
            quality: 'default',
            nodeRepulsion: 4500,
            idealEdgeLength: 50,
            edgeElasticity: 0.45,
            nestingFactor: 0.1,
            gravity: 0.25,
            numIter: 2500,
            tile: false,
            animate: 'end',
            animationDuration: 1000,
            fit: true,
            padding: 30
        },
        
        // Enable panning and zooming
        zoomingEnabled: true,
        userZoomingEnabled: true,
        panningEnabled: true,
        userPanningEnabled: true,
        
        // Set min/max zoom
        minZoom: 0.1,
        maxZoom: 3,
        
        // Disable node selection box
        selectionType: 'single',
        
        // Responsive settings
        autoungrabify: false,
        autounselectify: false
    });
    
    // Add event listeners for interactivity
            setupCytoscapeEventListeners(kgCytoscapeInstance);
    
    // Initialize layout controls
    setupLayoutControls(kgCytoscapeInstance);
    
    console.log('✅ Cytoscape.js KG visualization rendered with', cytoscapeElements.length, 'elements');
}

// Prepare data for Cytoscape.js format
function prepareCytoscapeData(data) {
    const elements = [];
    const nodes = data.node_data || {};
    const edges = data.edges || [];
    
    // Add nodes
    Object.keys(nodes).forEach(nodeId => {
        const node = nodes[nodeId];
        const nodeType = node.node_type || 'unknown';
        
        elements.push({
            data: {
                id: nodeId,
                label: formatNodeLabel(nodeId, nodeType),
                type: nodeType,
                originalData: node
            },
            classes: nodeType
        });
    });
    
    // Add edges
    edges.forEach((edge, index) => {
        const edgeId = `edge-${index}`;
        elements.push({
            data: {
                id: edgeId,
                source: edge.source || edge[0],
                target: edge.target || edge[1],
                relationship: edge.relationship || 'connected',
                weight: edge.weight || 1
            },
            classes: edge.relationship || 'default'
        });
    });
    
    return elements;
}

// Format node labels for better readability
function formatNodeLabel(nodeId, nodeType) {
    if (nodeType === 'setup') {
        // For setup nodes, show ticker and date
        const parts = nodeId.split('_');
        if (parts.length >= 2) {
            return `${parts[0]}\n${parts[1]}`;
        }
    }
    
    // For other nodes, truncate if too long
    if (nodeId.length > 15) {
        return nodeId.substring(0, 12) + '...';
    }
    
    return nodeId;
}

// Get Cytoscape.js styles
function getCytoscapeStyles() {
    return [
        // Default node style
        {
            selector: 'node',
            style: {
                'background-color': '#3498db',
                'border-color': '#2980b9',
                'border-width': 2,
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'font-weight': 'bold',
                'color': '#fff',
                'text-outline-width': 2,
                'text-outline-color': '#333',
                'width': 40,
                'height': 40,
                'overlay-opacity': 0.2,
                'z-index': 10
            }
        },
        
        // Setup nodes (most important)
        {
            selector: 'node.setup',
            style: {
                'background-color': '#e74c3c',
                'border-color': '#c0392b',
                'width': 60,
                'height': 60,
                'font-size': '12px',
                'z-index': 15
            }
        },
        
        // Feature nodes
        {
            selector: 'node.feature',
            style: {
                'background-color': '#2ecc71',
                'border-color': '#27ae60',
                'width': 45,
                'height': 45,
                'font-size': '9px'
            }
        },
        
        // Reasoning nodes
        {
            selector: 'node.reasoning',
            style: {
                'background-color': '#f39c12',
                'border-color': '#d68910',
                'width': 35,
                'height': 35,
                'font-size': '8px'
            }
        },
        
        // Outcome nodes
        {
            selector: 'node.outcome',
            style: {
                'background-color': '#9b59b6',
                'border-color': '#8e44ad',
                'width': 50,
                'height': 50,
                'font-size': '10px'
            }
        },
        
        // Default edge style
        {
            selector: 'edge',
            style: {
                'width': 2,
                'line-color': '#bdc3c7',
                'target-arrow-color': '#bdc3c7',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 1,
                'opacity': 0.7
            }
        },
        
        // Correlation edges
        {
            selector: 'edge.correlation',
            style: {
                'line-color': '#3498db',
                'target-arrow-color': '#3498db',
                'width': 3
            }
        },
        
        // Reasoning edges
        {
            selector: 'edge.reasoning',
            style: {
                'line-color': '#f39c12',
                'target-arrow-color': '#f39c12',
                'width': 2,
                'line-style': 'dashed'
            }
        },
        
        // Selected node style
        {
            selector: 'node:selected',
            style: {
                'border-color': '#fff',
                'border-width': 4,
                'background-color': '#e67e22',
                'z-index': 20
            }
        },
        
        // Selected edge style
        {
            selector: 'edge:selected',
            style: {
                'line-color': '#e67e22',
                'target-arrow-color': '#e67e22',
                'width': 4,
                'opacity': 1
            }
        },
        
        // Hover effects
        {
            selector: 'node:active',
            style: {
                'overlay-opacity': 0.3,
                'overlay-color': '#fff'
            }
        }
    ];
}

// Setup interactive event listeners
function setupCytoscapeEventListeners(cy) {
    // Node click event
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        const nodeData = node.data();
        
        // Show node details in info panel
        showNodeDetails(nodeData);
        
        // Highlight connected nodes
        highlightConnectedNodes(cy, node);
        
        console.log('Node clicked:', nodeData.id);
    });
    
    // Edge click event
    cy.on('tap', 'edge', function(evt) {
        const edge = evt.target;
        const edgeData = edge.data();
        
        console.log('Edge clicked:', edgeData.id, 'connects', edgeData.source, 'to', edgeData.target);
    });
    
    // Background click (deselect)
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            cy.elements().removeClass('highlighted');
            hideNodeDetails();
        }
    });
    
    // Mouse over node
    cy.on('mouseover', 'node', function(evt) {
        const node = evt.target;
        showNodeTooltip(evt, node.data());
    });
    
    // Mouse out node
    cy.on('mouseout', 'node', function(evt) {
        hideNodeTooltip();
    });
}

// Show node details in info panel
function showNodeDetails(nodeData) {
    const infoPanel = document.getElementById('kg-info-panel');
    const nodeDetails = document.getElementById('kg-node-details');
    
    if (infoPanel && nodeDetails) {
        const originalData = nodeData.originalData || {};
        
        let detailsHTML = `
            <div><strong>ID:</strong> ${nodeData.id}</div>
            <div><strong>Type:</strong> ${nodeData.type}</div>
        `;
        
        // Add type-specific details
        if (nodeData.type === 'setup') {
            detailsHTML += `
                <div><strong>Ticker:</strong> ${originalData.stock_ticker || 'N/A'}</div>
                <div><strong>Date:</strong> ${originalData.setup_date || 'N/A'}</div>
                <div><strong>Outperformed:</strong> ${originalData.outperformed ? 'Yes' : 'No'}</div>
            `;
        } else if (nodeData.type === 'feature') {
            detailsHTML += `
                <div><strong>Feature:</strong> ${originalData.feature_name || nodeData.id}</div>
                <div><strong>Value:</strong> ${originalData.value || 'N/A'}</div>
            `;
        }
        
        nodeDetails.innerHTML = detailsHTML;
        infoPanel.classList.add('visible');
    }
}

// Hide node details
function hideNodeDetails() {
    const infoPanel = document.getElementById('kg-info-panel');
    if (infoPanel) {
        infoPanel.classList.remove('visible');
    }
}

// Highlight connected nodes
function highlightConnectedNodes(cy, node) {
    // Remove existing highlights
    cy.elements().removeClass('highlighted');
    
    // Get connected nodes and edges
    const connectedEdges = node.connectedEdges();
    const connectedNodes = connectedEdges.connectedNodes();
    
    // Add highlight class
    node.addClass('highlighted');
    connectedNodes.addClass('highlighted');
    connectedEdges.addClass('highlighted');
}

// Show node tooltip
function showNodeTooltip(evt, nodeData) {
    // Create tooltip element if it doesn't exist
    let tooltip = document.getElementById('kg-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'kg-tooltip';
        tooltip.className = 'kg-tooltip';
        document.body.appendChild(tooltip);
    }
    
    // Set tooltip content
    tooltip.innerHTML = `<strong>${nodeData.id}</strong><br>Type: ${nodeData.type}`;
    
    // Position tooltip
    const containerRect = document.getElementById('kg-graph-cytoscape').getBoundingClientRect();
    tooltip.style.left = (evt.renderedPosition.x + containerRect.left + 10) + 'px';
    tooltip.style.top = (evt.renderedPosition.y + containerRect.top - 30) + 'px';
    tooltip.style.display = 'block';
}

// Hide node tooltip
function hideNodeTooltip() {
    const tooltip = document.getElementById('kg-tooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// Setup layout controls
function setupLayoutControls(cy) {
    const layoutSelect = document.getElementById('kg-layout-select');
    const layoutApply = document.getElementById('kg-layout-apply');
    
    if (layoutApply) {
        layoutApply.addEventListener('click', () => {
            const selectedLayout = layoutSelect.value;
            applyLayout(cy, selectedLayout);
        });
    }
}

// Apply different layouts
function applyLayout(cy, layoutName) {
    let layoutOptions = {
        name: layoutName,
        animate: true,
        animationDuration: 1000,
        fit: true,
        padding: 30
    };
    
    // Layout-specific options
    switch(layoutName) {
        case 'cose-bilkent':
            layoutOptions = {
                ...layoutOptions,
                quality: 'default',
                nodeRepulsion: 4500,
                idealEdgeLength: 50,
                edgeElasticity: 0.45,
                nestingFactor: 0.1,
                gravity: 0.25,
                numIter: 2500,
                tile: false
            };
            break;
        
        case 'dagre':
            layoutOptions = {
                ...layoutOptions,
                rankDir: 'TB',
                align: 'UL',
                ranker: 'network-simplex'
            };
            break;
        
        case 'cola':
            layoutOptions = {
                ...layoutOptions,
                maxSimulationTime: 4000,
                ungrabifyWhileSimulating: true,
                centerGraph: true
            };
            break;
        
        case 'concentric':
            layoutOptions = {
                ...layoutOptions,
                concentric: function(node) {
                    return node.data('type') === 'setup' ? 100 : 1;
                },
                levelWidth: function(nodes) {
                    return 2;
                }
            };
            break;
        
        case 'circle':
            layoutOptions = {
                ...layoutOptions,
                radius: 200,
                sort: function(a, b) {
                    return a.data('type').localeCompare(b.data('type'));
                }
            };
            break;
        
        case 'grid':
            layoutOptions = {
                ...layoutOptions,
                rows: undefined,
                cols: undefined,
                position: function(node) {}
            };
            break;
    }
    
    const layout = cy.layout(layoutOptions);
    layout.run();
    
    console.log('✅ Applied layout:', layoutName);
}

// Reset KG view
function resetKGView() {
    // Reset selections
    document.getElementById('kg-setup-select').value = '';
    document.getElementById('kg-analysis-depth').value = 'medium';
    document.getElementById('kg-explore-btn').disabled = true;
    
    // Clear results
    document.getElementById('kg-similar-setups').innerHTML = '<p>Select a setup to view similar setups analysis...</p>';
    document.getElementById('kg-reasoning-paths').innerHTML = '<p>Select a setup to view reasoning paths...</p>';
    document.getElementById('kg-insights').innerHTML = '<p>Select a setup to view insights and recommendations...</p>';
    
    // Clear visualization
    if (kgCytoscapeInstance) {
        kgCytoscapeInstance.destroy();
        kgCytoscapeInstance = null;
    }
    
    // Hide info panel
    hideNodeDetails();
    
    // Reset state
    kgState.selectedSetup = null;
    kgState.graphData = null;
    kgState.currentVisualization = null;
    
    console.log('🔄 KG view reset');
}

// Show KG loading
function showKGLoading() {
    const loadingElement = document.getElementById('kg-loading');
    if (loadingElement) {
        loadingElement.classList.add('visible');
    }
}

// Hide KG loading
function hideKGLoading() {
    const loadingElement = document.getElementById('kg-loading');
    if (loadingElement) {
        loadingElement.classList.remove('visible');
    }
}

// Expose global functions for navigation
window.showPage = showPage;