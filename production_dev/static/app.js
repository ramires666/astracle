/**
 * Bitcoin Astro Predictor - Frontend Application
 * 
 * Handles API communication, chart rendering, and UI interactions.
 * Fetches historical BTC prices from CoinGecko for chart context.
 * 
 * @version 1.0.0
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
    API_BASE: '',  // Same origin
    COINGECKO_API: 'https://api.coingecko.com/api/v3',
    HISTORICAL_DAYS: 30,  // Days of historical data to show
    DEFAULT_FORECAST_DAYS: 90,
    CHART_ANIMATION_DURATION: 800,
};

// =============================================================================
// STATE
// =============================================================================

let state = {
    chart: null,
    predictions: [],
    historicalPrices: [],
    isLoading: false,
    modelInfo: null,
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const elements = {
    // Header
    modelStatus: document.getElementById('model-status'),
    headerStats: document.getElementById('header-stats'),

    // Model Info
    accuracyBadge: document.getElementById('accuracy-badge'),
    natalDate: document.getElementById('natal-date'),
    coordMode: document.getElementById('coord-mode'),
    orbMult: document.getElementById('orb-mult'),
    nEstimators: document.getElementById('n-estimators'),

    // Controls
    daysSlider: document.getElementById('days-slider'),
    daysValue: document.getElementById('days-value'),
    predictBtn: document.getElementById('predict-btn'),

    // Chart
    chartCanvas: document.getElementById('prediction-chart'),
    chartLoading: document.getElementById('chart-loading'),

    // Summary
    summarySection: document.getElementById('summary-section'),
    upCount: document.getElementById('up-count'),
    downCount: document.getElementById('down-count'),
    avgConfidence: document.getElementById('avg-confidence'),
    priceChange: document.getElementById('price-change'),

    // Table
    tableSection: document.getElementById('table-section'),
    tableBody: document.getElementById('table-body'),
    exportBtn: document.getElementById('export-btn'),
};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Bitcoin Astro Predictor - Initializing...');

    // Set up event listeners
    setupEventListeners();

    // Load historical data and model info in parallel
    await Promise.all([
        fetchHistoricalPrices(),
        checkModelHealth(),
    ]);

    // Initial chart with just historical data
    initializeChart();

    console.log('✅ Initialization complete');
});

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // Days slider
    elements.daysSlider.addEventListener('input', (e) => {
        elements.daysValue.textContent = `${e.target.value} days`;
    });

    // Predict button
    elements.predictBtn.addEventListener('click', () => {
        const days = parseInt(elements.daysSlider.value);
        generateForecast(days);
    });

    // Export button
    elements.exportBtn.addEventListener('click', exportToCSV);
}

// =============================================================================
// API CALLS
// =============================================================================

/**
 * Check model health and update UI with model info.
 */
async function checkModelHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/health`);
        const data = await response.json();

        updateModelStatus(data);

        if (data.status === 'healthy') {
            // Fetch full config
            const configResponse = await fetch(`${CONFIG.API_BASE}/api/config`);
            state.modelInfo = await configResponse.json();
            updateModelInfo(state.modelInfo);
        }
    } catch (error) {
        console.error('Health check failed:', error);
        elements.modelStatus.textContent = 'Offline';
        elements.modelStatus.style.color = 'var(--accent-red)';
    }
}

/**
 * Fetch historical BTC prices from CoinGecko API.
 * This provides context for the prediction chart.
 */
async function fetchHistoricalPrices() {
    try {
        const url = `${CONFIG.COINGECKO_API}/coins/bitcoin/market_chart`;
        const params = new URLSearchParams({
            vs_currency: 'usd',
            days: CONFIG.HISTORICAL_DAYS,
            interval: 'daily',
        });

        const response = await fetch(`${url}?${params}`);

        if (!response.ok) {
            throw new Error(`CoinGecko API error: ${response.status}`);
        }

        const data = await response.json();

        // Transform to our format
        state.historicalPrices = data.prices.map(([timestamp, price]) => ({
            date: new Date(timestamp).toISOString().split('T')[0],
            price: price,
        }));

        console.log(`📈 Loaded ${state.historicalPrices.length} days of historical data`);

    } catch (error) {
        console.error('Failed to fetch historical prices:', error);
        // Use fallback data if API fails
        state.historicalPrices = generateFallbackHistoricalData();
    }
}

/**
 * Generate forecast predictions from our API.
 */
async function generateForecast(days) {
    if (state.isLoading) return;

    state.isLoading = true;
    showLoading(true);
    elements.predictBtn.disabled = true;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/predict?days=${days}`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        state.predictions = data.predictions;

        // Update all UI components
        updateChart();
        updateSummary(data.summary);
        updateTable(data.predictions);

        // Show summary and table sections
        elements.summarySection.style.display = 'grid';
        elements.tableSection.style.display = 'block';

        console.log(`🔮 Generated ${data.predictions.length} day forecast`);

    } catch (error) {
        console.error('Forecast generation failed:', error);
        alert(`Forecast error: ${error.message}`);
    } finally {
        state.isLoading = false;
        showLoading(false);
        elements.predictBtn.disabled = false;
    }
}

// =============================================================================
// UI UPDATES
// =============================================================================

function updateModelStatus(healthData) {
    const statusBadge = elements.headerStats.querySelector('.stat-badge');
    statusBadge.classList.remove('loading');

    if (healthData.status === 'healthy') {
        elements.modelStatus.textContent = 'Online';
        elements.modelStatus.style.color = 'var(--accent-green)';
    } else {
        elements.modelStatus.textContent = 'Error';
        elements.modelStatus.style.color = 'var(--accent-red)';
    }
}

function updateModelInfo(info) {
    if (!info) return;

    const config = info.config || {};

    elements.accuracyBadge.textContent = `R_MIN: ${(config.r_min || 0).toFixed(3)}`;
    elements.natalDate.textContent = config.birth_date || '--';
    elements.coordMode.textContent = config.coord_mode || '--';
    elements.orbMult.textContent = config.orb_mult || '--';
    elements.nEstimators.textContent = config.n_estimators || '--';
}

function updateSummary(summary) {
    elements.upCount.textContent = summary.up_predictions;
    elements.downCount.textContent = summary.down_predictions;
    elements.avgConfidence.textContent = `${(summary.average_confidence * 100).toFixed(1)}%`;

    const priceChangePercent = ((summary.end_price - summary.start_price) / summary.start_price * 100);
    elements.priceChange.textContent = `${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(1)}%`;
    elements.priceChange.style.color = priceChangePercent >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
}

function updateTable(predictions) {
    elements.tableBody.innerHTML = predictions.map(pred => `
        <tr>
            <td>${pred.date}</td>
            <td class="${pred.direction === 'UP' ? 'direction-up' : 'direction-down'}">
                ${pred.direction === 'UP' ? '▲' : '▼'} ${pred.direction}
            </td>
            <td>${(pred.confidence * 100).toFixed(1)}%</td>
            <td>$${pred.simulated_price.toLocaleString()}</td>
        </tr>
    `).join('');
}

function showLoading(show) {
    if (show) {
        elements.chartLoading.classList.remove('hidden');
    } else {
        elements.chartLoading.classList.add('hidden');
    }
}

// =============================================================================
// CHART
// =============================================================================

function initializeChart() {
    const ctx = elements.chartCanvas.getContext('2d');

    // Prepare historical data for chart
    const historicalData = state.historicalPrices.map(hp => ({
        x: hp.date,
        y: hp.price,
    }));

    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Historical Price',
                    data: historicalData,
                    borderColor: 'rgba(255, 255, 255, 0.6)',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    borderWidth: 2,
                },
                {
                    label: 'Predicted Price',
                    data: [],  // Will be filled on forecast
                    borderColor: '#f7b731',
                    backgroundColor: 'rgba(247, 183, 49, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 8,
                    borderWidth: 2.5,
                    segment: {
                        borderColor: ctx => {
                            // Color based on prediction direction
                            const pred = state.predictions[ctx.p0DataIndex];
                            if (pred) {
                                return pred.direction === 'UP' ? '#00d4aa' : '#ff6b6b';
                            }
                            return '#f7b731';
                        },
                    },
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: CONFIG.CHART_ANIMATION_DURATION,
            },
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        padding: 20,
                        font: {
                            family: "'Inter', sans-serif",
                        },
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 30, 0.95)',
                    titleColor: '#fff',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            const value = context.parsed.y;
                            const label = context.dataset.label;

                            if (context.datasetIndex === 1) {
                                // Prediction dataset
                                const pred = state.predictions[context.dataIndex];
                                if (pred) {
                                    return [
                                        `${label}: $${value.toLocaleString()}`,
                                        `Direction: ${pred.direction}`,
                                        `Confidence: ${(pred.confidence * 100).toFixed(1)}%`,
                                    ];
                                }
                            }

                            return `${label}: $${value.toLocaleString()}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d',
                        },
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        maxRotation: 45,
                        font: {
                            family: "'Inter', sans-serif",
                            size: 11,
                        },
                    },
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        callback: (value) => '$' + value.toLocaleString(),
                        font: {
                            family: "'Inter', sans-serif",
                            size: 11,
                        },
                    },
                },
            },
        },
    });

    // Hide loading
    elements.chartLoading.classList.add('hidden');
}

function updateChart() {
    if (!state.chart) return;

    // Prepare prediction data
    const predictionData = state.predictions.map(pred => ({
        x: pred.date,
        y: pred.simulated_price,
    }));

    // Update prediction dataset
    state.chart.data.datasets[1].data = predictionData;

    // Animate update
    state.chart.update('active');
}

// =============================================================================
// EXPORT
// =============================================================================

function exportToCSV() {
    if (state.predictions.length === 0) {
        alert('No predictions to export. Generate a forecast first.');
        return;
    }

    // Create CSV content
    const headers = ['Date', 'Direction', 'Confidence', 'Simulated Price'];
    const rows = state.predictions.map(pred => [
        pred.date,
        pred.direction,
        pred.confidence.toFixed(4),
        pred.simulated_price.toFixed(2),
    ]);

    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(',')),
    ].join('\n');

    // Download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `btc_astro_forecast_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
}

// =============================================================================
// UTILITIES
// =============================================================================

/**
 * Generate fallback historical data if CoinGecko API fails.
 */
function generateFallbackHistoricalData() {
    const data = [];
    const basePrice = 100000;
    let price = basePrice;

    for (let i = CONFIG.HISTORICAL_DAYS; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);

        // Random walk
        price *= 1 + (Math.random() - 0.5) * 0.04;

        data.push({
            date: date.toISOString().split('T')[0],
            price: price,
        });
    }

    return data;
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

window.onerror = function (msg, url, lineNo, columnNo, error) {
    console.error('Application error:', { msg, url, lineNo, columnNo, error });
    return false;
};
