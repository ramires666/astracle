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
    HISTORICAL_DAYS: 1200,  // Load enough history to cover full backtest (3 years+)
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
    cachedBacktest: [],      // Past predictions with accuracy
    cachedForecast: [],      // Future cached predictions
    accuracyStats: null,     // Backtest accuracy statistics
    isLoading: false,
    modelInfo: null,
    backtestDays: 180,       // How many history days to show (controlled by slider)
    forecastDays: 90,        // How many forecast days to show (controlled by slider)
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
    backtestSlider: document.getElementById('backtest-slider'),
    backtestValue: document.getElementById('backtest-value'),
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

    // Load data in parallel
    await Promise.all([
        fetchHistoricalPrices(),
        checkModelHealth(),
        fetchCachedPredictions(),
    ]);

    // Initial chart with historical + cached data
    initializeChart();

    // Show accuracy stats if available
    if (state.accuracyStats) {
        updateAccuracyDisplay(state.accuracyStats);
    }

    console.log('✅ Initialization complete');
});

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // History slider - updates how much past data to show
    elements.backtestSlider.addEventListener('input', (e) => {
        const days = parseInt(e.target.value);
        elements.backtestValue.textContent = `${days} days`;
        state.backtestDays = days;

        // Rebuild chart completely with new data range
        if (state.chart) {
            state.chart.destroy();
            initializeChart();
        }
    });

    // Forecast slider - updates how much future to show
    elements.daysSlider.addEventListener('input', (e) => {
        const days = parseInt(e.target.value);
        elements.daysValue.textContent = `${days} days`;
        state.forecastDays = days;

        // Rebuild chart with new forecast range
        if (state.chart) {
            state.chart.destroy();
            initializeChart();
        }
    });

    // Predict button - regenerates forecast prices
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
 * Fetch historical BTC prices from our project database.
 * Uses the same data loading functions as initial data collection.
 */
async function fetchHistoricalPrices() {
    try {
        const url = `${CONFIG.API_BASE}/api/historical?days=${CONFIG.HISTORICAL_DAYS}`;

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        // Transform to our format
        state.historicalPrices = data.prices.map(item => ({
            date: item.date,
            price: item.price,
        }));

        console.log(`📈 Loaded ${state.historicalPrices.length} days of historical data from database`);

    } catch (error) {
        console.error('Failed to fetch historical prices:', error);
        // Use fallback data if API fails
        state.historicalPrices = generateFallbackHistoricalData();
    }
}

/**
 * Fetch cached predictions (backtest + forecast) from API.
 * These are pre-calculated and stored in memory on the server.
 */
async function fetchCachedPredictions() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/predictions/full`);

        if (!response.ok) {
            console.warn('Cached predictions not available:', response.status);
            return;
        }

        const data = await response.json();

        // Store backtest (past predictions with accuracy)
        if (data.backtest && data.backtest.length > 0) {
            state.cachedBacktest = data.backtest;
            console.log(`📊 Loaded ${data.backtest.length} backtest predictions`);
        }

        // Store forecast (future predictions)
        if (data.forecast && data.forecast.length > 0) {
            state.cachedForecast = data.forecast;
            console.log(`🔮 Loaded ${data.forecast.length} forecast predictions`);
        }

        // Store accuracy stats
        if (data.accuracy) {
            state.accuracyStats = data.accuracy;
            console.log(`📈 Backtest accuracy: ${(data.accuracy.accuracy * 100).toFixed(1)}%`);
        }

    } catch (error) {
        console.warn('Could not fetch cached predictions:', error);
    }
}

/**
 * Update accuracy display in the UI.
 */
function updateAccuracyDisplay(stats) {
    // ---------------------------------------------------------------------
    // IMPORTANT UI RULE:
    // The big badge inside "Model Configuration" is reserved for notebook
    // metrics (R_MIN / MCC) because that is what the research notebook
    // produces and what the user expects to match 1:1.
    //
    // Backtest "accuracy" from the cache is a different number and can easily
    // confuse people if it overwrites the R_MIN badge.
    //
    // So we show cached backtest accuracy as a *separate* small badge
    // in the header (next to Model Status).
    // ---------------------------------------------------------------------
    if (!stats || stats.total === 0 || typeof stats.accuracy !== 'number') return;

    const accuracyPct = (stats.accuracy * 100).toFixed(1);

    // Create a second header badge lazily (only when we have data).
    let accBadge = document.getElementById('backtest-acc-badge');
    if (!accBadge) {
        accBadge = document.createElement('div');
        accBadge.id = 'backtest-acc-badge';
        accBadge.className = 'stat-badge';
        accBadge.innerHTML = `
            <span class="stat-label">Backtest Acc</span>
            <span class="stat-value" id="backtest-acc-value">--</span>
        `;
        elements.headerStats.appendChild(accBadge);
    }

    const valueEl = document.getElementById('backtest-acc-value');
    if (valueEl) {
        valueEl.textContent = `${accuracyPct}%`;

        // Simple color scale: green (good) -> yellow (meh) -> red (bad)
        if (stats.accuracy >= 0.55) {
            valueEl.style.color = 'var(--accent-green)';
        } else if (stats.accuracy >= 0.50) {
            valueEl.style.color = 'var(--accent-gold)';
        } else {
            valueEl.style.color = 'var(--accent-red)';
        }
    }
}

/**
 * Generate forecast using cached predictions.
 * Uses pre-calculated simulated prices from cache.
 */
async function generateForecast(days) {
    if (state.isLoading) return;

    state.isLoading = true;
    showLoading(true);
    elements.predictBtn.disabled = true;

    try {
        // Update forecast days
        state.forecastDays = days;
        elements.daysValue.textContent = `${days} days`;

        // Use cached forecast (already has simulated prices from generate_cache.py)
        if (state.cachedForecast && state.cachedForecast.length > 0) {
            // Just slice to requested number of days
            state.predictions = state.cachedForecast.slice(0, days);
            console.log(`🔮 Using cached forecast: ${state.predictions.length} days`);
        } else {
            // Fallback to API if no cache
            const response = await fetch(`${CONFIG.API_BASE}/api/predict?days=${days}`);
            if (!response.ok) throw new Error(`API error: ${response.status}`);
            const data = await response.json();
            state.predictions = data.predictions;
        }

        // Calculate summary
        const upDays = state.predictions.filter(p => p.direction === 'UP').length;
        const downDays = state.predictions.filter(p => p.direction === 'DOWN').length;
        const avgConf = state.predictions.reduce((sum, p) => sum + (p.confidence || 0.5), 0) / state.predictions.length;
        const startPrice = state.predictions[0]?.simulated_price || 0;
        const endPrice = state.predictions[state.predictions.length - 1]?.simulated_price || 0;
        const priceChange = startPrice > 0 ? ((endPrice - startPrice) / startPrice * 100) : 0;

        const summary = {
            up_days: upDays,
            down_days: downDays,
            avg_confidence: avgConf,
            price_change_percent: priceChange,
        };

        // Rebuild chart with new forecast range
        if (state.chart) {
            state.chart.destroy();
        }
        initializeChart();

        updateSummary(summary);
        updateTable(state.predictions);

        // Show summary and table sections
        elements.summarySection.style.display = 'grid';
        elements.tableSection.style.display = 'block';

        console.log(`🔮 Generated ${state.predictions.length} day forecast`);

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

    // Plugin to draw colored vertical regions for predictions
    const predictionBackgroundPlugin = {
        id: 'predictionBackground',
        beforeDraw: (chart) => {
            const { ctx, chartArea, scales } = chart;
            if (!chartArea || !scales.x) return;

            const today = new Date().toISOString().split('T')[0];

            // Combine backtest and forecast for background coloring
            // Use same slicing as chart data
            const backtestSlice = state.cachedBacktest.slice(-state.backtestDays);
            const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
            const allPredictions = [
                ...backtestSlice.map(p => ({ ...p, isPast: true })),
                ...forecastSlice.map(p => ({ ...p, isPast: false })),
            ];

            // Debug logging (remove after fixing)
            if (allPredictions.length === 0) {
                console.warn('⚠️ No predictions for background:', {
                    cachedBacktest: state.cachedBacktest.length,
                    cachedForecast: state.cachedForecast.length,
                    backtestDays: state.backtestDays,
                });
                return;
            }

            ctx.save();

            let drawnCount = 0;
            allPredictions.forEach((pred, index) => {
                // Parse date properly
                const dateStr = pred.date;
                const dateObj = new Date(dateStr);
                const x = scales.x.getPixelForValue(dateObj);

                // Calculate width for each day region
                const nextPred = allPredictions[index + 1];
                let nextX;
                if (nextPred) {
                    nextX = scales.x.getPixelForValue(new Date(nextPred.date));
                } else {
                    nextX = x + 10;  // Default width for last item
                }
                const width = Math.max(nextX - x, 2);

                // Skip if outside chart area
                if (x < chartArea.left - width || x > chartArea.right + width) return;

                // Solid background colors
                // Past = lighter shades, Future = more saturated
                let color;
                if (pred.direction === 'UP') {
                    color = pred.isPast
                        ? 'rgba(187, 247, 208, 0.45)'  // Light green (past/history)
                        : 'rgba(74, 222, 128, 0.55)';  // Saturated green (future)
                } else if (pred.direction === 'DOWN') {
                    color = pred.isPast
                        ? 'rgba(254, 202, 202, 0.45)'  // Light red/pink (past/history)
                        : 'rgba(248, 113, 113, 0.55)'; // Saturated red (future)
                } else {
                    color = 'rgba(150, 150, 150, 0.2)';
                }

                ctx.fillStyle = color;
                ctx.fillRect(
                    Math.max(x, chartArea.left),
                    chartArea.top,
                    Math.min(width, chartArea.right - x),
                    chartArea.bottom - chartArea.top
                );
                drawnCount++;
            });

            // Debug: log how many backgrounds were drawn
            if (drawnCount === 0 && allPredictions.length > 0) {
                console.warn('⚠️ 0 backgrounds drawn but', allPredictions.length, 'predictions exist');
                console.log('Sample prediction:', allPredictions[0]);
            }

            // Draw vertical line for today
            const todayX = scales.x.getPixelForValue(today);
            if (todayX >= chartArea.left && todayX <= chartArea.right) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(todayX, chartArea.top);
                ctx.lineTo(todayX, chartArea.bottom);
                ctx.stroke();
                ctx.setLineDash([]);

                // Label "Today"
                ctx.font = '11px Inter, sans-serif';
                ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.fillText('Today', todayX + 5, chartArea.top + 15);
            }

            ctx.restore();
        }
    };

    // Prepare historical data for chart - SLICED by backtestDays
    const historicalSlice = state.historicalPrices.slice(-state.backtestDays);
    const historicalData = historicalSlice.map(hp => ({
        x: hp.date,
        y: hp.price,
    }));

    // Prepare forecast data from cache - SLICED by forecastDays
    const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
    const cachedForecastData = forecastSlice.map(f => ({
        x: f.date,
        y: f.simulated_price || 0,
    }));

    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                // Historical actual prices
                {
                    label: 'Actual Price',
                    data: historicalData,
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointStyle: false,  // Completely disable points
                    pointHitRadius: 10,  // Still allow tooltip interaction
                    borderWidth: 2,
                    order: 1,
                },
                // Predicted future prices (from cache or API)
                {
                    label: 'Forecast',
                    data: cachedForecastData,
                    borderColor: 'rgba(50, 50, 50, 0.9)',  // Dark gray like matplotlib black
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointStyle: false,  // Completely disable points
                    pointHitRadius: 10,
                    borderWidth: 2,
                    order: 0,
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
                            const pointDate = new Date(context.parsed.x);
                            const today = new Date();
                            today.setHours(0, 0, 0, 0);

                            // For Forecast dataset - show prediction details
                            if (label === 'Forecast') {
                                const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
                                const pred = forecastSlice[context.dataIndex];
                                if (pred) {
                                    return [
                                        `Forecast: $${value.toLocaleString()}`,
                                        `Direction: ${pred.direction}`,
                                        `Confidence: ${(pred.confidence * 100).toFixed(1)}%`,
                                    ];
                                }
                                return `Forecast: $${value.toLocaleString()}`;
                            }

                            // For Actual Price - ONLY show for past dates, hide for future
                            if (label === 'Actual Price') {
                                if (pointDate > today) {
                                    // Future date - don't show actual price tooltip
                                    return null;
                                }
                                return `${label}: $${value.toLocaleString()}`;
                            }

                            return `${label}: $${value.toLocaleString()}`;
                        },
                        // Filter out null labels
                        filter: function (tooltipItem) {
                            return tooltipItem.formattedValue !== null;
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
        plugins: [predictionBackgroundPlugin],  // Register our custom plugin
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
