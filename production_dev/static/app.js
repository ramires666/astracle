/**
 * Bitcoin Astro Predictor - Frontend Entry Point
 *
 * This file is intentionally small.
 * Big files become impossible to reason about and the project rules require
 * each module to stay under 500 lines.
 *
 * Main responsibilities:
 * - bind DOM elements
 * - wire event listeners
 * - call API loaders
 * - (re)build the chart
 * - render the forecast table
 */

import { bindElements, elements } from './js/elements.js';
import { state } from './js/state.js';
import { checkModelHealth, fetchCachedPredictions, getForecastPredictions } from './js/api.js';
import { destroyChart, initializeChart } from './js/chart.js';
import {
    showLoading,
    updateBacktestSliderLabel,
    updateForecastSliderLabel,
    updateForecastTable,
} from './js/ui.js';
import { exportForecastToCSV } from './js/csv.js';

const AUTO_META_REFRESH_MS = 15 * 60 * 1000;
let lastMetaRefreshAt = 0;

function rebuildChart() {
    destroyChart();
    initializeChart();
}

async function refreshLiveData({ rebuild = false } = {}) {
    await checkModelHealth();
    await fetchCachedPredictions();
    lastMetaRefreshAt = Date.now();
    if (rebuild) rebuildChart();
}

function setupEventListeners() {
    // History slider (how many past days to SHOW)
    elements.backtestSlider?.addEventListener('input', (e) => {
        const days = Number(e.target.value);
        state.backtestDays = days;
        updateBacktestSliderLabel(days);
        rebuildChart();
    });

    // Forecast slider (how many future days to SHOW)
    elements.daysSlider?.addEventListener('input', (e) => {
        const days = Number(e.target.value);
        state.forecastDays = days;
        updateForecastSliderLabel(days);
        rebuildChart();
    });

    // Generate Forecast button
    elements.predictBtn?.addEventListener('click', async () => {
        if (state.isLoading) return;

        const days = Number(elements.daysSlider?.value || state.forecastDays);

        state.isLoading = true;
        showLoading(true);
        if (elements.predictBtn) elements.predictBtn.disabled = true;

        try {
            state.forecastDays = days;
            updateForecastSliderLabel(days);

            // Table uses the same source as the chart forecast line.
            const preds = await getForecastPredictions(days);
            state.forecastTable = preds;

            rebuildChart();
            updateForecastTable(preds);

            // Ensure table is visible (in case CSS hides it in the future)
            if (elements.tableSection) elements.tableSection.style.display = 'block';
        } catch (error) {
            console.error('Forecast generation failed:', error);
            alert(`Forecast error: ${error.message}`);
        } finally {
            state.isLoading = false;
            showLoading(false);
            if (elements.predictBtn) elements.predictBtn.disabled = false;
        }
    });

    // CSV export button
    elements.exportBtn?.addEventListener('click', () => exportForecastToCSV(state.forecastTable));
}

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Bitcoin Astro Predictor - Initializing...');

    bindElements();

    // Read slider defaults from the DOM (so HTML is the single source of truth).
    state.backtestDays = Number(elements.backtestSlider?.value || state.backtestDays);
    state.forecastDays = Number(elements.daysSlider?.value || state.forecastDays);
    updateBacktestSliderLabel(state.backtestDays);
    updateForecastSliderLabel(state.forecastDays);

    setupEventListeners();

    // Load live model/cache snapshots (sequential so badges get model info first).
    await refreshLiveData({ rebuild: false });

    // Build initial chart (uses cached data if available).
    rebuildChart();

    // Keep header badges and chart metadata aligned with daily retrain/cache refresh.
    setInterval(async () => {
        try {
            await refreshLiveData({ rebuild: true });
        } catch (error) {
            console.warn('Periodic live refresh failed:', error);
        }
    }, AUTO_META_REFRESH_MS);

    // Also refresh when user returns to the tab after being away for a while.
    document.addEventListener('visibilitychange', async () => {
        if (document.visibilityState !== 'visible') return;
        const stale = Date.now() - lastMetaRefreshAt > AUTO_META_REFRESH_MS;
        if (!stale) return;
        try {
            await refreshLiveData({ rebuild: true });
        } catch (error) {
            console.warn('Visibility refresh failed:', error);
        }
    });

    console.log('✅ Initialization complete');
});

// Global error trap to make debugging easier for non-dev users.
window.onerror = function (msg, url, lineNo, columnNo, error) {
    console.error('Application error:', { msg, url, lineNo, columnNo, error });
    return false;
};
