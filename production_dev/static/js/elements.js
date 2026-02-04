/**
 * DOM Elements Binder
 *
 * Why this module exists:
 * - We don't want every file to call `document.getElementById` repeatedly.
 * - We also want a single place to see "which IDs must exist in index.html".
 *
 * Important:
 * - `bindElements()` must be called AFTER DOMContentLoaded.
 */

export const elements = {};

export function bindElements() {
    // Header
    elements.modelStatus = document.getElementById('model-status');
    elements.headerStats = document.getElementById('header-stats');

    // Model card
    elements.accuracyBadge = document.getElementById('accuracy-badge');
    elements.natalDate = document.getElementById('natal-date');
    elements.coordMode = document.getElementById('coord-mode');
    elements.orbMult = document.getElementById('orb-mult');
    elements.nEstimators = document.getElementById('n-estimators');

    // Controls
    elements.backtestSlider = document.getElementById('backtest-slider');
    elements.backtestValue = document.getElementById('backtest-value');
    elements.daysSlider = document.getElementById('days-slider');
    elements.daysValue = document.getElementById('days-value');
    elements.predictBtn = document.getElementById('predict-btn');

    // Chart
    elements.chartCanvas = document.getElementById('prediction-chart');
    elements.chartLoading = document.getElementById('chart-loading');
    elements.splitCaption = document.getElementById('split-caption');

    // Table
    elements.tableSection = document.getElementById('table-section');
    elements.tableBody = document.getElementById('table-body');
    elements.exportBtn = document.getElementById('export-btn');
}
