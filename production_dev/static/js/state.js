/**
 * Global App State (in-memory)
 *
 * We keep state in one place so:
 * - the chart module can read it
 * - API calls can update it
 * - UI code can render it
 *
 * Note: This is NOT persisted anywhere. Reloading the page resets it.
 */

import { CONFIG } from './config.js';

export const state = {
    // Chart.js instance (created in chart.js)
    chart: null,

    // Cached data returned by /api/predictions/full
    cachedBacktest: [],
    cachedForecast: [],
    accuracyStats: null,

    // A fast lookup for tooltips: "YYYY-MM-DD" -> backtest row
    backtestByDate: new Map(),

    // Model info returned by /api/config (split model)
    modelInfo: null,

    // UI controls (slider values)
    backtestDays: CONFIG.DEFAULT_BACKTEST_DAYS,
    forecastDays: CONFIG.DEFAULT_FORECAST_DAYS,

    // Forecast table data (what user requested via the button)
    forecastTable: [],

    // UI flags
    isLoading: false,
};

