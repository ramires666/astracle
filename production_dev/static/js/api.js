/**
 * API Calls
 *
 * Only this file should "know" endpoint URLs.
 */

import { CONFIG } from './config.js';
import { state } from './state.js';
import { updateModelStatus, updateModelInfo, updateBacktestStatsBadges, updateBacktestSliderLimits } from './ui.js';

export async function checkModelHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/health`);
        const data = await response.json();

        updateModelStatus(data);

        if (data.status === 'healthy') {
            const cfgResp = await fetch(`${CONFIG.API_BASE}/api/config`);
            state.modelInfo = await cfgResp.json();
            updateModelInfo(state.modelInfo);
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateModelStatus({ status: 'unhealthy' });
    }
}

export async function fetchCachedPredictions() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/predictions/full`);
        if (!response.ok) {
            console.warn('Cached predictions not available:', response.status);
            return;
        }

        const data = await response.json();

        if (Array.isArray(data.backtest) && data.backtest.length > 0) {
            state.cachedBacktest = data.backtest;
            state.backtestByDate = new Map(data.backtest.map((r) => [r.date, r]));
            updateBacktestSliderLimits(state.cachedBacktest.length);
        }

        if (Array.isArray(data.forecast) && data.forecast.length > 0) {
            state.cachedForecast = data.forecast;
        }

        if (data.accuracy) {
            state.accuracyStats = data.accuracy;
            updateBacktestStatsBadges(state.accuracyStats);
        }

        // Fetch fresh actual prices from market data source (can be newer than backtest cache).
        // This keeps the white "Actual Price" line up to date for the latest days.
        const histDays = Math.min(1500, Math.max(120, Number(state.cachedBacktest.length || 0) + 14));
        const histResp = await fetch(`${CONFIG.API_BASE}/api/historical?days=${histDays}`);
        if (histResp.ok) {
            const histData = await histResp.json();
            if (Array.isArray(histData?.prices)) {
                state.cachedActualPrices = histData.prices;
            }
        } else {
            console.warn('Historical prices not available:', histResp.status);
        }
    } catch (error) {
        console.warn('Could not fetch cached predictions:', error);
    }
}

export async function getForecastPredictions(days) {
    // Prefer cached forecast because it is instant and already has simulated prices.
    if (Array.isArray(state.cachedForecast) && state.cachedForecast.length > 0) {
        return state.cachedForecast.slice(0, days);
    }

    // Fallback: ask the API to generate forecast on demand.
    const response = await fetch(`${CONFIG.API_BASE}/api/predict?days=${days}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    return data.predictions || [];
}
