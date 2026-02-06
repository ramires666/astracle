/**
 * Chart Rendering (Chart.js)
 *
 * This module is responsible for:
 * - building the datasets (actual price + forecast line)
 * - drawing background colors for predicted UP/DOWN days
 * - drawing a small "split band" on top of the chart (train/val/test)
 *
 * Important UX rule:
 * - We must visually label which part of history is TRAIN / VAL / TEST
 *   so users don't accidentally think the in-sample area is "proof".
 */

import { CONFIG } from './config.js';
import { elements } from './elements.js';
import { state } from './state.js';

function toTimeValue(dateStr) {
    // Convert "YYYY-MM-DD" into a millisecond timestamp that matches the
    // *local calendar day* for the user.
    //
    // Why we do NOT use `Date.parse(dateStr)` or `new Date(dateStr)`:
    // - Some JS engines treat date-only strings as UTC, others as local time.
    // - Chart.js (with the date-fns adapter) formats ticks in LOCAL time.
    // - If we accidentally mix UTC parsing + local formatting, the chart can
    //   shift by 1 day for US users (exactly the kind of bug you reported).
    //
    // The most reliable approach is: parse the pieces ourselves and construct
    // a local Date at midnight.
    const [y, m, d] = String(dateStr).split('-').map((x) => Number(x));
    return new Date(y, m - 1, d).getTime();
}

function toIsoDateLocal(dateObj) {
    // Format a Date as "YYYY-MM-DD" in LOCAL time.
    //
    // We use local time because Chart.js + date-fns displays ticks in local
    // time by default. This keeps labels and tooltip dates consistent.
    const y = dateObj.getFullYear();
    const m = String(dateObj.getMonth() + 1).padStart(2, '0');
    const d = String(dateObj.getDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
}

function addDaysIso(dateStr, days) {
    // Add N days to a "YYYY-MM-DD" string (local calendar logic).
    const [y, m, d] = String(dateStr).split('-').map((x) => Number(x));
    const dt = new Date(y, m - 1, d);
    dt.setDate(dt.getDate() + days);
    return toIsoDateLocal(dt);
}

function diffDays(tsA, tsB) {
    // Integer day gap between two unix-ms timestamps.
    return Math.round((Number(tsB) - Number(tsA)) / 86400000);
}

function clipHistoricalToBacktestRange(historicalRows, backtestSlice) {
    // Keep actual-price line within the same date window as backtest labels.
    // This avoids "left side without split bands" when sources have different ranges.
    if (!Array.isArray(historicalRows) || historicalRows.length === 0) return [];
    if (!Array.isArray(backtestSlice) || backtestSlice.length === 0) return historicalRows;

    const startTs = toTimeValue(backtestSlice[0].date);
    const endTs = toTimeValue(backtestSlice[backtestSlice.length - 1].date);
    if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) return historicalRows;

    const clipped = historicalRows.filter((row) => {
        const ts = toTimeValue(row?.date);
        return Number.isFinite(ts) && ts >= startTs && ts <= endTs;
    });

    // Fallback to original data if clipping produced nothing.
    return clipped.length > 0 ? clipped : historicalRows;
}

function buildActualLinePoints(rows, maxGapDays = 2) {
    // Build Chart.js points and insert null separators across large date gaps.
    // This prevents fake diagonal lines over missing-history intervals.
    const clean = (Array.isArray(rows) ? rows : [])
        .map((row) => ({ x: toTimeValue(row?.date), y: Number(row?.price) }))
        .filter((pt) => Number.isFinite(pt.x) && Number.isFinite(pt.y))
        .sort((a, b) => a.x - b.x);

    if (clean.length === 0) return [];

    const out = [];
    let prevTs = null;
    for (const pt of clean) {
        if (prevTs != null) {
            const gap = diffDays(prevTs, pt.x);
            if (Number.isFinite(gap) && gap > maxGapDays) {
                // Small x-offset keeps point order stable while breaking the line.
                out.push({ x: prevTs + 1, y: null });
            }
        }
        out.push(pt);
        prevTs = pt.x;
    }

    return out;
}

function splitColor(split) {
    // We keep split colors subtle because the main background already encodes UP/DOWN.
    // Train/Val/Test are about honesty, not about direction.
    switch (split) {
        case 'train':
            return 'rgba(77, 171, 247, 0.35)'; // blue
        case 'val':
            return 'rgba(247, 183, 49, 0.35)'; // gold
        case 'test':
            return 'rgba(151, 117, 250, 0.35)'; // purple
        default:
            return null;
    }
}

function buildSplitSegments(backtestSlice) {
    // Convert a list of daily rows into contiguous segments like:
    // [{ split: "train", startDate: "...", endDate: "..." }, ...]
    if (!backtestSlice || backtestSlice.length === 0) return [];
    if (!backtestSlice[0].split) return [];

    const segments = [];
    let currentSplit = backtestSlice[0].split;
    let segStart = backtestSlice[0].date;

    for (let i = 1; i < backtestSlice.length; i++) {
        const s = backtestSlice[i].split;
        if (s !== currentSplit) {
            segments.push({
                split: currentSplit,
                startDate: segStart,
                endDate: backtestSlice[i - 1].date,
            });
            currentSplit = s;
            segStart = backtestSlice[i].date;
        }
    }

    segments.push({
        split: currentSplit,
        startDate: segStart,
        endDate: backtestSlice[backtestSlice.length - 1].date,
    });

    return segments;
}

function buildPredictionBackgroundPlugin() {
    return {
        id: 'predictionBackground',
        beforeDraw: (chart) => {
            const { ctx, chartArea, scales } = chart;
            if (!chartArea || !scales?.x) return;

            const backtestSlice = state.cachedBacktest.slice(-state.backtestDays);
            const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
            const allPredictions = [
                ...backtestSlice.map((p) => ({ ...p, isPast: true })),
                ...forecastSlice.map((p) => ({ ...p, isPast: false })),
            ].sort((a, b) => toTimeValue(a.date) - toTimeValue(b.date));

            if (allPredictions.length === 0) return;

            ctx.save();

            allPredictions.forEach((pred, index) => {
                const predTs = toTimeValue(pred.date);
                if (!Number.isFinite(predTs)) return;
                const x = scales.x.getPixelForValue(predTs);

                // Width = distance to next day (pixel space).
                const nextPred = allPredictions[index + 1];
                let nextX = scales.x.getPixelForValue(toTimeValue(addDaysIso(pred.date, 1)));
                if (nextPred) {
                    const nextTs = toTimeValue(nextPred.date);
                    const gapDays = diffDays(predTs, nextTs);
                    if (Number.isFinite(nextTs) && Number.isFinite(gapDays) && gapDays <= 2) {
                        nextX = scales.x.getPixelForValue(nextTs);
                    }
                }
                const width = Math.max(nextX - x, 2);

                // Defensive: if scale returns NaN (date parse issue), skip drawing.
                if (!Number.isFinite(x) || !Number.isFinite(nextX)) return;

                // Skip if outside chart area (with a small margin for performance).
                if (x < chartArea.left - width || x > chartArea.right + width) return;

                let color;
                if (pred.direction === 'UP') {
                    color = pred.isPast
                        ? 'rgba(187, 247, 208, 0.45)' // light green (past)
                        : 'rgba(74, 222, 128, 0.55)'; // saturated green (future)
                } else if (pred.direction === 'DOWN') {
                    color = pred.isPast
                        ? 'rgba(254, 202, 202, 0.45)' // light red/pink (past)
                        : 'rgba(248, 113, 113, 0.55)'; // saturated red (future)
                } else {
                    color = 'rgba(150, 150, 150, 0.2)';
                }

                const left = Math.max(chartArea.left, x);
                const right = Math.min(chartArea.right, x + width);
                if (right <= left) return;

                ctx.fillStyle = color;
                ctx.fillRect(left, chartArea.top, right - left, chartArea.bottom - chartArea.top);
            });

            // "Today" line for orientation:
            // Prefer backend-provided anchor date to avoid timezone mismatch
            // between server UTC day and client local day.
            const nowAnchor = forecastSlice.find((p) => p?.is_now_anchor === true);
            const todayStr = nowAnchor?.date || toIsoDateLocal(new Date());
            const todayX = scales.x.getPixelForValue(toTimeValue(todayStr));
            if (todayX >= chartArea.left && todayX <= chartArea.right) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(todayX, chartArea.top);
                ctx.lineTo(todayX, chartArea.bottom);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.font = '11px Inter, sans-serif';
                ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.fillText('Today', todayX + 5, chartArea.top + 15);
            }

            ctx.restore();
        },
    };
}

function buildSplitBandPlugin() {
    return {
        id: 'splitBand',
        afterDraw: (chart) => {
            const { ctx, chartArea, scales } = chart;
            if (!chartArea || !scales?.x) return;

            const backtestSlice = state.cachedBacktest.slice(-state.backtestDays);
            const segments = buildSplitSegments(backtestSlice);
            if (segments.length === 0) return;

            const bandHeight = 14;
            const y = chartArea.top;

            ctx.save();
            ctx.font = '11px Inter, sans-serif';
            ctx.textBaseline = 'middle';

            // Draw a thin band at the top of the chart area.
            segments.forEach((seg) => {
                const color = splitColor(seg.split);
                if (!color) return;

                const x0 = scales.x.getPixelForValue(toTimeValue(seg.startDate));

                // End pixel: use endDate + 1 day so the segment includes the full last day.
                const endPlusOne = addDaysIso(seg.endDate, 1);
                const x1 = scales.x.getPixelForValue(toTimeValue(endPlusOne));

                const left = Math.max(chartArea.left, x0);
                const right = Math.min(chartArea.right, x1);
                const width = right - left;
                if (width <= 0) return;

                ctx.fillStyle = color;
                ctx.fillRect(left, y, width, bandHeight);

                // Text label (only if there is enough space)
                if (width >= 70) {
                    const label =
                        seg.split === 'train' ? 'Train' : seg.split === 'val' ? 'Validation' : 'Test';
                    ctx.fillStyle = 'rgba(10, 10, 15, 0.85)';
                    ctx.fillText(label, left + width / 2 - ctx.measureText(label).width / 2, y + bandHeight / 2);
                }
            });

            // A thin separator line under the band for clarity.
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(chartArea.left, y + bandHeight + 0.5);
            ctx.lineTo(chartArea.right, y + bandHeight + 0.5);
            ctx.stroke();

            ctx.restore();
        },
    };
}

export function destroyChart() {
    if (state.chart) {
        state.chart.destroy();
        state.chart = null;
    }
}

export function initializeChart() {
    if (!elements.chartCanvas) return;

    const ctx = elements.chartCanvas.getContext('2d');

    const backtestSlice = state.cachedBacktest.slice(-state.backtestDays);
    const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
    // Actual line source priority:
    // 1) Fresh market prices from /api/historical (can be newer than backtest cache)
    // 2) Fallback to actual_price embedded in backtest cache
    const rawHistoricalSource = Array.isArray(state.cachedActualPrices) && state.cachedActualPrices.length > 0
        ? state.cachedActualPrices.slice(-state.backtestDays)
        : backtestSlice
            .filter((p) => p.actual_price != null)
            .map((p) => ({ date: p.date, price: p.actual_price }));

    const historicalSource = clipHistoricalToBacktestRange(rawHistoricalSource, backtestSlice);
    const historicalData = buildActualLinePoints(historicalSource, 2);

    // Always include "today" anchor price if backend provided it in forecast cache.
    const nowAnchor = forecastSlice.find((p) => p?.is_now_anchor === true);
    if (nowAnchor && nowAnchor.simulated_price != null) {
        const anchorX = toTimeValue(nowAnchor.date);
        const exists = historicalData.some((pt) => pt.x === anchorX);
        if (!exists) {
            historicalData.push({ x: anchorX, y: Number(nowAnchor.simulated_price) });
        }
    }
    historicalData.sort((a, b) => a.x - b.x);

    const forecastData = forecastSlice.map((p) => ({ x: toTimeValue(p.date), y: p.simulated_price || 0 }));

    const predictionBackgroundPlugin = buildPredictionBackgroundPlugin();
    const splitBandPlugin = buildSplitBandPlugin();

    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Actual Price',
                    data: historicalData,
                    borderColor: 'rgba(255, 255, 255, 0.85)',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointStyle: false,
                    pointHitRadius: 10,
                    borderWidth: 2,
                    order: 1,
                },
                {
                    label: 'Forecast',
                    data: forecastData,
                    // The forecast line must be visible on a dark background.
                    // We use the project's gold accent and a dashed stroke so it
                    // can't be confused with the white "Actual Price" line.
                    borderColor: 'rgba(247, 183, 49, 0.85)',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointStyle: false,
                    pointHitRadius: 10,
                    borderWidth: 2,
                    borderDash: [7, 5],
                    order: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: CONFIG.CHART_ANIMATION_DURATION },
            // IMPORTANT:
            // We use `mode: "x"` so the tooltip matches points by *date* (x value),
            // NOT by array index. Index-based tooltips can incorrectly show an
            // "Actual Price" value for a future forecast day (because dataset lengths differ).
            interaction: { intersect: false, mode: 'x' },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        padding: 20,
                        font: { family: "'Inter', sans-serif" },
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
                    mode: 'x',
                    intersect: false,
                    callbacks: {
                        label: (context) => {
                            const label = context.dataset.label;
                            const value = context.parsed.y;

                            if (label === 'Forecast') {
                                const pred = forecastSlice[context.dataIndex];
                                if (!pred) return `Forecast: $${value.toLocaleString()}`;
                                return [
                                    `Forecast: $${value.toLocaleString()}`,
                                    `Direction: ${pred.direction}`,
                                    `Confidence: ${((pred.confidence ?? 0.5) * 100).toFixed(1)}%`,
                                ];
                            }

                            if (label === 'Actual Price') {
                                // Prefer a local ISO date string so it matches our `backtestByDate` keys.
                                const dateStr = toIsoDateLocal(new Date(context.parsed.x));
                                const row = state.backtestByDate.get(dateStr);
                                const lines = [`Actual: $${value.toLocaleString()}`];

                                if (row) {
                                    const split = row.split ? row.split.toUpperCase() : 'UNKNOWN';
                                    const conf = ((row.confidence ?? 0.5) * 100).toFixed(1);
                                    lines.push(`Split: ${split}`);
                                    lines.push(`Pred: ${row.direction} (${conf}%)`);
                                    if (row.actual_direction) lines.push(`Actual Label: ${row.actual_direction}`);
                                    if (row.correct === true) lines.push('Correct: YES');
                                    if (row.correct === false) lines.push('Correct: NO');
                                }

                                return lines;
                            }

                            return `${label}: $${value.toLocaleString()}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day', displayFormats: { day: 'MMM d' } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        maxRotation: 45,
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        callback: (v) => '$' + Number(v).toLocaleString(),
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                },
            },
        },
        plugins: [predictionBackgroundPlugin, splitBandPlugin],
    });

    // Hide loading overlay
    elements.chartLoading?.classList.add('hidden');
}
