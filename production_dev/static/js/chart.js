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

function toIsoDateUTC(dateObj) {
    // Format a Date as "YYYY-MM-DD" in UTC (timezone-safe).
    return dateObj.toISOString().slice(0, 10);
}

function addDaysIsoUTC(dateStr, days) {
    // Add N days to an ISO date string in UTC (timezone-safe).
    const d = new Date(`${dateStr}T00:00:00Z`);
    d.setUTCDate(d.getUTCDate() + days);
    return toIsoDateUTC(d);
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
            ];

            if (allPredictions.length === 0) return;

            ctx.save();

            allPredictions.forEach((pred, index) => {
                // IMPORTANT: Pass the ISO date string directly.
                // Using `new Date("YYYY-MM-DD")` can shift by timezone in some browsers.
                const x = scales.x.getPixelForValue(pred.date);

                // Width = distance to next day (pixel space).
                const nextPred = allPredictions[index + 1];
                let nextX = x + 10;
                if (nextPred) {
                    nextX = scales.x.getPixelForValue(nextPred.date);
                }
                const width = Math.max(nextX - x, 2);

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

                ctx.fillStyle = color;
                ctx.fillRect(
                    Math.max(x, chartArea.left),
                    chartArea.top,
                    Math.min(width, chartArea.right - x),
                    chartArea.bottom - chartArea.top,
                );
            });

            // "Today" line for orientation
            const todayStr = toIsoDateUTC(new Date());
            const todayX = scales.x.getPixelForValue(todayStr);
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

                const x0 = scales.x.getPixelForValue(seg.startDate);

                // End pixel: use endDate + 1 day so the segment includes the full last day.
                const endPlusOne = addDaysIsoUTC(seg.endDate, 1);
                const x1 = scales.x.getPixelForValue(endPlusOne);

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
    const historicalData = backtestSlice
        .filter((p) => p.actual_price != null)
        .map((p) => ({ x: p.date, y: p.actual_price }));

    const forecastSlice = state.cachedForecast.slice(0, state.forecastDays);
    const forecastData = forecastSlice.map((p) => ({ x: p.date, y: p.simulated_price || 0 }));

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
                    borderColor: 'rgba(50, 50, 50, 0.9)',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointStyle: false,
                    pointHitRadius: 10,
                    borderWidth: 2,
                    order: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: CONFIG.CHART_ANIMATION_DURATION },
            interaction: { intersect: false, mode: 'index' },
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
                                // Prefer the original x value (string) to avoid timezone shifts.
                                const rawX = context.raw?.x;
                                const dateStr = typeof rawX === 'string' ? rawX : toIsoDateUTC(new Date(context.parsed.x));
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
