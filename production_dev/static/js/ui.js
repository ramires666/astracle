/**
 * UI Rendering Helpers
 *
 * These functions only touch the DOM. They do not fetch data and do not build charts.
 * This separation makes it much easier to debug.
 */

import { elements } from './elements.js';

function ensureHeaderBadge(id, labelText) {
    let badge = document.getElementById(id);
    if (badge) return badge;

    badge = document.createElement('div');
    badge.id = id;
    badge.className = 'stat-badge';
    badge.innerHTML = `
        <span class="stat-label">${labelText}</span>
        <span class="stat-value" id="${id}-value">--</span>
    `;
    elements.headerStats.appendChild(badge);
    return badge;
}

export function showLoading(show) {
    if (!elements.chartLoading) return;
    if (show) {
        elements.chartLoading.classList.remove('hidden');
    } else {
        elements.chartLoading.classList.add('hidden');
    }
}

export function updateModelStatus(healthData) {
    const statusBadge = elements.headerStats?.querySelector('.stat-badge');
    if (statusBadge) statusBadge.classList.remove('loading');

    if (!elements.modelStatus) return;

    if (healthData?.status === 'healthy') {
        elements.modelStatus.textContent = 'Online';
        elements.modelStatus.style.color = 'var(--accent-green)';
    } else {
        elements.modelStatus.textContent = 'Error';
        elements.modelStatus.style.color = 'var(--accent-red)';
    }
}

export function updateModelInfo(info) {
    const config = info?.config || {};

    // Big badge on the model card is reserved for notebook metrics.
    if (elements.accuracyBadge) {
        const rMin = Number(config.r_min || 0);
        elements.accuracyBadge.textContent = `R_MIN: ${rMin.toFixed(3)}`;
    }

    if (elements.natalDate) elements.natalDate.textContent = config.birth_date || '--';
    if (elements.coordMode) elements.coordMode.textContent = config.coord_mode || '--';
    if (elements.orbMult) elements.orbMult.textContent = config.orb_mult ?? '--';
    if (elements.nEstimators) elements.nEstimators.textContent = config.n_estimators ?? '--';
}

export function updateBacktestSliderLimits(maxDays) {
    if (!elements.backtestSlider || !elements.backtestValue) return;

    // Keep the slider usable even if cache is empty.
    const safeMax = Math.max(30, Number(maxDays || 0));
    elements.backtestSlider.max = String(safeMax);

    // If current value is larger than max, clamp it.
    const current = Number(elements.backtestSlider.value || 0);
    if (current > safeMax) {
        elements.backtestSlider.value = String(safeMax);
    }

    elements.backtestValue.textContent = `${elements.backtestSlider.value} days`;
}

export function updateForecastSliderLabel(days) {
    if (!elements.daysValue) return;
    elements.daysValue.textContent = `${days} days`;
}

export function updateBacktestSliderLabel(days) {
    if (!elements.backtestValue) return;
    elements.backtestValue.textContent = `${days} days`;
}

export function updateBacktestStatsBadges(stats) {
    // We want to avoid confusing users by mixing "notebook metrics" with
    // random accuracy definitions. So we only show TEST-split metrics here.
    if (!stats || stats.total === 0) return;

    if (typeof stats.r_min === 'number') {
        ensureHeaderBadge('backtest-rmin-badge', 'Backtest Test R_MIN');
        const el = document.getElementById('backtest-rmin-badge-value');
        if (el) {
            el.textContent = stats.r_min.toFixed(3);
            el.style.color = 'var(--accent-gold)';
        }
    }

    if (typeof stats.mcc === 'number') {
        ensureHeaderBadge('backtest-mcc-badge', 'Backtest Test MCC');
        const el = document.getElementById('backtest-mcc-badge-value');
        if (el) {
            el.textContent = stats.mcc.toFixed(3);
            el.style.color = 'var(--accent-blue)';
        }
    }

    // A simple accuracy badge is still useful for non-technical users.
    if (typeof stats.accuracy === 'number') {
        ensureHeaderBadge('backtest-acc-badge', 'Backtest Test Acc');
        const el = document.getElementById('backtest-acc-badge-value');
        if (el) {
            el.textContent = `${(stats.accuracy * 100).toFixed(1)}%`;

            if (stats.accuracy >= 0.55) {
                el.style.color = 'var(--accent-green)';
            } else if (stats.accuracy >= 0.50) {
                el.style.color = 'var(--accent-gold)';
            } else {
                el.style.color = 'var(--accent-red)';
            }
        }
    }

    // Optional: show tuned threshold, if cache included it.
    if (typeof stats.decision_threshold === 'number') {
        ensureHeaderBadge('backtest-thr-badge', 'Val Threshold');
        const el = document.getElementById('backtest-thr-badge-value');
        if (el) {
            el.textContent = stats.decision_threshold.toFixed(2);
            el.style.color = 'rgba(255, 255, 255, 0.75)';
        }
    }

    // Also print the actual date ranges (people trust dates more than ratios).
    // This satisfies the "be honest: show train/val/test periods" requirement
    // even when the colored band is hard to read on a small mobile screen.
    if (elements.splitCaption) {
        const r = stats.split_ranges;
        if (r?.train && r?.val && r?.test) {
            elements.splitCaption.textContent =
                `Splits: Train ${r.train.start} -> ${r.train.end} | ` +
                `Validation ${r.val.start} -> ${r.val.end} | ` +
                `Test ${r.test.start} -> ${r.test.end}`;
        } else {
            elements.splitCaption.textContent = '';
        }
    }
}

export function updateForecastTable(predictions) {
    if (!elements.tableBody) return;

    if (!predictions || predictions.length === 0) {
        elements.tableBody.innerHTML = '';
        return;
    }

    elements.tableBody.innerHTML = predictions
        .map((pred) => {
            const directionClass = pred.direction === 'UP' ? 'direction-up' : 'direction-down';
            const directionIcon = pred.direction === 'UP' ? '▲' : '▼';
            const confPct = ((pred.confidence ?? 0.5) * 100).toFixed(1);
            const simPrice = Number(pred.simulated_price || 0);

            return `
                <tr>
                    <td>${pred.date}</td>
                    <td class="${directionClass}">${directionIcon} ${pred.direction}</td>
                    <td>${confPct}%</td>
                    <td>$${simPrice.toLocaleString()}</td>
                </tr>
            `;
        })
        .join('');
}
