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

function formatLift(value, baseline) {
    // Format the difference vs a baseline (default is 0.50 for random guess).
    // Example: value=0.631 -> "+13.1pp"
    const diff = Number(value) - Number(baseline);
    const sign = diff >= 0 ? '+' : '-';
    const abs = Math.abs(diff * 100).toFixed(1);
    return `${sign}${abs}pp`;
}

function isNearRandom(value, baseline = 0.5, tolerance = 0.03) {
    // Treat values within ±3 percentage points of random as "near random".
    // This avoids over-selling tiny differences.
    return Math.abs(Number(value) - Number(baseline)) <= tolerance;
}

function describeAccuracy(acc) {
    // Return a human-friendly label + color for accuracy.
    // We keep thresholds simple so a non-technical user can trust the tone:
    // - >= 55%: "good enough to highlight"
    // - 52%..55%: "weak edge"
    // - 48%..52%: "≈ random"
    // - < 48%: "worse than random"
    const value = Number(acc);
    if (value >= 0.55) {
        return { suffix: '', color: 'var(--accent-green)' };
    }
    if (value >= 0.52) {
        return { suffix: ' (weak)', color: 'var(--accent-gold)' };
    }
    if (value >= 0.48) {
        return { suffix: ' (≈ random)', color: 'var(--accent-red)' };
    }
    return { suffix: ' (worse)', color: 'var(--accent-red)' };
}

function describeRMin(rMin) {
    // R_MIN is the minimum recall across classes.
    // We interpret it with similar thresholds to accuracy, but slightly stricter
    // because R_MIN is already a "worst-case" score.
    const value = Number(rMin);
    if (value >= 0.55) {
        return { suffix: '', color: 'var(--accent-gold)' };
    }
    if (value >= 0.50) {
        return { suffix: ' (weak)', color: 'var(--accent-gold)' };
    }
    if (value >= 0.45) {
        return { suffix: ' (≈ random)', color: 'var(--accent-red)' };
    }
    return { suffix: ' (bad)', color: 'var(--accent-red)' };
}

function describeMcc(mcc) {
    // MCC is a correlation-like score:
    // - 0.0 means random
    // - 1.0 means perfect
    const value = Number(mcc);
    if (value >= 0.30) {
        return { suffix: '', color: 'var(--accent-blue)' };
    }
    if (value >= 0.15) {
        return { suffix: ' (moderate)', color: 'var(--accent-blue)' };
    }
    if (value >= 0.05) {
        return { suffix: ' (weak)', color: 'var(--accent-red)' };
    }
    return { suffix: ' (≈ random)', color: 'var(--accent-red)' };
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

    // We now show BOTH validation and test to avoid "looks too good" confusion.
    // The headline test numbers stay, but the validation badges reveal
    // whether the model generalizes or collapses to ~random.
    const testStats = stats?.splits?.test || stats;
    const valStats = stats?.splits?.val || null;

    if (typeof testStats.r_min === 'number') {
        ensureHeaderBadge('backtest-rmin-badge', 'Backtest Test R_MIN');
        const el = document.getElementById('backtest-rmin-badge-value');
        if (el) {
            const tone = describeRMin(testStats.r_min);
            el.textContent = `${testStats.r_min.toFixed(3)}${tone.suffix}`;
            el.style.color = tone.color;
        }
    }

    if (typeof testStats.mcc === 'number') {
        ensureHeaderBadge('backtest-mcc-badge', 'Backtest Test MCC');
        const el = document.getElementById('backtest-mcc-badge-value');
        if (el) {
            const tone = describeMcc(testStats.mcc);
            el.textContent = `${testStats.mcc.toFixed(3)}${tone.suffix}`;
            el.style.color = tone.color;
        }
    }

    // A simple accuracy badge is still useful for non-technical users.
    if (typeof testStats.accuracy === 'number') {
        ensureHeaderBadge('backtest-acc-badge', 'Backtest Test Acc');
        const el = document.getElementById('backtest-acc-badge-value');
        if (el) {
            const acc = Number(testStats.accuracy);
            const lift = formatLift(acc, 0.5);
            const tone = describeAccuracy(acc);
            el.textContent = `${(acc * 100).toFixed(1)}% (${lift})${tone.suffix}`;
            el.style.color = tone.color;
        }
    }

    // Validation badges (to make honesty visible at a glance).
    if (valStats && typeof valStats.accuracy === 'number') {
        ensureHeaderBadge('backtest-val-acc-badge', 'Validation Acc');
        const el = document.getElementById('backtest-val-acc-badge-value');
        if (el) {
            const acc = Number(valStats.accuracy);
            const lift = formatLift(acc, 0.5);
            const tone = describeAccuracy(acc);
            el.textContent = `${(acc * 100).toFixed(1)}% (${lift})${tone.suffix}`;
            el.style.color = tone.color;
        }
    }

    if (valStats && typeof valStats.r_min === 'number') {
        ensureHeaderBadge('backtest-val-rmin-badge', 'Validation R_MIN');
        const el = document.getElementById('backtest-val-rmin-badge-value');
        if (el) {
            const tone = describeRMin(valStats.r_min);
            el.textContent = `${valStats.r_min.toFixed(3)}${tone.suffix}`;
            el.style.color = tone.color;
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
