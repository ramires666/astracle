/**
 * UI Rendering Helpers
 *
 * These functions only touch the DOM. They do not fetch data and do not build charts.
 * This separation makes it easier to maintain and debug the UI layer.
 */

import { elements } from './elements.js';

function setValue(id, text, color = null) {
    const node = document.getElementById(id);
    if (!node) return;
    node.textContent = text;
    if (color) node.style.color = color;
}

export function renderLucideIcons() {
    if (!window.lucide || typeof window.lucide.createIcons !== 'function') return;
    window.lucide.createIcons();
}

function formatUtcTimestamp(value) {
    if (!value) return '--';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    const y = parsed.getUTCFullYear();
    const m = String(parsed.getUTCMonth() + 1).padStart(2, '0');
    const d = String(parsed.getUTCDate()).padStart(2, '0');
    const hh = String(parsed.getUTCHours()).padStart(2, '0');
    const mm = String(parsed.getUTCMinutes()).padStart(2, '0');
    return `${y}-${m}-${d} ${hh}:${mm} UTC`;
}

function formatLift(value, baseline) {
    // Example: value=0.631 vs baseline=0.50 -> "+13.1pp"
    const diff = Number(value) - Number(baseline);
    const sign = diff >= 0 ? '+' : '-';
    const abs = Math.abs(diff * 100).toFixed(1);
    return `${sign}${abs}pp`;
}

function describeAccuracy(acc) {
    const value = Number(acc);
    if (value >= 0.55) return { suffix: '', color: 'var(--accent-green)' };
    if (value >= 0.52) return { suffix: ' (weak)', color: 'var(--accent-gold)' };
    if (value >= 0.48) return { suffix: ' (~ random)', color: 'var(--accent-red)' };
    return { suffix: ' (worse)', color: 'var(--accent-red)' };
}

function describeRMin(rMin) {
    const value = Number(rMin);
    if (value >= 0.55) return { suffix: '', color: 'var(--accent-gold)' };
    if (value >= 0.50) return { suffix: ' (weak)', color: 'var(--accent-gold)' };
    if (value >= 0.45) return { suffix: ' (~ random)', color: 'var(--accent-red)' };
    return { suffix: ' (bad)', color: 'var(--accent-red)' };
}

function describeMcc(mcc) {
    const value = Number(mcc);
    if (value >= 0.30) return { suffix: '', color: 'var(--accent-blue)' };
    if (value >= 0.15) return { suffix: ' (moderate)', color: 'var(--accent-blue)' };
    if (value >= 0.05) return { suffix: ' (weak)', color: 'var(--accent-red)' };
    return { suffix: ' (~ random)', color: 'var(--accent-red)' };
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
    if (!elements.modelStatus) return;

    if (healthData?.status === 'healthy') {
        setValue('model-status', 'Online', 'var(--accent-green)');
    } else {
        setValue('model-status', 'Error', 'var(--accent-red)');
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

    // Keep slider usable even if cache is empty.
    const safeMax = Math.max(30, Number(maxDays || 0));
    elements.backtestSlider.max = String(safeMax);

    // Clamp current value to new max.
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
    if (!stats || stats.total === 0) return;

    // Show both validation and test snapshots.
    const testStats = stats?.splits?.test || stats;
    const valStats = stats?.splits?.val || null;

    if (typeof testStats.r_min === 'number') {
        const tone = describeRMin(testStats.r_min);
        setValue('quality-test-rmin', `${testStats.r_min.toFixed(3)}${tone.suffix}`, tone.color);
    }

    if (typeof testStats.mcc === 'number') {
        const tone = describeMcc(testStats.mcc);
        setValue('quality-test-mcc', `${testStats.mcc.toFixed(3)}${tone.suffix}`, tone.color);
    }

    if (typeof testStats.accuracy === 'number') {
        const acc = Number(testStats.accuracy);
        const lift = formatLift(acc, 0.5);
        const tone = describeAccuracy(acc);
        setValue('quality-test-acc', `${(acc * 100).toFixed(1)}% (${lift})${tone.suffix}`, tone.color);
    }

    if (valStats && typeof valStats.accuracy === 'number') {
        const acc = Number(valStats.accuracy);
        const lift = formatLift(acc, 0.5);
        const tone = describeAccuracy(acc);
        setValue('quality-val-acc', `${(acc * 100).toFixed(1)}% (${lift})${tone.suffix}`, tone.color);
    }

    if (valStats && typeof valStats.r_min === 'number') {
        const tone = describeRMin(valStats.r_min);
        setValue('quality-val-rmin', `${valStats.r_min.toFixed(3)}${tone.suffix}`, tone.color);
    }

    if (typeof stats.decision_threshold === 'number') {
        setValue('model-threshold', stats.decision_threshold.toFixed(2), 'rgba(255, 255, 255, 0.78)');
    }

    if (!elements.splitCaption) return;
    const ranges = stats.split_ranges;
    if (ranges?.train && ranges?.val && ranges?.test) {
        elements.splitCaption.textContent =
            `Splits: Train ${ranges.train.start} -> ${ranges.train.end} | ` +
            `Validation ${ranges.val.start} -> ${ranges.val.end} | ` +
            `Test ${ranges.test.start} -> ${ranges.test.end}`;
    } else {
        elements.splitCaption.textContent = '';
    }
}

export function updateOperationalBadges({
    modelInfo = null,
    cacheInfo = null,
    marketSummary = null,
    refreshStatus = null,
} = {}) {
    const artifact = modelInfo?.artifact || {};
    const evalId = artifact?.source_eval_id;
    const trainedAt = artifact?.trained_at_utc;
    const threshold = Number(artifact?.decision_threshold ?? modelInfo?.config?.decision_threshold);

    if (evalId != null) setValue('model-eval-id', String(evalId), 'var(--accent-gold)');
    if (trainedAt) setValue('model-trained-at', formatUtcTimestamp(trainedAt), 'rgba(255, 255, 255, 0.8)');
    if (Number.isFinite(threshold)) {
        setValue('model-threshold', threshold.toFixed(2), 'rgba(255, 255, 255, 0.78)');
    }

    if (marketSummary?.end_date) {
        setValue('runtime-market-date', String(marketSummary.end_date), 'var(--accent-blue)');
    }
    if (cacheInfo?.last_updated) {
        setValue('runtime-cache-updated', formatUtcTimestamp(cacheInfo.last_updated), 'rgba(255, 255, 255, 0.82)');
    }

    const refreshLast = refreshStatus?.last_result || null;
    if (refreshLast?.checked_at_utc) {
        const action = refreshLast.action || 'none';
        const reason = refreshLast.reason || 'n/a';
        setValue(
            'runtime-refresh-cycle',
            `${action} (${reason})`,
            action === 'none' ? 'rgba(255, 255, 255, 0.76)' : 'var(--accent-green)',
        );
    }

    if (elements.v2LiveEval) {
        elements.v2LiveEval.textContent = `eval_id: ${evalId != null ? String(evalId) : '--'}`;
    }
    if (elements.v2LiveCache) {
        const ts = cacheInfo?.last_updated ? formatUtcTimestamp(cacheInfo.last_updated) : '--';
        elements.v2LiveCache.textContent = `cache: ${ts}`;
    }
}

export function updateForecastTable(predictions) {
    if (!elements.tableBody) return;

    if (!predictions || predictions.length === 0) {
        elements.tableBody.innerHTML = '';
        renderLucideIcons();
        return;
    }

    elements.tableBody.innerHTML = predictions
        .map((pred) => {
            const directionClass = pred.direction === 'UP' ? 'direction-up' : 'direction-down';
            const directionIcon = pred.direction === 'UP' ? 'arrow-up-right' : 'arrow-down-right';
            const confPct = ((pred.confidence ?? 0.5) * 100).toFixed(1);
            const simPrice = Number(pred.simulated_price || 0);

            return `
                <tr>
                    <td>${pred.date}</td>
                    <td class="${directionClass}">
                        <span class="direction-cell">
                            <i class="direction-icon" data-lucide="${directionIcon}" aria-hidden="true"></i>
                            <span>${pred.direction}</span>
                        </span>
                    </td>
                    <td>${confPct}%</td>
                    <td>$${simPrice.toLocaleString()}</td>
                </tr>
            `;
        })
        .join('');

    renderLucideIcons();
}
