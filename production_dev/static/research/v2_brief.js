/**
 * V2 statistical brief live bindings.
 *
 * Purpose:
 * - Avoid hardcoded top badges in the publication-style brief.
 * - Pull current values from summary JSON + production API.
 * - Refresh periodically so daily retrain/cache updates are visible.
 */

const AUTO_REFRESH_MS = 15 * 60 * 1000;

function el(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const node = el(id);
    if (!node) return;
    node.textContent = value;
}

function fmtNum(value, digits = 3) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return n.toFixed(digits);
}

function fmtSigned(value, digits = 3) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    const sign = n >= 0 ? '+' : '';
    return `${sign}${n.toFixed(digits)}`;
}

function fmtP(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    if (n < 1e-4) return n.toExponential(2);
    return n.toFixed(5);
}

async function fetchJson(url) {
    try {
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) return null;
        return await resp.json();
    } catch (_error) {
        return null;
    }
}

function applySummary(summary) {
    if (!summary) return;

    const n = Number(summary.n_candidates);
    const c005 = Number(summary.count_p_lt_0_05);
    const c001 = Number(summary.count_p_lt_0_01);
    const dPos = Number(summary.count_delta_gt_0);
    const sharePos = Number(summary.share_delta_gt_0);

    setText('q-n-candidates', Number.isFinite(n) ? String(n) : '--');
    setText('q-p-lt-005', Number.isFinite(c005) && Number.isFinite(n) ? `${c005} / ${n}` : '--');
    setText('q-delta-positive', Number.isFinite(sharePos) ? `${(sharePos * 100).toFixed(1)}%` : '--');
    setText('q-fisher-p', fmtP(summary.fisher_p));

    setText('m-p-lt-005', Number.isFinite(c005) && Number.isFinite(n) ? `${c005} / ${n}` : '--');
    setText('m-p-lt-001', Number.isFinite(c001) && Number.isFinite(n) ? `${c001} / ${n}` : '--');
    setText('m-delta-pos', Number.isFinite(dPos) && Number.isFinite(n) ? `${dPos} / ${n}` : '--');
    setText('m-sign-p', fmtP(summary.binom_p_delta_gt_0));
    setText('m-fisher-p', fmtP(summary.fisher_p));
    setText('m-stouffer', `z=${fmtNum(summary.stouffer_z, 3)}, p=${fmtP(summary.stouffer_p)}`);
    setText(
        'm-mean-delta',
        `${fmtNum(summary.mean_delta, 3)} [${fmtNum(summary.mean_delta_ci95_low, 3)}, ${fmtNum(summary.mean_delta_ci95_high, 3)}]`,
    );

    setText('b-p', fmtP(summary.best_p_shift_weighted_hit));
    setText('b-delta', fmtSigned(summary.best_delta_weighted_hit_vs_null, 4));
    setText(
        'b-hit',
        `${fmtNum(summary.best_weighted_hit_rate_obs, 4)} / ${fmtNum(summary.best_weighted_hit_rate_null_mean, 4)}`,
    );
    setText('b-profit', fmtSigned(summary.best_test_profit_y_obj, 5));
    setText('b-rmin', fmtNum(summary.best_test_recall_min, 3));
    setText('b-gap', fmtNum(summary.best_test_recall_gap, 3));
    setText('b-mcc', fmtNum(summary.best_test_mcc, 3));
}

function applyLive(modelInfo, fullPredictions, historical) {
    const evalId = modelInfo?.artifact?.source_eval_id;
    const marketDate = historical?.summary?.end_date;
    const cacheUpdated = fullPredictions?.cache_info?.last_updated;

    setText('q-active-eval', evalId != null ? String(evalId) : '--');
    setText('q-market-date', marketDate || '--');

    if (cacheUpdated) {
        const stamp = new Date(cacheUpdated);
        if (!Number.isNaN(stamp.getTime())) {
            const y = stamp.getUTCFullYear();
            const m = String(stamp.getUTCMonth() + 1).padStart(2, '0');
            const d = String(stamp.getUTCDate()).padStart(2, '0');
            const hh = String(stamp.getUTCHours()).padStart(2, '0');
            const mm = String(stamp.getUTCMinutes()).padStart(2, '0');
            const t = `${y}-${m}-${d} ${hh}:${mm} UTC`;
            setText('q-active-eval', evalId != null ? `${evalId} (${t})` : t);
        }
    }
}

async function refreshBrief() {
    const [summary, modelInfo, fullPredictions, historical] = await Promise.all([
        fetchJson('/static/research/v2_assets/summary_metrics.json'),
        fetchJson('/api/config'),
        fetchJson('/api/predictions/full'),
        fetchJson('/api/historical?days=14'),
    ]);

    applySummary(summary);
    applyLive(modelInfo, fullPredictions, historical);
}

document.addEventListener('DOMContentLoaded', async () => {
    await refreshBrief();
    setInterval(() => {
        refreshBrief().catch((error) => console.warn('v2 brief refresh failed:', error));
    }, AUTO_REFRESH_MS);
});
