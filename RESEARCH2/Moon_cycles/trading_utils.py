"""
Trading sanity-check backtest helpers for research notebooks.

Goal: convert model signals into transparent long/flat (and plotting) PnL.
Assumptions: daily-close execution, close-to-close stop approximation, fixed fee.
This is intentionally simple and is not a production-grade execution simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from .eval_visuals import VisualizationConfig, _draw_direction_background, _draw_split_bands, _style_axis, _style_figure, _style_legend


@dataclass(frozen=True)
class TradingConfig:
    """
    Parameters that control the simple trading simulation.

    We keep them explicit (instead of many function arguments) because:
    - notebooks become easier to read,
    - caching / comparing runs becomes easier.
    """

    fee_rate: float = 0.001  # 0.1% per trade side (buy OR sell).
    stop_loss_pct: float = 0.0  # 0.0 disables stop-loss.
    exit_on_no_signal: bool = False  # If True: when signal is NaN, we go flat.
    close_final_position: bool = True  # Close any open position at the end (with fee).
    initial_cash: float = 1.0  # We start with "1 unit" of cash to keep curves normalized.


def build_signal_from_proba(
    proba_up: Iterable[float],
    threshold_up: float = 0.5,
    threshold_down: float = 0.5,
) -> np.ndarray:
    """
    Convert model probabilities into a 3-state trading signal.

    Output encoding (float array so we can represent NaN):
    - 1.0  -> "UP signal"    (we want to be LONG)
    - 0.0  -> "DOWN signal"  (we want to be FLAT)
    - NaN  -> "NO SIGNAL"    (abstain / unsure)

    Rules:
    - If proba >= threshold_up  -> UP
    - If proba <= threshold_down -> DOWN
    - Else -> NO SIGNAL

    Typical usage:
    - Classic "always decide": threshold_up=0.5, threshold_down=0.5
    - Add neutral zone:       threshold_up=0.55, threshold_down=0.45
    """
    p = np.asarray(list(proba_up), dtype=float)
    out = np.full(p.shape[0], np.nan, dtype=float)
    out[p >= float(threshold_up)] = 1.0
    out[p <= float(threshold_down)] = 0.0
    return out


def _normalize_signal_col(sig: pd.Series) -> np.ndarray:
    """
    Convert a user-provided signal column to our internal encoding.

    We accept several "human" formats:
    - 1/0 ints
    - True/False booleans
    - NaN / None for no-signal

    We DO NOT accept strings like "UP"/"DOWN" here to keep code simple.
    """
    s = pd.Series(sig).copy()
    s = s.replace({True: 1.0, False: 0.0})
    s = pd.to_numeric(s, errors="coerce")  # non-numeric -> NaN
    return s.to_numpy(dtype=float)


def _ulcer_index_from_equity(equity: np.ndarray) -> float:
    """
    Ulcer Index (UI) is a drawdown-based risk metric.

    Intuition (simple):
    - UI is small if the equity curve has shallow / short drawdowns.
    - UI is large if the equity curve spends a lot of time far below its peak.

    Formula (standard):
    - drawdown_pct[t] = (peak[t] - equity[t]) / peak[t] * 100
    - UI = sqrt(mean(drawdown_pct^2))
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return float("nan")

    peak = np.maximum.accumulate(eq)
    peak = np.where(peak <= 0.0, np.nan, peak)
    dd_pct = (peak - eq) / peak * 100.0
    dd_pct = np.nan_to_num(dd_pct, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(dd_pct**2)))


def backtest_long_flat_signals(
    df: pd.DataFrame,
    signal_col: str,
    cfg: TradingConfig = TradingConfig(),
    date_col: str = "date",
    price_col: str = "close",
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Simulate a very simple long/flat strategy driven by model signals.

    Strategy rule (as requested, in plain words):
    - If model says "UP": buy BTC (go long).
    - If model says "DOWN": sell BTC (go flat in cash).
    - If model says "NO SIGNAL":
        - by default we HOLD the current position (stay as we are),
        - if cfg.exit_on_no_signal=True, we EXIT to cash.

    Portfolio model (kept simple):
    - We start with cfg.initial_cash units of cash (normalized, default = 1.0).
    - When we buy: we invest 100% of cash into BTC (so position size is "all-in").
    - When we sell: we sell 100% of BTC back into cash.

    Fees:
    - A fee is paid on every buy and every sell (cfg.fee_rate).

    Stop-loss (approximation with daily closes only):
    - If we are long and close <= entry_price * (1 - stop_loss_pct),
      we sell at that day's close.
    - We do NOT re-enter on the same day after a stop. This avoids churn.
    """
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"df must contain columns: {date_col!r}, {price_col!r}")
    if signal_col not in df.columns:
        raise ValueError(f"df must contain signal column: {signal_col!r}")

    df_bt = df.copy()
    df_bt[date_col] = pd.to_datetime(df_bt[date_col])
    df_bt = df_bt.sort_values(date_col).drop_duplicates(date_col).reset_index(drop=True)
    df_bt = df_bt.dropna(subset=[price_col]).reset_index(drop=True)

    dates = df_bt[date_col].to_numpy()
    prices = df_bt[price_col].astype(float).to_numpy()
    signals = _normalize_signal_col(df_bt[signal_col])

    fee = float(cfg.fee_rate)
    stop_pct = float(cfg.stop_loss_pct)

    cash = float(cfg.initial_cash)
    qty = 0.0  # BTC units we hold. qty=0 means we are flat.
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None
    entry_cash: Optional[float] = None  # cash right before we entered (for trade return calc)

    trades: List[Dict[str, object]] = []
    equity: List[float] = []
    pos: List[int] = []
    entry_marks: List[int] = []
    exit_marks: List[int] = []

    for i, (dt, px, sig) in enumerate(zip(dates, prices, signals)):
        # Mark-to-market: equity is cash + (qty * current price).
        # If we are long, this automatically reflects daily PnL.
        eq_before = cash + qty * float(px)

        stopped_today = False
        if qty > 0.0 and stop_pct > 0.0 and entry_price is not None:
            stop_level = float(entry_price) * (1.0 - stop_pct)
            if float(px) <= stop_level:
                # Stop-loss exit at current close (daily approximation).
                cash = qty * float(px) * (1.0 - fee)
                qty = 0.0
                exit_marks.append(i)
                stopped_today = True

                # Record the trade if we have a known entry.
                if entry_date is not None and entry_cash is not None:
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "entry_price": float(entry_price),
                            "exit_date": pd.to_datetime(dt),
                            "exit_price": float(px),
                            "exit_reason": "stop_loss",
                            "return_pct": float(cash / entry_cash - 1.0),
                            "holding_days": int((pd.to_datetime(dt) - entry_date).days),
                        }
                    )
                entry_price, entry_date, entry_cash = None, None, None

        # If stop-loss happened, we deliberately ignore "UP" signal on the same day.
        if not stopped_today:
            # Decide desired state for NEXT day based on today's signal.
            if np.isnan(sig):
                desired = "flat" if cfg.exit_on_no_signal else "keep"
            else:
                desired = "long" if int(sig) == 1 else "flat"

            if desired == "flat" and qty > 0.0:
                cash = qty * float(px) * (1.0 - fee)
                qty = 0.0
                exit_marks.append(i)

                if entry_date is not None and entry_cash is not None:
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "entry_price": float(entry_price) if entry_price is not None else float("nan"),
                            "exit_date": pd.to_datetime(dt),
                            "exit_price": float(px),
                            "exit_reason": "signal_down" if not np.isnan(sig) else "no_signal_exit",
                            "return_pct": float(cash / entry_cash - 1.0),
                            "holding_days": int((pd.to_datetime(dt) - entry_date).days),
                        }
                    )
                entry_price, entry_date, entry_cash = None, None, None

            if desired == "long" and qty == 0.0 and cash > 0.0:
                # Buy BTC with 100% of available cash.
                entry_cash = float(cash)
                qty = (cash * (1.0 - fee)) / float(px)
                cash = 0.0
                entry_price = float(px)
                entry_date = pd.to_datetime(dt)
                entry_marks.append(i)

        eq_after = cash + qty * float(px)
        equity.append(float(eq_after))
        pos.append(1 if qty > 0.0 else 0)

        if verbose and (i % 200 == 0 or i == len(prices) - 1):
            print(f"[bt] {i+1}/{len(prices)} date={pd.to_datetime(dt).date()} equity={eq_after:.4f} pos={pos[-1]}")

    # Optional: close any open position at the end so final equity is in cash.
    if cfg.close_final_position and qty > 0.0:
        px_last = float(prices[-1])
        cash = qty * px_last * (1.0 - fee)
        qty = 0.0
        exit_marks.append(len(prices) - 1)
        equity[-1] = float(cash)
        pos[-1] = 0

        if entry_date is not None and entry_cash is not None:
            trades.append(
                {
                    "entry_date": entry_date,
                    "entry_price": float(entry_price) if entry_price is not None else float("nan"),
                    "exit_date": pd.to_datetime(dates[-1]),
                    "exit_price": float(px_last),
                    "exit_reason": "end_close",
                    "return_pct": float(cash / entry_cash - 1.0),
                    "holding_days": int((pd.to_datetime(dates[-1]) - entry_date).days),
                }
            )

    equity_arr = np.asarray(equity, dtype=float)

    # Benchmark: buy-and-hold with the same initial cash and the same fee model.
    # - We "buy" at the first close and (optionally) "sell" at the last close.
    hold_qty = (float(cfg.initial_cash) * (1.0 - fee)) / float(prices[0]) if len(prices) else 0.0
    hold_equity = hold_qty * prices
    if cfg.close_final_position and len(prices):
        hold_equity = hold_equity * (1.0 - fee)

    # Drawdown series (in percent) for risk metrics + plotting.
    peak = np.maximum.accumulate(equity_arr)
    dd = (peak - equity_arr) / np.where(peak > 0.0, peak, np.nan) * 100.0
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)

    trades_df = pd.DataFrame(trades)
    winrate = float((trades_df["return_pct"] > 0.0).mean()) if not trades_df.empty else float("nan")

    metrics = {
        "final_equity": float(equity_arr[-1]) if equity_arr.size else float("nan"),
        "hold_final_equity": float(hold_equity[-1]) if len(hold_equity) else float("nan"),
        "return_pct": float(equity_arr[-1] / cfg.initial_cash - 1.0) if equity_arr.size else float("nan"),
        "hold_return_pct": float(hold_equity[-1] / cfg.initial_cash - 1.0) if len(hold_equity) else float("nan"),
        "excess_return_pct": float(equity_arr[-1] / hold_equity[-1] - 1.0) if equity_arr.size and len(hold_equity) else float("nan"),
        "num_trades": int(len(trades_df)),
        "winrate": float(winrate),
        "max_drawdown_pct": float(np.max(dd)) if dd.size else float("nan"),
        "ulcer_index": float(_ulcer_index_from_equity(equity_arr)),
        "exposure_pct": float(np.mean(pos) * 100.0) if len(pos) else float("nan"),
    }
    metrics["ulcer_adjusted_return"] = float(metrics["return_pct"] / max(metrics["ulcer_index"], 1e-12)) if np.isfinite(metrics["ulcer_index"]) else float("nan")

    out = df_bt.copy()
    out["signal"] = signals
    out["position"] = pos
    out["equity"] = equity_arr
    out["hold_equity"] = hold_equity
    out["drawdown_pct"] = dd

    return {
        "config": cfg,
        "equity_df": out,
        "trades": trades_df,
        "metrics": metrics,
        "entry_idx": entry_marks,
        "exit_idx": exit_marks,
    }


def sweep_trading_params(
    df: pd.DataFrame,
    signal_col: str,
    stop_losses: Sequence[float],
    exit_on_no_signal_options: Sequence[bool] = (False, True),
    fee_rate: float = 0.001,
    close_final_position: bool = True,
    initial_cash: float = 1.0,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Minimal fast parameter sweep for the trading wrapper.

    Why sweep at all:
    - Stop-loss can change risk profile a lot.
    - If we allow "no-signal", we must decide whether we stay in the trade or exit.

    Output:
    - results_table: one row per config (metrics only, easy to sort)
    - best_run: full backtest dict for the best row (for plotting)
    """
    rows: List[Dict[str, object]] = []
    best_run: Optional[Dict[str, object]] = None
    best_score = -1e18

    combos = [(float(sl), bool(ex)) for sl in stop_losses for ex in exit_on_no_signal_options]
    total = len(combos)

    for i, (sl, ex) in enumerate(combos, start=1):
        cfg = TradingConfig(
            fee_rate=float(fee_rate),
            stop_loss_pct=float(sl),
            exit_on_no_signal=bool(ex),
            close_final_position=bool(close_final_position),
            initial_cash=float(initial_cash),
        )
        run = backtest_long_flat_signals(df=df, signal_col=signal_col, cfg=cfg, verbose=False)
        m = dict(run["metrics"])
        m["stop_loss_pct"] = float(sl)
        m["exit_on_no_signal"] = bool(ex)
        rows.append(m)

        # We rank by ulcer-adjusted return by default (return / ulcer).
        # This avoids picking a "lucky" high-return curve with terrible drawdowns.
        score = float(m.get("ulcer_adjusted_return", float("nan")))
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_run = run

        if verbose:
            left = total - i
            print(
                f"[sweep {i}/{total}] left={left} stop={sl:.3f} exit_no_sig={ex} "
                f"| ret={m['return_pct']:.3%} UI={m['ulcer_index']:.2f} trades={m['num_trades']} win={m['winrate']:.1%} "
                f"| BEST score={best_score:.3g}"
            )

    results = pd.DataFrame(rows).sort_values(["ulcer_adjusted_return", "return_pct"], ascending=[False, False]).reset_index(drop=True)
    return {"results_table": results, "best_run": best_run}


def plot_backtest_price_and_equity(
    run: Dict[str, object],
    title: str,
    vis_cfg: VisualizationConfig,
    price_col: str = "close",
) -> None:
    """
    Plot price + predicted zones + trade markers + equity curve.

    We keep it as 2 stacked panels because it is the clearest layout:
    - top: what the market did + what the model predicted (visual "sanity check")
    - bottom: what the strategy equity did vs buy-and-hold
    """
    df_eq = run["equity_df"].copy()
    entry_idx = list(run.get("entry_idx", []))
    exit_idx = list(run.get("exit_idx", []))
    m = dict(run.get("metrics", {}))
    cfg = run.get("config", TradingConfig())

    dates = pd.to_datetime(df_eq["date"])
    price = df_eq[price_col].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax_p, ax_e = axes

    _style_figure(fig, vis_cfg, title)
    _style_axis(ax_p, vis_cfg, with_grid=True, grid_alpha=0.30)
    _style_axis(ax_e, vis_cfg, with_grid=True, grid_alpha=0.30)

    # --- TOP: price + direction background + trades ---
    ax_p.plot(dates, price, color=vis_cfg.price_color, linewidth=2.0, label="Price")

    y_min = float(np.nanmin(price)) if len(price) else 0.0
    y_max = float(np.nanmax(price)) if len(price) else 1.0

    # Shade background by signal (UP=green, DOWN=red). NaN -> no shading.
    sig_int = pd.Series(df_eq["signal"]).fillna(-1).astype(int).to_numpy()
    # If strategy uses short=-1 encoding, render it with DOWN color.
    sig_int = np.where(sig_int < 0, 0, sig_int)
    _draw_direction_background(
        ax=ax_p,
        dates=dates,
        labels=sig_int,
        y_min=y_min,
        y_max=y_max,
        up_color=vis_cfg.up_color,
        down_color=vis_cfg.down_color,
        alpha=0.16,
    )

    # Split bands (train/val/test) if present.
    _draw_split_bands(ax_p, df_eq, vis_cfg)

    # Entry/exit markers. On flip bars (exit+entry same index) draw one marker.
    entry_set = set(int(i) for i in entry_idx)
    exit_set = set(int(i) for i in exit_idx)
    flip_idx = sorted(entry_set & exit_set)
    entry_only = sorted(entry_set - exit_set)
    exit_only = sorted(exit_set - entry_set)

    if entry_only:
        ax_p.scatter(
            dates.iloc[entry_only],
            price.iloc[entry_only],
            marker="^",
            s=80,
            color=vis_cfg.up_color,
            edgecolor="#0b1220",
            linewidth=0.8,
            label="Entry",
            zorder=5,
        )
    if exit_only:
        ax_p.scatter(
            dates.iloc[exit_only],
            price.iloc[exit_only],
            marker="v",
            s=80,
            color=vis_cfg.down_color,
            edgecolor="#0b1220",
            linewidth=0.8,
            label="Exit",
            zorder=5,
        )
    if flip_idx:
        ax_p.scatter(
            dates.iloc[flip_idx],
            price.iloc[flip_idx],
            marker="D",
            s=62,
            color="#ffd166",
            edgecolor="#0b1220",
            linewidth=0.8,
            label="Flip",
            zorder=6,
        )

    # Force plain USD ticks (no scientific notation/hidden offset text on dark theme).
    ax_p.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax_p.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax_p.set_ylabel("Price (USD)", color=vis_cfg.text_color)
    _style_legend(ax_p, vis_cfg, loc="upper left")

    # --- BOTTOM: equity vs hold ---
    ax_e.plot(dates, df_eq["equity"], color="#ffd166", linewidth=2.0, label="Strategy equity")
    ax_e.plot(dates, df_eq["hold_equity"], color=vis_cfg.muted_text, linewidth=1.8, linestyle="--", label="Buy & Hold")
    ax_e.set_ylabel("Equity", color=vis_cfg.text_color)
    _style_legend(ax_e, vis_cfg, loc="upper left")

    # Add a readable metric summary in the bottom title.
    ax_e.set_title(
        (
            f"ret={m.get('return_pct', float('nan')):.2%}  "
            f"hold={m.get('hold_return_pct', float('nan')):.2%}  "
            f"win={m.get('winrate', float('nan')):.1%}  "
            f"trades={m.get('num_trades', 0)}  "
            f"UI={m.get('ulcer_index', float('nan')):.2f}  "
            f"maxDD={m.get('max_drawdown_pct', float('nan')):.1f}%  "
            f"fee={getattr(cfg, 'fee_rate', 0.0):.3%}  stop={getattr(cfg, 'stop_loss_pct', 0.0):.1%}"
        ),
        color=vis_cfg.text_color,
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
