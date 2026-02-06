"""
Long/short trading sanity-check helpers for Moon-cycle research notebooks.

This module mirrors the output shape of `backtest_long_flat_signals` but allows
SHORT positions when the signal says DOWN.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .trading_utils import TradingConfig, _ulcer_index_from_equity


def _normalize_signal_col(sig: pd.Series) -> np.ndarray:
    """
    Normalize raw signal values to {-1, 0, 1, NaN}.

    Accepted inputs:
    - -1 / 0 / 1 numeric values
    - booleans (True->1, False->0)
    - NaN / None for no-signal
    """
    s = pd.Series(sig).copy()
    s = s.replace({True: 1.0, False: 0.0})
    s = pd.to_numeric(s, errors="coerce")

    out = np.full(len(s), np.nan, dtype=float)
    arr = s.to_numpy(dtype=float)

    out[arr >= 0.5] = 1.0
    out[arr <= -0.5] = -1.0
    out[(arr > -0.5) & (arr < 0.5)] = 0.0
    return out


def _signal_to_position(sig: float, current_pos: int, exit_on_no_signal: bool) -> int:
    """
    Convert signal value into desired position.

    Mapping:
    - +1.0 -> LONG
    - -1.0 -> SHORT
    -  0.0 -> FLAT
    - NaN  -> keep current position OR flat (if exit_on_no_signal=True)
    """
    if np.isnan(sig):
        return 0 if exit_on_no_signal else int(current_pos)

    if sig >= 0.5:
        return 1
    if sig <= -0.5:
        return -1
    return 0


def backtest_long_short_signals(
    df: pd.DataFrame,
    signal_col: str,
    cfg: TradingConfig = TradingConfig(),
    date_col: str = "date",
    price_col: str = "close",
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Simulate close-to-close long/short/flat strategy.

    Trading model:
    - position in {-1, 0, +1}
    - equity evolves by close-to-close returns while holding position
    - fee is charged whenever position changes (close and/or open leg)
    - optional stop-loss closes open position at current close
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

    equity_now = float(cfg.initial_cash)
    pos = 0  # -1 short, 0 flat, +1 long

    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None
    entry_equity: Optional[float] = None
    entry_side: Optional[str] = None

    trades: List[Dict[str, object]] = []
    equity: List[float] = []
    pos_list: List[int] = []
    entry_marks: List[int] = []
    exit_marks: List[int] = []

    for i, (dt, px, sig) in enumerate(zip(dates, prices, signals)):
        px = float(px)

        # 1) Mark-to-market for the period [i-1, i] with previous position.
        if i > 0:
            prev_px = float(prices[i - 1])
            ret = (px / prev_px) - 1.0 if prev_px > 0 else 0.0
            equity_now = float(equity_now * (1.0 + float(pos) * ret))

        # 2) Optional stop-loss check at today's close.
        stopped_today = False
        if pos != 0 and stop_pct > 0.0 and entry_price is not None:
            hit_long_stop = pos > 0 and px <= float(entry_price) * (1.0 - stop_pct)
            hit_short_stop = pos < 0 and px >= float(entry_price) * (1.0 + stop_pct)
            if hit_long_stop or hit_short_stop:
                equity_now *= (1.0 - fee)
                exit_marks.append(i)

                if entry_date is not None and entry_equity is not None:
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "entry_price": float(entry_price),
                            "entry_side": entry_side,
                            "exit_date": pd.to_datetime(dt),
                            "exit_price": px,
                            "exit_reason": "stop_loss",
                            "return_pct": float(equity_now / entry_equity - 1.0),
                            "holding_days": int((pd.to_datetime(dt) - entry_date).days),
                        }
                    )

                pos = 0
                entry_price = None
                entry_date = None
                entry_equity = None
                entry_side = None
                stopped_today = True

        # 3) Signal -> desired position (for next close-to-close interval).
        desired = _signal_to_position(sig=sig, current_pos=pos, exit_on_no_signal=bool(cfg.exit_on_no_signal))
        if stopped_today:
            desired = 0

        # 4) Rebalance position at current close, with fee per leg.
        if desired != pos:
            if pos != 0:
                # Close existing leg.
                equity_now *= (1.0 - fee)
                exit_marks.append(i)

                if entry_date is not None and entry_equity is not None:
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "entry_price": float(entry_price) if entry_price is not None else float("nan"),
                            "entry_side": entry_side,
                            "exit_date": pd.to_datetime(dt),
                            "exit_price": px,
                            "exit_reason": "signal_flip_or_flat",
                            "return_pct": float(equity_now / entry_equity - 1.0),
                            "holding_days": int((pd.to_datetime(dt) - entry_date).days),
                        }
                    )

            if desired != 0:
                # Open new leg.
                equity_now *= (1.0 - fee)
                entry_marks.append(i)
                entry_price = px
                entry_date = pd.to_datetime(dt)
                entry_equity = float(equity_now)
                entry_side = "LONG" if desired > 0 else "SHORT"
            else:
                entry_price = None
                entry_date = None
                entry_equity = None
                entry_side = None

            pos = int(desired)

        equity.append(float(equity_now))
        pos_list.append(int(pos))

        if verbose and (i % 200 == 0 or i == len(prices) - 1):
            print(f"[bt-ls] {i+1}/{len(prices)} date={pd.to_datetime(dt).date()} equity={equity_now:.4f} pos={pos}")

    # Optional final close so final equity is in cash-equivalent.
    if cfg.close_final_position and pos != 0 and len(equity) > 0:
        equity_now *= (1.0 - fee)
        exit_marks.append(len(prices) - 1)
        equity[-1] = float(equity_now)
        pos_list[-1] = 0

        if entry_date is not None and entry_equity is not None:
            trades.append(
                {
                    "entry_date": entry_date,
                    "entry_price": float(entry_price) if entry_price is not None else float("nan"),
                    "entry_side": entry_side,
                    "exit_date": pd.to_datetime(dates[-1]),
                    "exit_price": float(prices[-1]),
                    "exit_reason": "end_close",
                    "return_pct": float(equity_now / entry_equity - 1.0),
                    "holding_days": int((pd.to_datetime(dates[-1]) - entry_date).days),
                }
            )

    equity_arr = np.asarray(equity, dtype=float)

    # Benchmark: buy-and-hold with same fee assumption.
    if len(prices) == 0:
        hold_equity = np.asarray([], dtype=float)
    else:
        hold_equity = np.empty(len(prices), dtype=float)
        hold_equity[0] = float(cfg.initial_cash) * (1.0 - fee)
        for i in range(1, len(prices)):
            prev_px = float(prices[i - 1])
            ret = (float(prices[i]) / prev_px) - 1.0 if prev_px > 0 else 0.0
            hold_equity[i] = hold_equity[i - 1] * (1.0 + ret)
        if cfg.close_final_position:
            hold_equity[-1] *= (1.0 - fee)

    peak = np.maximum.accumulate(equity_arr) if equity_arr.size else np.asarray([], dtype=float)
    dd = (peak - equity_arr) / np.where(peak > 0.0, peak, np.nan) * 100.0 if equity_arr.size else np.asarray([], dtype=float)
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)

    trades_df = pd.DataFrame(trades)
    winrate = float((trades_df["return_pct"] > 0.0).mean()) if not trades_df.empty else float("nan")

    final_eq = float(equity_arr[-1]) if equity_arr.size else float("nan")
    hold_final = float(hold_equity[-1]) if hold_equity.size else float("nan")

    metrics = {
        "final_equity": final_eq,
        "hold_final_equity": hold_final,
        "return_pct": float(final_eq / cfg.initial_cash - 1.0) if np.isfinite(final_eq) else float("nan"),
        "hold_return_pct": float(hold_final / cfg.initial_cash - 1.0) if np.isfinite(hold_final) else float("nan"),
        "excess_return_pct": float(final_eq / hold_final - 1.0)
        if np.isfinite(final_eq) and np.isfinite(hold_final) and hold_final != 0.0
        else float("nan"),
        "num_trades": int(len(trades_df)),
        "winrate": float(winrate),
        "max_drawdown_pct": float(np.max(dd)) if dd.size else float("nan"),
        "ulcer_index": float(_ulcer_index_from_equity(equity_arr)),
        "exposure_pct": float(np.mean(np.abs(pos_list)) * 100.0) if len(pos_list) else float("nan"),
        "exposure_long_pct": float(np.mean(np.asarray(pos_list) > 0) * 100.0) if len(pos_list) else float("nan"),
        "exposure_short_pct": float(np.mean(np.asarray(pos_list) < 0) * 100.0) if len(pos_list) else float("nan"),
    }
    metrics["ulcer_adjusted_return"] = (
        float(metrics["return_pct"] / max(metrics["ulcer_index"], 1e-12))
        if np.isfinite(metrics["ulcer_index"])
        else float("nan")
    )

    out = df_bt.copy()
    out["signal"] = signals
    out["position"] = pos_list
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
