"""
Hourly live refresh for market availability and forecast cache.

Behavior:
- Run once on service startup, then every N seconds (default: 1 hour).
- If market data is lagging behind current UTC date, trigger market update.
- If new market day appears, rebuild forecast cache.
- Always check live BTC price from CoinGecko and, if moved >= threshold
  (default: 3%), quickly rebase simulated forecast prices.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from production_dev.cache_service import (
    FORECAST_DAYS,
    init_memory_cache,
    load_cached_predictions,
    save_predictions_to_cache,
)
from production_dev.generate_cache import generate_forecast_predictions
from production_dev.market_update import update_full_market_data
from production_dev.predictor import BtcAstroPredictor
from production_dev.coingecko_client import fetch_current_btc_price_usd
from src.models.xgb import XGBBaseline

from RESEARCH.model_training import check_cuda_available
from RESEARCH.numba_utils import check_numba_available, warmup_jit


PROJECT_ROOT = Path(__file__).parent.parent
MARKET_DAILY_PATH = PROJECT_ROOT / "data" / "market" / "processed" / "BTC_full_market_daily.parquet"


@dataclass
class RefreshResult:
    """
    Snapshot of one refresh cycle.
    """

    checked_at_utc: str
    latest_market_date: Optional[str]
    is_future_gap: bool
    market_status: str
    live_price: Optional[float]
    price_change_ratio: float
    action: str
    reason: str


class HourlyRefreshService:
    """
    Background service that keeps forecast cache operationally fresh.
    """

    def __init__(
        self,
        predictor_factory: Callable[[], BtcAstroPredictor],
        interval_seconds: int = 3600,
        price_change_threshold: float = 0.03,
    ):
        self._predictor_factory = predictor_factory
        self.interval_seconds = max(60, int(interval_seconds))
        self.price_change_threshold = max(0.0, float(price_change_threshold))

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._run_lock = asyncio.Lock()

        self.last_live_price: Optional[float] = None
        self.last_result: Optional[RefreshResult] = None
        self.runtime_device: str = "cpu"
        self.cuda_available: bool = False
        self.numba_available: bool = False
        self._acceleration_ready: bool = False

    async def start(self) -> None:
        """
        Start background loop and run one immediate check.
        """
        if self._task is not None and not self._task.done():
            return

        self._stop_event.clear()
        await asyncio.to_thread(self._prepare_acceleration)
        await self.run_once(trigger="startup")
        self._task = asyncio.create_task(self._run_loop(), name="hourly-forecast-refresh")
        print(
            f"✅ Hourly refresh started: interval={self.interval_seconds}s "
            f"threshold={self.price_change_threshold:.1%} device={self.runtime_device}"
        )

    async def stop(self) -> None:
        """
        Stop background loop gracefully.
        """
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await self._task
        finally:
            self._task = None

    async def run_once(self, trigger: str = "manual") -> RefreshResult:
        """
        Execute a single refresh cycle.
        """
        async with self._run_lock:
            result = await asyncio.to_thread(self._run_once_sync, trigger)
            self.last_result = result
            return result

    async def _run_loop(self) -> None:
        """
        Periodic loop: run checks every configured interval.
        """
        while not self._stop_event.is_set():
            try:
                await self.run_once(trigger="timer")
            except Exception as e:
                print(f"⚠️ Hourly refresh failed: {e}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    def _run_once_sync(self, trigger: str) -> RefreshResult:
        now = datetime.utcnow()
        action = "none"
        reason = "no_change"
        market_status = "not_checked"

        latest_market_date = self._load_latest_market_date()
        is_future_gap = bool(latest_market_date and now.date() > latest_market_date)

        if is_future_gap:
            print(
                f"⏳ Market lag detected: latest={latest_market_date.isoformat()} "
                f"today={now.date().isoformat()}. Checking new data availability..."
            )
            try:
                update_result = update_full_market_data(progress=False, verbose=True)
                market_status = str(update_result.get("status", "unknown"))
                latest_market_date = self._load_latest_market_date()
                if market_status == "updated":
                    action = "forecast_rebuild"
                    reason = "new_market_data"
                    print("📥 New market day detected. Rebuilding forecast cache...")
                else:
                    print("ℹ️ No new closed daily row yet from market source.")
            except Exception as e:
                market_status = f"error:{e}"
                print(f"⚠️ Market update check failed: {e}")
        else:
            market_status = "up_to_date"

        live_price = self._fetch_live_price()
        price_change_ratio = self._calc_price_change_ratio(live_price)

        if action == "forecast_rebuild":
            self._rebuild_forecast_cache(start_price=live_price)
        else:
            if price_change_ratio >= self.price_change_threshold:
                action = "forecast_rebuild"
                reason = (
                    f"price_move_{price_change_ratio:.2%}"
                    if trigger != "startup"
                    else f"startup_price_drift_{price_change_ratio:.2%}"
                )
                print(
                    f"💡 Live price moved {price_change_ratio:.2%} "
                    f"(threshold {self.price_change_threshold:.2%}). Regenerating forecast..."
                )
                self._rebuild_forecast_cache(start_price=live_price)
            elif trigger == "startup" and self._forecast_cache_empty():
                action = "forecast_rebuild"
                reason = "startup_missing_cache"
                print("📦 Forecast cache missing on startup. Building it now...")
                self._rebuild_forecast_cache(start_price=live_price)
            elif live_price is not None and live_price > 0:
                # Keep today's value in cache synchronized every hour.
                if self._sync_today_anchor(live_price):
                    action = "forecast_anchor_update"
                    reason = "hourly_live_price_sync"

        if live_price is not None and live_price > 0:
            self.last_live_price = float(live_price)

        result = RefreshResult(
            checked_at_utc=now.isoformat(timespec="seconds") + "Z",
            latest_market_date=latest_market_date.isoformat() if latest_market_date else None,
            is_future_gap=is_future_gap,
            market_status=market_status,
            live_price=live_price,
            price_change_ratio=float(price_change_ratio),
            action=action,
            reason=reason,
        )
        print(
            f"🕒 Refresh done: action={result.action} reason={result.reason} "
            f"market={result.market_status} latest={result.latest_market_date}"
        )
        return result

    def _prepare_acceleration(self) -> None:
        """
        Detect and warm up available acceleration backends.
        """
        if self._acceleration_ready:
            return

        self.cuda_available, device = check_cuda_available()
        self.runtime_device = "cuda" if self.cuda_available else device
        print(f"⚙️ XGBoost runtime device: {self.runtime_device}")

        self.numba_available = check_numba_available()
        if self.numba_available:
            if os.getenv("LIVE_REFRESH_WARMUP_NUMBA", "1").strip().lower() not in {"0", "false", "no"}:
                try:
                    warmup_jit()
                    print("⚙️ Numba JIT warmed up")
                except Exception as e:
                    print(f"⚠️ Numba warmup failed: {e}")
        else:
            print("⚙️ Numba not available, using Python fallback")

        self._acceleration_ready = True

    def _configure_predictor_runtime(self, predictor: BtcAstroPredictor) -> None:
        """
        Prefer CUDA for XGBoost inference if available.
        """
        if not self.cuda_available or predictor.model is None:
            return

        model = predictor.model
        if isinstance(model, XGBBaseline):
            if getattr(model, "constant_class", None) is not None:
                return
            try:
                model.device = "cuda"
                model.model.set_params(device="cuda")
                model.model.get_booster().set_param({"device": "cuda"})
                predictor.config["runtime_device"] = "cuda"
            except Exception as e:
                print(f"⚠️ Failed to enable CUDA for XGBBaseline: {e}")
            return

        try:
            if hasattr(model, "set_params"):
                model.set_params(device="cuda")
            if hasattr(model, "get_booster"):
                model.get_booster().set_param({"device": "cuda"})
            predictor.config["runtime_device"] = "cuda"
        except Exception as e:
            print(f"⚠️ Failed to enable CUDA for model: {e}")

    def _load_latest_market_date(self) -> Optional[date]:
        if not MARKET_DAILY_PATH.exists():
            return None
        try:
            df = pd.read_parquet(MARKET_DAILY_PATH, columns=["date"])
            if df.empty:
                return None
            return pd.to_datetime(df["date"]).max().date()
        except Exception:
            return None

    def _fetch_live_price(self) -> Optional[float]:
        try:
            # Strict live source: if CoinGecko is unavailable, return None.
            # Do not inject synthetic fallback values into hourly drift logic.
            price = float(fetch_current_btc_price_usd(timeout=10))
            if price <= 0:
                return None
            return price
        except Exception as e:
            print(f"⚠️ Live price fetch failed: {e}")
            return None

    def _forecast_cache_empty(self) -> bool:
        df = load_cached_predictions("forecast")
        return df is None or len(df) == 0

    def _calc_price_change_ratio(self, live_price: Optional[float]) -> float:
        if live_price is None or live_price <= 0:
            return 0.0

        ref_price = self.last_live_price
        if ref_price is None:
            ref_price = self._forecast_anchor_price()

        if ref_price is None or ref_price <= 0:
            return 0.0

        return abs(float(live_price) - float(ref_price)) / float(ref_price)

    def _forecast_anchor_price(self) -> Optional[float]:
        df = load_cached_predictions("forecast")
        if df is None or len(df) == 0:
            return None

        if "simulated_price" not in df.columns:
            return None

        rows = df.sort_values("date").reset_index(drop=True)
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        rows["simulated_price"] = pd.to_numeric(rows["simulated_price"], errors="coerce")

        if rows["simulated_price"].notna().sum() == 0:
            return None

        today = date.today()
        today_row = rows[(rows["date"] == today) & rows["simulated_price"].notna()]
        if len(today_row) > 0:
            return float(today_row.iloc[0]["simulated_price"])

        future = rows[(rows["date"] >= today) & rows["simulated_price"].notna()]
        if len(future) > 0:
            return float(future.iloc[0]["simulated_price"])

        # Fallback: if all forecast rows are in the past, use the latest known point.
        latest = rows[rows["simulated_price"].notna()]
        if len(latest) > 0:
            return float(latest.iloc[-1]["simulated_price"])
        return None

    def _resolve_forecast_start_date(self) -> date:
        """
        Forecast starts from max(backtest_end+1, tomorrow) to avoid past rows.
        """
        tomorrow = date.today() + timedelta(days=1)
        backtest_df = load_cached_predictions("backtest")
        if backtest_df is None or len(backtest_df) == 0:
            return tomorrow

        try:
            backtest_end = pd.to_datetime(backtest_df["date"]).max().date()
            return max(tomorrow, backtest_end + timedelta(days=1))
        except Exception:
            return tomorrow

    def _attach_today_anchor(
        self,
        forecast_rows: list[Dict],
        start_price: Optional[float],
    ) -> list[Dict]:
        """
        Ensure forecast includes an explicit "today" point with live price.
        """
        if start_price is None or start_price <= 0:
            return forecast_rows

        today_str = date.today().isoformat()

        # Remove any pre-existing current-day row to avoid duplicates.
        cleaned = [r for r in forecast_rows if str(r.get("date")) != today_str]

        if cleaned:
            first = cleaned[0]
            direction_code = int(first.get("direction_code", 0))
            direction = "UP" if direction_code == 1 else "DOWN"
            confidence = float(first.get("confidence", 0.5))
        else:
            direction_code = 0
            direction = "DOWN"
            confidence = 0.5

        anchor = {
            "date": today_str,
            "direction": direction,
            "direction_code": direction_code,
            "confidence": confidence,
            "simulated_price": round(float(start_price), 2),
            "is_now_anchor": True,
        }
        return [anchor, *cleaned]

    def _rebuild_forecast_cache(self, start_price: Optional[float]) -> None:
        predictor = self._predictor_factory()
        self._configure_predictor_runtime(predictor)
        forecast_start_date = self._resolve_forecast_start_date()
        forecast = generate_forecast_predictions(
            predictor=predictor,
            days=FORECAST_DAYS,
            forecast_start_date=forecast_start_date,
            start_price=start_price,
        )
        forecast = self._attach_today_anchor(forecast, start_price)
        if save_predictions_to_cache(forecast, "forecast"):
            init_memory_cache()

    def _sync_today_anchor(self, start_price: float) -> bool:
        """
        Lightweight hourly update:
        - keep a "today" row in forecast cache,
        - rebuild only simulated future price path from current live price
          (without expensive astro recomputation).
        """
        if start_price <= 0:
            return False

        forecast_df = load_cached_predictions("forecast")
        if forecast_df is None or len(forecast_df) == 0:
            self._rebuild_forecast_cache(start_price=start_price)
            return True

        rows = forecast_df.sort_values("date").reset_index(drop=True)
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        today = date.today()

        old_today = rows[rows["date"] == today]
        if len(old_today) > 0:
            old_price = pd.to_numeric(old_today.iloc[0].get("simulated_price"), errors="coerce")
            if pd.notna(old_price):
                diff = abs(float(old_price) - float(start_price))
                if diff < 1e-6:
                    return False

        # Keep only future days and regenerate their simulated prices from live point.
        future = rows[rows["date"] > today].copy()
        predictions = []
        for _, row in future.iterrows():
            direction = str(row.get("direction", "DOWN")).upper()
            if "direction_code" in row.index and pd.notna(row.get("direction_code")):
                direction_code = int(row["direction_code"])
            else:
                direction_code = 1 if direction == "UP" else 0

            predictions.append(
                {
                    "date": row["date"].isoformat(),
                    "direction": "UP" if direction_code == 1 else "DOWN",
                    "direction_code": direction_code,
                    "confidence": float(row.get("confidence", 0.5)),
                }
            )

        predictor = self._predictor_factory()
        self._configure_predictor_runtime(predictor)
        future_path = predictor.generate_price_path(
            predictions=predictions,
            start_price=float(start_price),
            seed=42,
        )
        merged = self._attach_today_anchor(future_path, start_price)

        if save_predictions_to_cache(merged, "forecast"):
            init_memory_cache()
            return True
        return False
