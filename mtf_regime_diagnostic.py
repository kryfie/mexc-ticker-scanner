#!/usr/bin/env python3
"""
MTF Pullback Quality v8 Robust Multi-Window Scanner for MEXC USDT-M perpetual futures.

Robust multi-window diagnostic model based on Pine MTF Pullback Quality Strategy v6 — 3R.
    M15 -> fresh directional regime with age/slope/stretch limits
    M5  -> controlled pullback
    M1  -> micro-touch + breakout quality trigger
    SL  -> M5 swing, dynamically widened by ATR / price floor
    TP  -> minimum 3R from simulated fill price

Outputs:
    v8_robust_scorecard.csv
    v8_robust_window_summary.csv
    v8_robust_ticker_summary.csv
    v8_robust_trades.csv
    v8_robust_rejections.csv
    v8_robust_equity.csv
    v8_robust_open.csv
    v8_robust_primary_leave_one_ticker_out.csv
    v8_robust_errors.csv
    v8_robust_config.json

The scanner uses only public MEXC market endpoints. No API key is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


BASE_URLS = (
    "https://api.mexc.com",
    "https://contract.mexc.com",
)

INTERVAL_SECONDS = {
    "Min1": 60,
    "Min5": 5 * 60,
    "Min15": 15 * 60,
}

PRIMARY_VARIANT_NAME = "C_RVD_CORE"


@dataclass(frozen=True)
class ContractMeta:
    ticker: str
    max_leverage: Optional[int] = None
    base_coin: Optional[str] = None
    quote_coin: Optional[str] = None
    contract_size: Optional[float] = None
    min_vol: Optional[float] = None
    price_scale: Optional[int] = None
    amount_scale: Optional[int] = None
    price_unit: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    api_allowed: bool = True
    state: Optional[int] = None


@dataclass
class StrategyConfig:
    days: int = 30
    warmup_days: int = 7
    reward_risk: float = 3.0
    commission_percent: float = 0.06
    slippage_ticks: int = 1
    conservative_same_candle: bool = True
    exit_on_regime_loss: bool = False
    cooldown_after_exit_bars: int = 10
    one_trade_per_pullback: bool = True
    allow_long: bool = True
    allow_short: bool = True

    # M15 regime — Pine v6
    regime_fast_len: int = 50
    regime_slow_len: int = 200
    regime_atr_len: int = 14
    regime_slope_len: int = 6
    regime_er_len: int = 20
    min_regime_er: float = 0.22
    min_regime_slope_atr: float = 0.08
    max_regime_slope_atr: float = 0.80
    min_regime_stretch_atr: float = 0.00
    max_regime_stretch_atr: float = 2.50
    min_regime_strength: float = 60.0
    max_regime_age_bars: int = 20

    # M5 controlled pullback — Pine v6
    pull_fast_len: int = 12
    pull_mid_len: int = 50
    pull_atr_len: int = 14
    pull_atr_avg_len: int = 20
    pullback_valid_bars: int = 3
    zone_buffer_atr: float = 0.15
    max_mid_overshoot_atr: float = 0.25
    min_pullback_atr_ratio: float = 0.60
    max_pullback_atr_ratio: float = 1.00
    require_counter_candle: bool = True
    pullback_impulse_lookback: int = 12

    # M1 quality trigger — Pine v6
    trigger_fast_len: int = 12
    trigger_mid_len: int = 50
    trigger_atr_len: int = 14
    micro_touch_lookback: int = 3
    breakout_lookback: int = 2
    range_avg_len: int = 20
    min_range_ratio: float = 0.80
    max_range_ratio: float = 1.60
    min_body_ratio: float = 0.45
    max_body_ratio: float = 0.85
    min_close_location: float = 0.70
    max_entry_stretch_atr: float = 0.50

    # Risk / 3R target — Pine v6
    m5_swing_lookback: int = 10
    stop_buffer_m5_atr: float = 0.10
    min_stop_distance_m5_atr: float = 0.75
    max_stop_distance_m5_atr: float = 2.50
    min_stop_distance_pct: float = 0.50
    max_stop_distance_pct: float = 3.00
    min_target_distance_pct: float = 1.50
    max_target_distance_pct: float = 9.00


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    allow_long: bool = True
    allow_short: bool = True
    max_m5_volume_ratio: Optional[float] = None
    max_m5_pullback_depth_atr: Optional[float] = None
    min_m1_range_ratio: Optional[float] = None
    min_m1_distance_ema50_atr: Optional[float] = None
    protect_trigger_r: Optional[float] = None
    protect_lock_r: float = 0.0


@dataclass
class OpenTrade:
    variant: str
    ticker: str
    direction: str
    entry_index: int
    entry_time: int
    signal_close: float
    entry_price: float
    initial_stop_price: float
    active_stop_price: float
    target_price: float
    initial_risk: float
    entry_snapshot: Dict[str, Any]

    mfe_price: float = 0.0
    mae_price: float = 0.0
    bars_to_mfe: int = 0
    bars_to_mae: int = 0
    bars_held: int = 0
    protection_triggered: bool = False
    protection_trigger_bar: Optional[int] = None
    protection_lock_r: float = 0.0


class MexcPublicClient:
    def __init__(
        self,
        retries: int = 4,
        backoff: float = 1.5,
        request_sleep: float = 0.12,
    ) -> None:
        self.retries = retries
        self.backoff = backoff
        self.request_sleep = request_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "mtf-pullback-quality-v8-robust-windows/1.0",
                "Accept": "application/json",
            }
        )

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        last_error: Optional[Exception] = None

        for base_url in BASE_URLS:
            url = f"{base_url}{path}"

            for attempt in range(1, self.retries + 1):
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    payload = response.json()

                    if isinstance(payload, dict) and payload.get("success") is False:
                        raise RuntimeError(f"MEXC error response: {payload}")

                    time.sleep(self.request_sleep)
                    return payload

                except Exception as exc:  # noqa: BLE001 - scanner should retry network failures
                    last_error = exc
                    print(
                        f"MEXC request failed {attempt}/{self.retries}: "
                        f"{url} params={params} error={exc}",
                        file=sys.stderr,
                    )
                    if attempt < self.retries:
                        time.sleep(self.backoff * attempt)

        if last_error is None:
            raise RuntimeError(f"Unknown MEXC request error: {path}")
        raise last_error


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_int(value: Any) -> Optional[int]:
    number = to_float(value)
    return int(number) if number is not None else None


def normalize_contract_rows(payload: dict) -> List[dict]:
    data = payload.get("data", [])

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        if "symbol" in data:
            return [data]
        for key in ("resultList", "list", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]

    return []


def get_contract_meta(client: MexcPublicClient) -> Dict[str, ContractMeta]:
    errors: List[str] = []
    rows: List[dict] = []

    for path in ("/api/v1/contract/detail", "/api/v1/contract/detail/country"):
        try:
            rows = normalize_contract_rows(client.get(path))
            if rows:
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")

    if not rows:
        raise RuntimeError("Could not load contract metadata. " + " | ".join(errors))

    meta: Dict[str, ContractMeta] = {}

    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue

        price_scale = to_int(row.get("priceScale"))
        price_unit = to_float(row.get("priceUnit"))
        if price_unit is None and price_scale is not None:
            price_unit = 10.0 ** (-price_scale)

        max_leverage_candidates = [
            to_int(row.get("countryConfigContractMaxLeverage")),
            to_int(row.get("maxLeverage")),
            to_int(row.get("max_leverage")),
        ]
        max_leverage_candidates = [
            value for value in max_leverage_candidates if value is not None and value > 0
        ]

        meta[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=max(max_leverage_candidates) if max_leverage_candidates else None,
            base_coin=row.get("baseCoin"),
            quote_coin=row.get("quoteCoin"),
            contract_size=to_float(row.get("contractSize")),
            min_vol=to_float(row.get("minVol")),
            price_scale=price_scale,
            amount_scale=to_int(row.get("amountScale")),
            price_unit=price_unit,
            taker_fee_rate=to_float(row.get("takerFeeRate")),
            api_allowed=bool(row.get("apiAllowed", True)),
            state=to_int(row.get("state")),
        )

    return meta


def get_contract_symbols(
    meta_map: Dict[str, ContractMeta],
    limit: Optional[int] = None,
    min_max_leverage: Optional[int] = None,
) -> List[str]:
    symbols: List[str] = []

    for symbol, meta in meta_map.items():
        if meta.quote_coin != "USDT":
            continue
        if meta.state not in (None, 0):
            continue
        if not meta.api_allowed:
            continue
        if min_max_leverage is not None:
            if meta.max_leverage is None or meta.max_leverage < min_max_leverage:
                continue
        symbols.append(symbol)

    symbols = sorted(set(symbols))
    return symbols[:limit] if limit else symbols


def fetch_klines(
    client: MexcPublicClient,
    symbol: str,
    interval: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")

    step_seconds = INTERVAL_SECONDS[interval]
    max_points = 2000
    chunk_span = step_seconds * (max_points - 1)
    cursor = start
    frames: List[pd.DataFrame] = []

    while cursor <= end:
        chunk_end = min(end, cursor + chunk_span)
        payload = client.get(
            f"/api/v1/contract/kline/{symbol}",
            {
                "interval": interval,
                "start": cursor,
                "end": chunk_end,
            },
        )
        data = payload.get("data", {}) if isinstance(payload, dict) else {}

        if not isinstance(data, dict) or not data.get("time"):
            cursor = chunk_end + step_seconds
            continue

        count = len(data["time"])
        frame = pd.DataFrame(
            {
                "time": data["time"],
                "open": data.get("open", [np.nan] * count),
                "high": data.get("high", [np.nan] * count),
                "low": data.get("low", [np.nan] * count),
                "close": data.get("close", [np.nan] * count),
                "vol": data.get("vol", [np.nan] * count),
            }
        )
        frames.append(frame)
        cursor = chunk_end + step_seconds

    columns = ["time", "open", "high", "low", "close", "vol"]
    if not frames:
        return pd.DataFrame(columns=columns)

    output = pd.concat(frames, ignore_index=True)
    output = output.drop_duplicates(subset=["time"], keep="last").sort_values("time")

    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["time", "open", "high", "low", "close"])
    output["time"] = output["time"].astype("int64")
    return output.reset_index(drop=True)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)

    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) < length:
        return pd.Series(result, index=series.index)

    first = int(valid_positions[0])
    seed_end = first + length
    if seed_end > len(values) or not np.isfinite(values[first:seed_end]).all():
        return pd.Series(result, index=series.index)

    result[seed_end - 1] = float(np.mean(values[first:seed_end]))
    alpha = 1.0 / float(length)

    for index in range(seed_end, len(values)):
        value = values[index]
        previous = result[index - 1]
        if not np.isfinite(value) or not np.isfinite(previous):
            result[index] = np.nan
        else:
            result[index] = previous + alpha * (value - previous)

    return pd.Series(result, index=series.index)


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    output = ranges.max(axis=1, skipna=True)
    output.iloc[0] = frame["high"].iloc[0] - frame["low"].iloc[0]
    return output


def atr(frame: pd.DataFrame, length: int) -> pd.Series:
    return rma(true_range(frame), length)


def efficiency_ratio(close: pd.Series, length: int) -> pd.Series:
    net_move = (close - close.shift(length)).abs()
    gross_move = close.diff().abs().rolling(length, min_periods=length).sum()
    return (net_move / gross_move.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)


def bars_since(condition: pd.Series) -> pd.Series:
    result = np.full(len(condition), np.nan)
    last_true: Optional[int] = None
    values = condition.fillna(False).to_numpy(dtype=bool)

    for index, is_true in enumerate(values):
        if is_true:
            last_true = index
            result[index] = 0.0
        elif last_true is not None:
            result[index] = float(index - last_true)

    return pd.Series(result, index=condition.index)


def consecutive_true_count(condition: pd.Series) -> pd.Series:
    result = np.zeros(len(condition), dtype=float)
    count = 0
    for index, value in enumerate(condition.fillna(False).to_numpy(dtype=bool)):
        count = count + 1 if value else 0
        result[index] = float(count)
    return pd.Series(result, index=condition.index)


def directional_move_since_start(
    close: pd.Series,
    active: pd.Series,
    direction: int,
    atr_value: pd.Series,
) -> pd.Series:
    output = np.full(len(close), np.nan)
    start_price: Optional[float] = None

    for index in range(len(close)):
        is_active = bool(active.iloc[index]) if pd.notna(active.iloc[index]) else False
        if not is_active:
            start_price = None
            continue

        if start_price is None:
            start_price = float(close.iloc[index])

        current_atr = float(atr_value.iloc[index]) if pd.notna(atr_value.iloc[index]) else np.nan
        if np.isfinite(current_atr) and current_atr > 0:
            output[index] = direction * (float(close.iloc[index]) - start_price) / current_atr

    return pd.Series(output, index=close.index)


def resample_ohlcv(frame_m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame_m1.empty:
        return frame_m1.copy()

    frame = frame_m1.copy()
    frame["dt"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("dt")

    rule = f"{minutes}min"
    aggregated = frame.resample(rule, label="left", closed="left", origin="epoch").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "vol": "sum",
            "time": "count",
        }
    )
    aggregated = aggregated.rename(columns={"time": "source_bar_count"})
    aggregated = aggregated[aggregated["source_bar_count"] >= minutes]
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated["time"] = (aggregated.index.view("int64") // 10**9).astype("int64")
    return aggregated[["time", "open", "high", "low", "close", "vol"]]


def prepare_m15(frame: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output["time"], unit="s", utc=True)

    output["m15_ema_fast"] = ema(output["close"], cfg.regime_fast_len)
    output["m15_ema_slow"] = ema(output["close"], cfg.regime_slow_len)
    output["m15_atr"] = atr(output, cfg.regime_atr_len)
    output["m15_er"] = efficiency_ratio(output["close"], cfg.regime_er_len)

    output["m15_slope_fast_atr"] = (
        output["m15_ema_fast"] - output["m15_ema_fast"].shift(cfg.regime_slope_len)
    ) / output["m15_atr"]
    output["m15_slope_slow_atr"] = (
        output["m15_ema_slow"] - output["m15_ema_slow"].shift(cfg.regime_slope_len)
    ) / output["m15_atr"]
    output["m15_stretch_atr"] = (
        output["close"] - output["m15_ema_fast"]
    ).abs() / output["m15_atr"]

    long_alignment = (output["m15_ema_fast"] > output["m15_ema_slow"]).astype(float) * 25.0
    long_location = (output["close"] > output["m15_ema_fast"]).astype(float) * 15.0
    long_fast_slope = (
        output["m15_slope_fast_atr"]
        / max(cfg.min_regime_slope_atr * 2.0, 1e-12)
    ).clip(0.0, 1.0) * 20.0
    long_slow_slope = (
        output["m15_slope_slow_atr"]
        / max(cfg.min_regime_slope_atr, 1e-12)
    ).clip(0.0, 1.0) * 15.0

    short_alignment = (output["m15_ema_fast"] < output["m15_ema_slow"]).astype(float) * 25.0
    short_location = (output["close"] < output["m15_ema_fast"]).astype(float) * 15.0
    short_fast_slope = (
        -output["m15_slope_fast_atr"]
        / max(cfg.min_regime_slope_atr * 2.0, 1e-12)
    ).clip(0.0, 1.0) * 20.0
    short_slow_slope = (
        -output["m15_slope_slow_atr"]
        / max(cfg.min_regime_slope_atr, 1e-12)
    ).clip(0.0, 1.0) * 15.0

    efficiency_score = (
        (output["m15_er"] - cfg.min_regime_er)
        / max(1.0 - cfg.min_regime_er, 1e-12)
    ).clip(0.0, 1.0) * 15.0
    stretch_score = (
        1.0 - (output["m15_stretch_atr"] / cfg.max_regime_stretch_atr).clip(0.0, 1.0)
    ) * 10.0

    output["m15_long_strength"] = (
        long_alignment + long_location + long_fast_slope + long_slow_slope
        + efficiency_score + stretch_score
    )
    output["m15_short_strength"] = (
        short_alignment + short_location + short_fast_slope + short_slow_slope
        + efficiency_score + stretch_score
    )

    output["m15_long_base"] = (
        (output["m15_ema_fast"] > output["m15_ema_slow"])
        & (output["close"] > output["m15_ema_fast"])
        & (output["m15_slope_fast_atr"] >= cfg.min_regime_slope_atr)
        & (output["m15_slope_fast_atr"] <= cfg.max_regime_slope_atr)
        & (output["m15_er"] >= cfg.min_regime_er)
        & (output["m15_stretch_atr"] >= cfg.min_regime_stretch_atr)
        & (output["m15_stretch_atr"] <= cfg.max_regime_stretch_atr)
        & (output["m15_long_strength"] >= cfg.min_regime_strength)
    )
    output["m15_short_base"] = (
        (output["m15_ema_fast"] < output["m15_ema_slow"])
        & (output["close"] < output["m15_ema_fast"])
        & (output["m15_slope_fast_atr"] <= -cfg.min_regime_slope_atr)
        & (output["m15_slope_fast_atr"] >= -cfg.max_regime_slope_atr)
        & (output["m15_er"] >= cfg.min_regime_er)
        & (output["m15_stretch_atr"] >= cfg.min_regime_stretch_atr)
        & (output["m15_stretch_atr"] <= cfg.max_regime_stretch_atr)
        & (output["m15_short_strength"] >= cfg.min_regime_strength)
    )

    # Equivalent to Pine: ready ? ta.barssince(not ready) : 0.
    output["m15_long_regime_age"] = consecutive_true_count(output["m15_long_base"])
    output["m15_short_regime_age"] = consecutive_true_count(output["m15_short_base"])

    output["m15_long_regime"] = (
        output["m15_long_base"]
        & output["m15_long_regime_age"].between(1, cfg.max_regime_age_bars)
    )
    output["m15_short_regime"] = (
        output["m15_short_base"]
        & output["m15_short_regime_age"].between(1, cfg.max_regime_age_bars)
    )

    output["m15_long_move_since_start_atr"] = directional_move_since_start(
        output["close"], output["m15_long_base"], 1, output["m15_atr"]
    )
    output["m15_short_move_since_start_atr"] = directional_move_since_start(
        output["close"], output["m15_short_base"], -1, output["m15_atr"]
    )
    return output


def prepare_m5(frame: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output["time"], unit="s", utc=True)

    output["m5_ema_fast"] = ema(output["close"], cfg.pull_fast_len)
    output["m5_ema_mid"] = ema(output["close"], cfg.pull_mid_len)
    output["m5_atr"] = atr(output, cfg.pull_atr_len)
    output["m5_atr_avg"] = output["m5_atr"].rolling(
        cfg.pull_atr_avg_len, min_periods=cfg.pull_atr_avg_len
    ).mean()
    output["m5_volume_avg"] = output["vol"].rolling(
        cfg.pull_atr_avg_len, min_periods=cfg.pull_atr_avg_len
    ).mean()
    output["m5_range_ratio"] = output["m5_atr"] / output["m5_atr_avg"]

    long_touch = (
        (output["m5_ema_fast"] > output["m5_ema_mid"])
        & (output["low"] <= output["m5_ema_fast"] + output["m5_atr"] * cfg.zone_buffer_atr)
        & (output["low"] >= output["m5_ema_mid"] - output["m5_atr"] * cfg.max_mid_overshoot_atr)
        & (output["m5_range_ratio"] >= cfg.min_pullback_atr_ratio)
        & (output["m5_range_ratio"] <= cfg.max_pullback_atr_ratio)
    )
    short_touch = (
        (output["m5_ema_fast"] < output["m5_ema_mid"])
        & (output["high"] >= output["m5_ema_fast"] - output["m5_atr"] * cfg.zone_buffer_atr)
        & (output["high"] <= output["m5_ema_mid"] + output["m5_atr"] * cfg.max_mid_overshoot_atr)
        & (output["m5_range_ratio"] >= cfg.min_pullback_atr_ratio)
        & (output["m5_range_ratio"] <= cfg.max_pullback_atr_ratio)
    )

    if cfg.require_counter_candle:
        long_touch &= output["close"] < output["open"]
        short_touch &= output["close"] > output["open"]

    output["m5_long_touch"] = long_touch
    output["m5_short_touch"] = short_touch
    output["m5_long_pullback_age"] = bars_since(long_touch)
    output["m5_short_pullback_age"] = bars_since(short_touch)

    output["m5_long_pullback"] = (
        output["m5_long_pullback_age"].between(0, cfg.pullback_valid_bars)
        & (output["m5_ema_fast"] > output["m5_ema_mid"])
        & (
            output["close"]
            >= output["m5_ema_mid"] - output["m5_atr"] * cfg.max_mid_overshoot_atr
        )
    )
    output["m5_short_pullback"] = (
        output["m5_short_pullback_age"].between(0, cfg.pullback_valid_bars)
        & (output["m5_ema_fast"] < output["m5_ema_mid"])
        & (
            output["close"]
            <= output["m5_ema_mid"] + output["m5_atr"] * cfg.max_mid_overshoot_atr
        )
    )

    # Pine valuewhen(touch, time, 0).
    output["m5_long_pullback_id"] = output["time"].where(long_touch).ffill()
    output["m5_short_pullback_id"] = output["time"].where(short_touch).ffill()

    output["m5_swing_low"] = output["low"].rolling(
        cfg.m5_swing_lookback, min_periods=cfg.m5_swing_lookback
    ).min()
    output["m5_swing_high"] = output["high"].rolling(
        cfg.m5_swing_lookback, min_periods=cfg.m5_swing_lookback
    ).max()

    impulse_high = output["high"].rolling(
        cfg.pullback_impulse_lookback, min_periods=cfg.pullback_impulse_lookback
    ).max()
    impulse_low = output["low"].rolling(
        cfg.pullback_impulse_lookback, min_periods=cfg.pullback_impulse_lookback
    ).min()

    output["m5_long_pullback_depth_atr"] = (impulse_high - output["low"]) / output["m5_atr"]
    output["m5_short_pullback_depth_atr"] = (output["high"] - impulse_low) / output["m5_atr"]
    output["m5_long_pullback_depth_pct"] = (
        (impulse_high - output["low"]) / impulse_high.replace(0.0, np.nan) * 100.0
    )
    output["m5_short_pullback_depth_pct"] = (
        (output["high"] - impulse_low) / impulse_low.replace(0.0, np.nan) * 100.0
    )
    output["m5_volume_ratio"] = output["vol"] / output["m5_volume_avg"]
    output["m5_distance_ema12_atr"] = (
        output["close"] - output["m5_ema_fast"]
    ) / output["m5_atr"]
    output["m5_distance_ema50_atr"] = (
        output["close"] - output["m5_ema_mid"]
    ) / output["m5_atr"]
    return output


def prepare_m1(frame: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output["time"], unit="s", utc=True)

    output["m1_ema_fast"] = ema(output["close"], cfg.trigger_fast_len)
    output["m1_ema_mid"] = ema(output["close"], cfg.trigger_mid_len)
    output["m1_atr"] = atr(output, cfg.trigger_atr_len)
    output["m1_true_range"] = true_range(output)
    output["m1_avg_range"] = output["m1_true_range"].rolling(
        cfg.range_avg_len, min_periods=cfg.range_avg_len
    ).mean()

    candle_range = (output["high"] - output["low"]).clip(lower=1e-12)
    output["m1_body_ratio"] = (output["close"] - output["open"]).abs() / candle_range
    output["m1_close_location_long"] = (output["close"] - output["low"]) / candle_range
    output["m1_close_location_short"] = (output["high"] - output["close"]) / candle_range

    output["m1_long_break_level"] = output["high"].shift(1).rolling(
        cfg.breakout_lookback, min_periods=cfg.breakout_lookback
    ).max()
    output["m1_short_break_level"] = output["low"].shift(1).rolling(
        cfg.breakout_lookback, min_periods=cfg.breakout_lookback
    ).min()

    output["m1_range_ratio"] = output["m1_true_range"] / output["m1_avg_range"]
    output["m1_fast_ema_slope_atr"] = (
        output["m1_ema_fast"] - output["m1_ema_fast"].shift(1)
    ) / output["m1_atr"]
    output["m1_distance_ema12_atr"] = (
        output["close"] - output["m1_ema_fast"]
    ) / output["m1_atr"]
    output["m1_distance_ema50_atr"] = (
        output["close"] - output["m1_ema_mid"]
    ) / output["m1_atr"]
    output["m1_long_breakout_distance_atr"] = (
        output["close"] - output["m1_long_break_level"]
    ) / output["m1_atr"]
    output["m1_short_breakout_distance_atr"] = (
        output["m1_short_break_level"] - output["close"]
    ) / output["m1_atr"]

    long_touch = output["low"] <= output["m1_ema_fast"]
    short_touch = output["high"] >= output["m1_ema_fast"]
    output["m1_recent_long_touch"] = (
        long_touch.shift(1).rolling(
            cfg.micro_touch_lookback, min_periods=cfg.micro_touch_lookback
        ).max().fillna(0.0) > 0.5
    )
    output["m1_recent_short_touch"] = (
        short_touch.shift(1).rolling(
            cfg.micro_touch_lookback, min_periods=cfg.micro_touch_lookback
        ).max().fillna(0.0) > 0.5
    )

    candle_quality = (
        (output["m1_range_ratio"] >= cfg.min_range_ratio)
        & (output["m1_range_ratio"] <= cfg.max_range_ratio)
        & (output["m1_body_ratio"] >= cfg.min_body_ratio)
        & (output["m1_body_ratio"] <= cfg.max_body_ratio)
    )

    output["m1_long_trigger"] = (
        output["m1_recent_long_touch"]
        & (output["close"] > output["m1_long_break_level"])
        & (output["close"] > output["open"])
        & (output["close"] > output["m1_ema_fast"])
        & (output["m1_ema_fast"] > output["m1_ema_fast"].shift(1))
        & (output["m1_ema_fast"] >= output["m1_ema_mid"])
        & candle_quality
        & (output["m1_close_location_long"] >= cfg.min_close_location)
        & (output["m1_distance_ema12_atr"] <= cfg.max_entry_stretch_atr)
    )
    output["m1_short_trigger"] = (
        output["m1_recent_short_touch"]
        & (output["close"] < output["m1_short_break_level"])
        & (output["close"] < output["open"])
        & (output["close"] < output["m1_ema_fast"])
        & (output["m1_ema_fast"] < output["m1_ema_fast"].shift(1))
        & (output["m1_ema_fast"] <= output["m1_ema_mid"])
        & candle_quality
        & (output["m1_close_location_short"] >= cfg.min_close_location)
        & (-output["m1_distance_ema12_atr"] <= cfg.max_entry_stretch_atr)
    )
    return output


def map_previous_closed_timeframe(
    source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    prefix_columns: Sequence[str],
) -> pd.DataFrame:
    shifted = source.loc[:, list(prefix_columns)].shift(1)
    return shifted.reindex(target_index, method="ffill")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_value(value: Any) -> Optional[float]:
    return float(value) if finite(value) else None


def safe_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def build_snapshot(row: pd.Series, direction: str, stop_distance_atr: float) -> Dict[str, Any]:
    is_long = direction == "LONG"
    side = "long" if is_long else "short"

    return {
        "m15_regime_strength": safe_value(row[f"m15_{side}_strength"]),
        "m15_efficiency_ratio": safe_value(row["m15_er"]),
        "m15_ema50_slope_atr": safe_value(row["m15_slope_fast_atr"]),
        "m15_ema200_slope_atr": safe_value(row["m15_slope_slow_atr"]),
        "m15_price_distance_ema50_atr": safe_value(row["m15_stretch_atr"]),
        "m15_trend_age_bars": safe_value(row[f"m15_{side}_regime_age"]),
        "m15_move_since_regime_start_atr": safe_value(
            row[f"m15_{side}_move_since_start_atr"]
        ),
        "m5_pullback_age": safe_value(row[f"m5_{side}_pullback_age"]),
        "m5_pullback_depth_atr": safe_value(row[f"m5_{side}_pullback_depth_atr"]),
        "m5_pullback_depth_percent": safe_value(row[f"m5_{side}_pullback_depth_pct"]),
        "m5_pullback_candles": (
            int(row[f"m5_{side}_pullback_age"]) + 1
            if finite(row[f"m5_{side}_pullback_age"])
            else None
        ),
        "m5_pullback_range_ratio": safe_value(row["m5_range_ratio"]),
        "m5_pullback_volume_ratio": safe_value(row["m5_volume_ratio"]),
        "m5_distance_ema12_atr": safe_value(row["m5_distance_ema12_atr"]),
        "m5_distance_ema50_atr": safe_value(row["m5_distance_ema50_atr"]),
        "m5_swing_distance_atr": safe_value(stop_distance_atr),
        "m1_trigger_range_ratio": safe_value(row["m1_range_ratio"]),
        "m1_body_ratio": safe_value(row["m1_body_ratio"]),
        "m1_breakout_distance_atr": safe_value(
            row[
                "m1_long_breakout_distance_atr"
                if is_long
                else "m1_short_breakout_distance_atr"
            ]
        ),
        "m1_entry_distance_ema12_atr": safe_value(
            row["m1_distance_ema12_atr"]
            if is_long
            else -row["m1_distance_ema12_atr"]
        ),
        "m1_entry_distance_ema50_atr": safe_value(
            row["m1_distance_ema50_atr"]
            if is_long
            else -row["m1_distance_ema50_atr"]
        ),
        "m1_fast_ema_slope_atr": safe_value(
            row["m1_fast_ema_slope_atr"]
            if is_long
            else -row["m1_fast_ema_slope_atr"]
        ),
    }


def rejection_reason(
    row: pd.Series,
    direction: str,
    risk_atr: float,
    cfg: StrategyConfig,
    flat: bool,
    cooldown_ready: bool,
) -> Optional[str]:
    is_long = direction == "LONG"

    regime_ok = safe_bool(row["m15_long_regime"] if is_long else row["m15_short_regime"])
    opposite_regime = safe_bool(
        row["m15_short_regime"] if is_long else row["m15_long_regime"]
    )
    pullback_ok = safe_bool(row["m5_long_pullback"] if is_long else row["m5_short_pullback"])

    if not flat:
        return "POSITION_ALREADY_OPEN"
    if not cooldown_ready:
        return "COOLDOWN"
    if not regime_ok:
        return "M15_REGIME_MISSING"
    if opposite_regime:
        return "OPPOSITE_M15_REGIME"
    if not pullback_ok:
        return "M5_PULLBACK_MISSING"
    if not finite(risk_atr):
        return "RISK_NOT_AVAILABLE"
    if risk_atr < cfg.min_stop_distance_m5_atr:
        return "SL_TOO_NARROW"
    if risk_atr > cfg.max_stop_distance_m5_atr:
        return "SL_TOO_WIDE"
    return None


def pnl_metrics(
    direction: str,
    entry_price: float,
    exit_price: float,
    initial_risk: float,
    commission_rate: float,
) -> Tuple[float, float, float, float]:
    gross_per_unit = (
        exit_price - entry_price
        if direction == "LONG"
        else entry_price - exit_price
    )
    fees_per_unit = commission_rate * (abs(entry_price) + abs(exit_price))
    net_per_unit = gross_per_unit - fees_per_unit

    gross_pct = gross_per_unit / entry_price * 100.0
    net_pct = net_per_unit / entry_price * 100.0
    gross_r = gross_per_unit / initial_risk
    net_r = net_per_unit / initial_risk
    return gross_pct, net_pct, gross_r, net_r


def close_trade_record(
    trade: OpenTrade,
    exit_time: int,
    exit_price: float,
    exit_reason: str,
    commission_rate: float,
) -> Dict[str, Any]:
    gross_pct, net_pct, gross_r, net_r = pnl_metrics(
        trade.direction,
        trade.entry_price,
        exit_price,
        trade.initial_risk,
        commission_rate,
    )

    mfe_r = trade.mfe_price / trade.initial_risk
    mae_r = trade.mae_price / trade.initial_risk

    record: Dict[str, Any] = {
        "variant": trade.variant,
        "ticker": trade.ticker,
        "direction": trade.direction,
        "entry_time_utc": pd.to_datetime(trade.entry_time, unit="s", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="s", utc=True),
        "entry_price": trade.entry_price,
        "signal_close": trade.signal_close,
        "initial_stop_price": trade.initial_stop_price,
        "final_stop_price": trade.active_stop_price,
        "target_price": trade.target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_r_gross": gross_r,
        "pnl_r_net": net_r,
        "pnl_pct_gross": gross_pct,
        "pnl_pct_net": net_pct,
        "duration_minutes": max(0, int((exit_time - trade.entry_time) / 60)),
        "bars_held": trade.bars_held,
        "mfe_price": trade.mfe_price,
        "mae_price": trade.mae_price,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "bars_to_mfe": trade.bars_to_mfe,
        "bars_to_mae": trade.bars_to_mae,
        "protection_triggered": trade.protection_triggered,
        "protection_trigger_bar": trade.protection_trigger_bar,
        "protection_lock_r": trade.protection_lock_r,
        "reached_0_5r": mfe_r >= 0.5,
        "reached_1r": mfe_r >= 1.0,
        "reached_1_5r": mfe_r >= 1.5,
        "reached_2r": mfe_r >= 2.0,
        "reached_3r": mfe_r >= 3.0,
    }
    record.update(trade.entry_snapshot)
    return record


def max_streak(values: Iterable[bool], target: bool) -> int:
    maximum = 0
    current = 0
    for value in values:
        if bool(value) == target:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum




def build_variants() -> List[VariantSpec]:
    """Test the primary Range+Volume+Depth hypothesis and close alternatives."""
    return [
        VariantSpec(
            name="A_BASE_V6_3R",
            description="Exact v6 baseline: pure SL or 3R TP.",
        ),
        VariantSpec(
            name="B_RANGE_DEPTH",
            description="M1 range >= 1.10 and M5 pullback depth <= 1.70 ATR.",
            min_m1_range_ratio=1.10,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name=PRIMARY_VARIANT_NAME,
            description="PRIMARY: M1 range >= 1.10, M5 volume <= 0.90 and depth <= 1.70 ATR.",
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="D_EMA50_RANGE_DEPTH",
            description="EMA50 distance >= 0.80 ATR plus range >= 1.10 and depth <= 1.70 ATR.",
            min_m1_distance_ema50_atr=0.80,
            min_m1_range_ratio=1.10,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="E_RVD_SHORT_ONLY",
            description="Primary Range+Volume+Depth filters, SHORT signals only.",
            allow_long=False,
            allow_short=True,
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="F_RVD_LONG_ONLY",
            description="Primary Range+Volume+Depth filters, LONG signals only.",
            allow_long=True,
            allow_short=False,
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="G_RVD_BE_AT_1_5R",
            description="Primary filters; after 1.5R move stop to entry from next M1 bar.",
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
            protect_trigger_r=1.50,
            protect_lock_r=0.00,
        ),
        VariantSpec(
            name="H_RVD_LOCK_0_5R_AT_2R",
            description="Primary filters; after 2R lock +0.5R from next M1 bar.",
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
            protect_trigger_r=2.00,
            protect_lock_r=0.50,
        ),
        VariantSpec(
            name="I_RVD_RANGE_12",
            description="Primary filters with stricter M1 trigger range >= 1.20.",
            min_m1_range_ratio=1.20,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="J_RVD_VOLUME_08",
            description="Primary filters with quieter M5 pullback volume <= 0.80.",
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.80,
            max_m5_pullback_depth_atr=1.70,
        ),
        VariantSpec(
            name="K_RVD_DEPTH_15",
            description="Primary filters with shallower M5 pullback depth <= 1.50 ATR.",
            min_m1_range_ratio=1.10,
            max_m5_volume_ratio=0.90,
            max_m5_pullback_depth_atr=1.50,
        ),
    ]


def variant_filter_reason(
    row: pd.Series,
    direction: str,
    variant: VariantSpec,
) -> Optional[str]:
    is_long = direction == "LONG"
    side = "long" if is_long else "short"

    tests = [
        ("m5_volume_ratio", variant.max_m5_volume_ratio, "FILTER_M5_VOLUME", "max"),
        (f"m5_{side}_pullback_depth_atr", variant.max_m5_pullback_depth_atr, "FILTER_M5_DEPTH", "max"),
        ("m1_range_ratio", variant.min_m1_range_ratio, "FILTER_M1_RANGE", "min"),
    ]

    for column, threshold, reason, mode in tests:
        if threshold is None:
            continue
        value = safe_value(row.get(column))
        if value is None:
            return "FILTER_DATA_MISSING"
        if mode == "max" and value > threshold:
            return reason
        if mode == "min" and value < threshold:
            return reason

    if variant.min_m1_distance_ema50_atr is not None:
        raw = safe_value(row.get("m1_distance_ema50_atr"))
        if raw is None:
            return "FILTER_DATA_MISSING"
        directional_distance = raw if is_long else -raw
        if directional_distance < variant.min_m1_distance_ema50_atr:
            return "FILTER_M1_EMA50_DISTANCE"

    return None


def apply_protection_after_bar(trade: OpenTrade, variant: VariantSpec) -> None:
    """Conservative: a newly moved stop is active from the next M1 bar."""
    if variant.protect_trigger_r is None or trade.protection_triggered:
        return

    mfe_r = trade.mfe_price / trade.initial_risk
    if mfe_r < variant.protect_trigger_r:
        return

    if trade.direction == "LONG":
        protected_stop = trade.entry_price + variant.protect_lock_r * trade.initial_risk
        if protected_stop > trade.active_stop_price:
            trade.active_stop_price = protected_stop
    else:
        protected_stop = trade.entry_price - variant.protect_lock_r * trade.initial_risk
        if protected_stop < trade.active_stop_price:
            trade.active_stop_price = protected_stop

    trade.protection_triggered = True
    trade.protection_trigger_bar = trade.bars_held
    trade.protection_lock_r = variant.protect_lock_r


def summarize_trade_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "tp": 0,
            "sl": 0,
            "be": 0,
            "lock": 0,
            "profitable_trades": 0,
            "win_rate_pct": None,
            "profit_factor_net_r": None,
            "expectancy_net_r": None,
            "net_r": 0.0,
            "max_drawdown_r": 0.0,
            "max_losing_streak": 0,
            "max_winning_streak": 0,
            "avg_mfe_r": None,
            "avg_mae_r": None,
            "avg_duration_min": None,
        }

    ordered = frame.sort_values("exit_time_utc").copy()
    ordered["cum_r"] = ordered["pnl_r_net"].cumsum()
    ordered["peak_r"] = ordered["cum_r"].cummax().clip(lower=0.0)
    ordered["dd_r"] = ordered["cum_r"] - ordered["peak_r"]
    positive = ordered.loc[ordered["pnl_r_net"] > 0, "pnl_r_net"].sum()
    negative_abs = -ordered.loc[ordered["pnl_r_net"] < 0, "pnl_r_net"].sum()
    wins = ordered["pnl_r_net"] > 0

    return {
        "trades": int(len(ordered)),
        "tp": int((ordered["exit_reason"] == "TP").sum()),
        "sl": int((ordered["exit_reason"] == "SL").sum()),
        "be": int((ordered["exit_reason"] == "BE").sum()),
        "lock": int((ordered["exit_reason"] == "LOCK").sum()),
        "profitable_trades": int(wins.sum()),
        "win_rate_pct": float(wins.mean() * 100.0),
        "profit_factor_net_r": float(positive / negative_abs) if negative_abs > 0 else None,
        "expectancy_net_r": float(ordered["pnl_r_net"].mean()),
        "net_r": float(ordered["pnl_r_net"].sum()),
        "max_drawdown_r": float(ordered["dd_r"].min()),
        "max_losing_streak": max_streak(wins, False),
        "max_winning_streak": max_streak(wins, True),
        "avg_mfe_r": float(ordered["mfe_r"].mean()),
        "avg_mae_r": float(ordered["mae_r"].mean()),
        "avg_duration_min": float(ordered["duration_minutes"].mean()),
    }


def simulate_variant(
    symbol: str,
    simulation: pd.DataFrame,
    cfg: StrategyConfig,
    variant: VariantSpec,
    price_tick: float,
) -> Tuple[List[Dict[str, Any]], Counter, List[Dict[str, Any]]]:
    commission_rate = cfg.commission_percent / 100.0
    slippage_value = cfg.slippage_ticks * price_tick
    trades: List[Dict[str, Any]] = []
    open_rows: List[Dict[str, Any]] = []
    rejection_counts: Counter = Counter()

    active_trade: Optional[OpenTrade] = None
    last_exit_index: Optional[int] = None
    last_traded_long_pullback_id: Optional[int] = None
    last_traded_short_pullback_id: Optional[int] = None

    for local_index, (_, row) in enumerate(simulation.iterrows()):
        current_time = int(row["time"])
        long_regime = safe_bool(row.get("m15_long_regime", False))
        short_regime = safe_bool(row.get("m15_short_regime", False))
        long_pullback = safe_bool(row.get("m5_long_pullback", False))
        short_pullback = safe_bool(row.get("m5_short_pullback", False))
        long_trigger = safe_bool(row.get("m1_long_trigger", False))
        short_trigger = safe_bool(row.get("m1_short_trigger", False))

        if active_trade is not None and local_index > active_trade.entry_index:
            active_trade.bars_held += 1

            if active_trade.direction == "LONG":
                favorable = max(0.0, float(row["high"]) - active_trade.entry_price)
                adverse = max(0.0, active_trade.entry_price - float(row["low"]))
                stop_hit = float(row["low"]) <= active_trade.active_stop_price
                target_hit = float(row["high"]) >= active_trade.target_price
            else:
                favorable = max(0.0, active_trade.entry_price - float(row["low"]))
                adverse = max(0.0, float(row["high"]) - active_trade.entry_price)
                stop_hit = float(row["high"]) >= active_trade.active_stop_price
                target_hit = float(row["low"]) <= active_trade.target_price

            if favorable > active_trade.mfe_price:
                active_trade.mfe_price = favorable
                active_trade.bars_to_mfe = active_trade.bars_held
            if adverse > active_trade.mae_price:
                active_trade.mae_price = adverse
                active_trade.bars_to_mae = active_trade.bars_held

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None

            if stop_hit and target_hit:
                stop_wins = cfg.conservative_same_candle
                exit_reason = None if not stop_wins else (
                    "LOCK" if active_trade.protection_triggered and active_trade.protection_lock_r > 0
                    else "BE" if active_trade.protection_triggered
                    else "SL"
                )
                if not stop_wins:
                    exit_reason = "TP"
            elif stop_hit:
                exit_reason = (
                    "LOCK" if active_trade.protection_triggered and active_trade.protection_lock_r > 0
                    else "BE" if active_trade.protection_triggered
                    else "SL"
                )
            elif target_hit:
                exit_reason = "TP"

            if exit_reason in {"SL", "BE", "LOCK"}:
                exit_price = (
                    active_trade.active_stop_price - slippage_value
                    if active_trade.direction == "LONG"
                    else active_trade.active_stop_price + slippage_value
                )
            elif exit_reason == "TP":
                exit_price = active_trade.target_price

            if exit_reason is None and cfg.exit_on_regime_loss:
                regime_alive = long_regime if active_trade.direction == "LONG" else short_regime
                if not regime_alive:
                    exit_reason = "REGIME"
                    exit_price = (
                        float(row["close"]) - slippage_value
                        if active_trade.direction == "LONG"
                        else float(row["close"]) + slippage_value
                    )

            if exit_reason is not None and exit_price is not None:
                trades.append(close_trade_record(
                    active_trade, current_time, exit_price, exit_reason, commission_rate
                ))
                last_exit_index = local_index
                active_trade = None
            else:
                apply_protection_after_bar(active_trade, variant)

        flat = active_trade is None
        cooldown_ready = (
            last_exit_index is None
            or local_index - last_exit_index > cfg.cooldown_after_exit_bars
        )

        for direction, trigger in (("LONG", long_trigger), ("SHORT", short_trigger)):
            is_long = direction == "LONG"
            if is_long and not variant.allow_long:
                continue
            if not is_long and not variant.allow_short:
                continue
            if is_long and not cfg.allow_long:
                continue
            if not is_long and not cfg.allow_short:
                continue
            if not trigger:
                continue

            regime_ok = long_regime if is_long else short_regime
            opposite_regime = short_regime if is_long else long_regime
            pullback_ok = long_pullback if is_long else short_pullback
            pullback_id_value = safe_value(row.get(
                "m5_long_pullback_id" if is_long else "m5_short_pullback_id"
            ))
            pullback_id = int(pullback_id_value) if pullback_id_value is not None else None
            last_traded_id = (
                last_traded_long_pullback_id if is_long else last_traded_short_pullback_id
            )
            pullback_unused = (
                not cfg.one_trade_per_pullback
                or (pullback_id is not None and pullback_id != last_traded_id)
            )

            reason: Optional[str] = None
            if not flat:
                reason = "POSITION_ALREADY_OPEN"
            elif not cooldown_ready:
                reason = "COOLDOWN"
            elif not regime_ok:
                reason = "M15_REGIME_MISSING"
            elif opposite_regime:
                reason = "OPPOSITE_M15_REGIME"
            elif not pullback_ok:
                reason = "M5_PULLBACK_MISSING"
            elif not pullback_unused:
                reason = "PULLBACK_ALREADY_TRADED"
            else:
                reason = variant_filter_reason(row, direction, variant)

            signal_close = float(row["close"])
            m5_atr = safe_value(row.get("m5_atr"))
            swing_price = safe_value(row.get("m5_swing_low" if is_long else "m5_swing_high"))
            raw_stop = np.nan
            stop_candidate = np.nan
            risk_atr = np.nan
            risk_pct = np.nan
            target_pct = np.nan
            stop_expanded = False

            if reason is None and (m5_atr is None or m5_atr <= 0 or swing_price is None):
                reason = "RISK_NOT_AVAILABLE"

            if reason is None and m5_atr is not None and swing_price is not None:
                minimum_stop_distance = max(
                    signal_close * cfg.min_stop_distance_pct / 100.0,
                    m5_atr * cfg.min_stop_distance_m5_atr,
                )
                if is_long:
                    raw_stop = swing_price - m5_atr * cfg.stop_buffer_m5_atr
                    stop_floor = signal_close - minimum_stop_distance
                    stop_candidate = min(raw_stop, stop_floor)
                    risk = signal_close - stop_candidate
                    stop_expanded = stop_candidate < raw_stop
                else:
                    raw_stop = swing_price + m5_atr * cfg.stop_buffer_m5_atr
                    stop_floor = signal_close + minimum_stop_distance
                    stop_candidate = max(raw_stop, stop_floor)
                    risk = stop_candidate - signal_close
                    stop_expanded = stop_candidate > raw_stop

                risk_atr = risk / m5_atr
                risk_pct = risk / signal_close * 100.0
                target_pct = risk_pct * cfg.reward_risk

                if risk <= price_tick:
                    reason = "NON_POSITIVE_INITIAL_RISK"
                elif risk_atr > cfg.max_stop_distance_m5_atr:
                    reason = "SL_TOO_WIDE_ATR"
                elif risk_pct > cfg.max_stop_distance_pct:
                    reason = "SL_TOO_WIDE_PCT"
                elif target_pct < cfg.min_target_distance_pct:
                    reason = "TARGET_TOO_CLOSE"
                elif target_pct > cfg.max_target_distance_pct:
                    reason = "TARGET_TOO_FAR"

            if reason is not None:
                rejection_counts[(direction, reason)] += 1
                continue

            entry_price = signal_close + slippage_value if is_long else signal_close - slippage_value
            initial_risk = (
                entry_price - float(stop_candidate)
                if is_long else float(stop_candidate) - entry_price
            )
            target_price = (
                entry_price + initial_risk * cfg.reward_risk
                if is_long else entry_price - initial_risk * cfg.reward_risk
            )
            snapshot = build_snapshot(row, direction, float(risk_atr))
            snapshot.update({
                "variant_description": variant.description,
                "pullback_id": pullback_id,
                "raw_stop_price": float(raw_stop),
                "dynamic_stop_expanded": stop_expanded,
                "signal_risk_pct": float(risk_pct),
                "signal_target_pct": float(target_pct),
                "reward_risk": cfg.reward_risk,
                "variant_max_m5_volume_ratio": variant.max_m5_volume_ratio,
                "variant_max_m5_pullback_depth_atr": variant.max_m5_pullback_depth_atr,
                "variant_min_m1_range_ratio": variant.min_m1_range_ratio,
                "variant_min_m1_distance_ema50_atr": variant.min_m1_distance_ema50_atr,
                "variant_protect_trigger_r": variant.protect_trigger_r,
                "variant_protect_lock_r": variant.protect_lock_r,
            })
            active_trade = OpenTrade(
                variant=variant.name,
                ticker=symbol,
                direction=direction,
                entry_index=local_index,
                entry_time=current_time,
                signal_close=signal_close,
                entry_price=entry_price,
                initial_stop_price=float(stop_candidate),
                active_stop_price=float(stop_candidate),
                target_price=float(target_price),
                initial_risk=initial_risk,
                entry_snapshot=snapshot,
            )

            if is_long:
                last_traded_long_pullback_id = pullback_id
            else:
                last_traded_short_pullback_id = pullback_id
            break

    if active_trade is not None:
        open_row = {
            "variant": variant.name,
            "ticker": symbol,
            "direction": active_trade.direction,
            "entry_time_utc": pd.to_datetime(active_trade.entry_time, unit="s", utc=True),
            "entry_price": active_trade.entry_price,
            "initial_stop_price": active_trade.initial_stop_price,
            "active_stop_price": active_trade.active_stop_price,
            "target_price": active_trade.target_price,
            "mfe_r_so_far": active_trade.mfe_price / active_trade.initial_risk,
            "mae_r_so_far": active_trade.mae_price / active_trade.initial_risk,
            "protection_triggered": active_trade.protection_triggered,
        }
        open_row.update(active_trade.entry_snapshot)
        open_rows.append(open_row)

    return trades, rejection_counts, open_rows


def prepare_symbol_data(
    client: MexcPublicClient,
    symbol: str,
    cfg: StrategyConfig,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    warmup_start = start_time - cfg.warmup_days * 24 * 60 * 60
    raw_m1 = fetch_klines(client, symbol, "Min1", warmup_start, end_time)
    if len(raw_m1) < cfg.regime_slow_len * 15:
        raise RuntimeError(f"Insufficient M1 history: {len(raw_m1)} rows")

    m1 = prepare_m1(raw_m1, cfg)
    m5 = prepare_m5(resample_ohlcv(raw_m1, 5), cfg)
    m15 = prepare_m15(resample_ohlcv(raw_m1, 15), cfg)
    m15_columns = [column for column in m15.columns if column.startswith("m15_")]
    m5_columns = [column for column in m5.columns if column.startswith("m5_")]
    m1 = m1.join(map_previous_closed_timeframe(m15, m1.index, m15_columns))
    m1 = m1.join(map_previous_closed_timeframe(m5, m1.index, m5_columns))
    return m1[(m1["time"] >= start_time) & (m1["time"] <= end_time)].sort_index()


def build_test_windows(
    end_time: int,
    window_days: int,
    window_count: int,
) -> List[Dict[str, Any]]:
    """Create contiguous independent windows, oldest first."""
    seconds = window_days * 24 * 60 * 60
    total_start = end_time - seconds * window_count
    windows: List[Dict[str, Any]] = []
    for index in range(window_count):
        start = total_start + index * seconds
        end = start + seconds
        if index == window_count - 1:
            end = end_time
        windows.append({
            "window": f"W{index + 1}_{window_days}D",
            "window_index": index + 1,
            "start_time": start,
            "end_time": end,
            "start_time_utc": pd.to_datetime(start, unit="s", utc=True),
            "end_time_utc": pd.to_datetime(end, unit="s", utc=True),
        })
    return windows


def window_summary_rows(
    trades: pd.DataFrame,
    variants: List[VariantSpec],
    windows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    window_names = [window["window"] for window in windows]

    for variant in variants:
        variant_frame = (
            trades.loc[trades["variant"] == variant.name].copy()
            if not trades.empty else trades
        )
        for window_name in ["ALL_WINDOWS", *window_names]:
            window_frame = (
                variant_frame
                if window_name == "ALL_WINDOWS"
                else variant_frame.loc[variant_frame["window"] == window_name]
            )
            for scope in ("ALL", "LONG", "SHORT"):
                scoped = (
                    window_frame
                    if scope == "ALL"
                    else window_frame.loc[window_frame["direction"] == scope]
                )
                row = {
                    "variant": variant.name,
                    "description": variant.description,
                    "is_primary_candidate": variant.name == PRIMARY_VARIANT_NAME,
                    "window": window_name,
                    "scope": scope,
                }
                row.update(summarize_trade_frame(scoped))
                rows.append(row)
    return rows


def _metric_value(lookup: pd.DataFrame, window: str, column: str) -> Any:
    return lookup.at[window, column] if window in lookup.index else None


def build_robust_scorecard(
    window_summary: pd.DataFrame,
    windows: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    window_names = [window["window"] for window in windows]

    for (variant, scope), group in window_summary.groupby(["variant", "scope"], dropna=False):
        lookup = group.set_index("window")
        total_trades = int(_metric_value(lookup, "ALL_WINDOWS", "trades") or 0)
        total_expectancy = _metric_value(lookup, "ALL_WINDOWS", "expectancy_net_r")
        total_pf = _metric_value(lookup, "ALL_WINDOWS", "profit_factor_net_r")
        total_net = float(_metric_value(lookup, "ALL_WINDOWS", "net_r") or 0.0)
        total_dd = float(_metric_value(lookup, "ALL_WINDOWS", "max_drawdown_r") or 0.0)

        expectations: List[float] = []
        nets: List[float] = []
        trades_per_window: List[int] = []
        description = str(group["description"].iloc[0]) if "description" in group.columns else ""
        row: Dict[str, Any] = {
            "variant": variant,
            "description": description,
            "scope": scope,
            "is_primary_candidate": variant == PRIMARY_VARIANT_NAME,
            "total_trades": total_trades,
            "total_tp": int(_metric_value(lookup, "ALL_WINDOWS", "tp") or 0),
            "total_sl": int(_metric_value(lookup, "ALL_WINDOWS", "sl") or 0),
            "total_be": int(_metric_value(lookup, "ALL_WINDOWS", "be") or 0),
            "total_lock": int(_metric_value(lookup, "ALL_WINDOWS", "lock") or 0),
            "total_pf": total_pf,
            "total_expectancy_r": total_expectancy,
            "total_net_r": total_net,
            "total_drawdown_r": total_dd,
        }

        for window_name in window_names:
            trades_value = int(_metric_value(lookup, window_name, "trades") or 0)
            expectancy_value = _metric_value(lookup, window_name, "expectancy_net_r")
            net_value = float(_metric_value(lookup, window_name, "net_r") or 0.0)
            pf_value = _metric_value(lookup, window_name, "profit_factor_net_r")
            trades_per_window.append(trades_value)
            nets.append(net_value)
            if expectancy_value is not None and pd.notna(expectancy_value):
                expectations.append(float(expectancy_value))

            prefix = window_name.lower()
            row[f"{prefix}_trades"] = trades_value
            row[f"{prefix}_pf"] = pf_value
            row[f"{prefix}_expectancy_r"] = expectancy_value
            row[f"{prefix}_net_r"] = net_value

        positive_windows = sum(1 for value in nets if value > 0)
        nonnegative_windows = sum(1 for value in nets if value >= 0)
        windows_with_trades = sum(1 for value in trades_per_window if value > 0)
        min_expectancy = min(expectations) if expectations else None
        avg_expectancy = float(np.mean(expectations)) if expectations else None
        expectancy_std = float(np.std(expectations, ddof=0)) if expectations else None
        worst_net = min(nets) if nets else None
        best_net = max(nets) if nets else None

        expectancy_num = float(total_expectancy) if total_expectancy is not None and pd.notna(total_expectancy) else -99.0
        min_exp_num = float(min_expectancy) if min_expectancy is not None else -99.0
        stability_penalty = float(expectancy_std or 0.0)
        sample_bonus = math.log1p(total_trades) * 0.15
        drawdown_penalty = abs(total_dd) * 0.03
        robust_score = (
            expectancy_num
            + 1.5 * min_exp_num
            + 0.35 * positive_windows
            + sample_bonus
            - stability_penalty
            - drawdown_penalty
        )

        if windows_with_trades == len(window_names) and positive_windows == len(window_names) and total_trades >= 20 and expectancy_num > 0:
            decision = "STRONG"
        elif windows_with_trades == len(window_names) and positive_windows == len(window_names) and expectancy_num > 0:
            decision = "PROMISING_SMALL_SAMPLE"
        elif positive_windows >= max(2, len(window_names) - 1) and total_net > 0:
            decision = "MIXED_POSITIVE"
        else:
            decision = "WEAK"

        row.update({
            "windows_with_trades": windows_with_trades,
            "positive_windows": positive_windows,
            "nonnegative_windows": nonnegative_windows,
            "all_windows_positive": positive_windows == len(window_names),
            "min_window_expectancy_r": min_expectancy,
            "avg_window_expectancy_r": avg_expectancy,
            "window_expectancy_std": expectancy_std,
            "worst_window_net_r": worst_net,
            "best_window_net_r": best_net,
            "decision": decision,
            "robust_score": robust_score,
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["scope", "all_windows_positive", "positive_windows", "robust_score", "total_trades"],
            ascending=[True, False, False, False, False],
            na_position="last",
        )
    return result


def ticker_summary_rows(
    trades: pd.DataFrame,
    variants: List[VariantSpec],
    symbols: List[str],
    windows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    window_names = [window["window"] for window in windows]
    for variant in variants:
        for symbol in symbols:
            subset = (
                trades.loc[(trades["variant"] == variant.name) & (trades["ticker"] == symbol)]
                if not trades.empty else trades
            )
            for window_name in ["ALL_WINDOWS", *window_names]:
                window_subset = (
                    subset if window_name == "ALL_WINDOWS"
                    else subset.loc[subset["window"] == window_name]
                )
                for scope in ("ALL", "LONG", "SHORT"):
                    scoped = (
                        window_subset if scope == "ALL"
                        else window_subset.loc[window_subset["direction"] == scope]
                    )
                    row = {
                        "variant": variant.name,
                        "is_primary_candidate": variant.name == PRIMARY_VARIANT_NAME,
                        "ticker": symbol,
                        "window": window_name,
                        "scope": scope,
                    }
                    row.update(summarize_trade_frame(scoped))
                    rows.append(row)
    return rows


def leave_one_ticker_out_rows(
    trades: pd.DataFrame,
    symbols: List[str],
    windows: List[Dict[str, Any]],
    primary_variant: str = PRIMARY_VARIANT_NAME,
) -> List[Dict[str, Any]]:
    primary = (
        trades.loc[trades["variant"] == primary_variant].copy()
        if not trades.empty else trades
    )
    window_names = [window["window"] for window in windows]
    rows: List[Dict[str, Any]] = []

    for excluded in ["NONE", *symbols]:
        reduced = (
            primary if excluded == "NONE" or primary.empty
            else primary.loc[primary["ticker"] != excluded]
        )
        aggregate = summarize_trade_frame(reduced)
        row: Dict[str, Any] = {
            "primary_variant": primary_variant,
            "excluded_ticker": excluded,
            **aggregate,
        }
        window_nets: List[float] = []
        for window_name in window_names:
            scoped = (
                reduced.loc[reduced["window"] == window_name]
                if not reduced.empty else reduced
            )
            metrics = summarize_trade_frame(scoped)
            prefix = window_name.lower()
            row[f"{prefix}_trades"] = metrics["trades"]
            row[f"{prefix}_pf"] = metrics["profit_factor_net_r"]
            row[f"{prefix}_expectancy_r"] = metrics["expectancy_net_r"]
            row[f"{prefix}_net_r"] = metrics["net_r"]
            window_nets.append(float(metrics["net_r"]))
        row["positive_windows"] = sum(1 for value in window_nets if value > 0)
        row["all_windows_positive"] = all(value > 0 for value in window_nets)
        row["worst_window_net_r"] = min(window_nets) if window_nets else None
        rows.append(row)
    return rows


def equity_rows(trades: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if trades.empty:
        return rows
    for (variant, window), group in trades.groupby(["variant", "window"]):
        ordered = group.sort_values("exit_time_utc").copy().reset_index(drop=True)
        ordered["trade_number"] = np.arange(1, len(ordered) + 1)
        ordered["cumulative_r"] = ordered["pnl_r_net"].cumsum()
        ordered["peak_r"] = ordered["cumulative_r"].cummax().clip(lower=0.0)
        ordered["drawdown_r"] = ordered["cumulative_r"] - ordered["peak_r"]
        rows.extend(ordered[[
            "variant", "window", "exit_time_utc", "ticker", "trade_number", "direction",
            "exit_reason", "pnl_r_net", "cumulative_r", "drawdown_r"
        ]].to_dict("records"))
    return rows


def csv_or_empty(rows: List[Dict[str, Any]], columns: Sequence[str], path: Path) -> None:
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the primary Range+Volume+Depth candidate and close alternatives "
            "across independent MEXC windows."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--all", action="store_true")
    source.add_argument("--symbols", nargs="+")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--rr", type=float, default=3.0)
    parser.add_argument("--commission-percent", type=float, default=0.06)
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument(
        "--same-candle-policy",
        choices=("conservative", "optimistic"),
        default="conservative",
    )
    parser.add_argument("--request-sleep", type=float, default=0.12)
    parser.add_argument("--output-prefix", default="v8_robust")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_days <= 0:
        raise SystemExit("--window-days must be greater than zero.")
    if args.window_count <= 0:
        raise SystemExit("--window-count must be greater than zero.")
    if args.rr < 3.0:
        raise SystemExit("This test requires RR >= 3.0.")

    total_days = args.window_days * args.window_count
    cfg = StrategyConfig(
        days=total_days,
        warmup_days=args.warmup_days,
        reward_risk=args.rr,
        commission_percent=args.commission_percent,
        slippage_ticks=args.slippage_ticks,
        conservative_same_candle=args.same_candle_policy == "conservative",
        exit_on_regime_loss=False,
    )
    variants = build_variants()
    client = MexcPublicClient(request_sleep=args.request_sleep)
    print("Loading MEXC contract metadata...")
    meta_map = get_contract_meta(client)

    if args.all:
        symbols = get_contract_symbols(meta_map, limit=args.limit)
    else:
        symbols = sorted(set(args.symbols or []))
        if args.limit:
            symbols = symbols[:args.limit]
    if not symbols:
        raise SystemExit("No symbols selected.")

    now = int(time.time())
    last_bar_time = (now // 60) * 60 - 60
    end_exclusive = last_bar_time + 60
    windows = build_test_windows(end_exclusive, args.window_days, args.window_count)
    start_time = int(windows[0]["start_time"])

    all_trades: List[Dict[str, Any]] = []
    all_open: List[Dict[str, Any]] = []
    rejection_rows: List[Dict[str, Any]] = []
    scan_errors: List[Dict[str, Any]] = []

    for position, symbol in enumerate(symbols, start=1):
        print(
            f"[{position}/{len(symbols)}] preparing {symbol} once for "
            f"{len(variants)} variants x {len(windows)} windows..."
        )
        meta = meta_map.get(symbol)
        success = False
        for attempt in range(1, 4):
            try:
                prepared = prepare_symbol_data(client, symbol, cfg, start_time, last_bar_time)
                price_tick = meta.price_unit if meta and meta.price_unit else None
                if price_tick is None or price_tick <= 0:
                    price_scale = meta.price_scale if meta and meta.price_scale is not None else 8
                    price_tick = 10.0 ** (-price_scale)

                for window in windows:
                    simulation = prepared.loc[
                        (prepared["time"] >= int(window["start_time"]))
                        & (prepared["time"] < int(window["end_time"]))
                    ].copy()
                    if simulation.empty:
                        scan_errors.append({
                            "ticker": symbol,
                            "window": window["window"],
                            "error": "no prepared rows in window",
                        })
                        continue

                    for variant in variants:
                        trades, reject_counts, open_rows = simulate_variant(
                            symbol, simulation, cfg, variant, float(price_tick)
                        )
                        for trade in trades:
                            trade["window"] = window["window"]
                            trade["window_start_utc"] = window["start_time_utc"]
                            trade["window_end_utc"] = window["end_time_utc"]
                        for open_row in open_rows:
                            open_row["window"] = window["window"]
                            open_row["window_start_utc"] = window["start_time_utc"]
                            open_row["window_end_utc"] = window["end_time_utc"]
                        all_trades.extend(trades)
                        all_open.extend(open_rows)

                        for (direction, reason), count in reject_counts.items():
                            rejection_rows.append({
                                "variant": variant.name,
                                "is_primary_candidate": variant.name == PRIMARY_VARIANT_NAME,
                                "window": window["window"],
                                "ticker": symbol,
                                "direction": direction,
                                "rejection_reason": reason,
                                "count": int(count),
                            })
                success = True
                break
            except Exception as exc:
                print(f"ERROR {symbol} attempt {attempt}/3: {exc}", file=sys.stderr)
                if attempt < 3:
                    time.sleep(3 * attempt)
        if not success:
            scan_errors.append({
                "ticker": symbol,
                "window": "ALL",
                "error": "failed after 3 attempts",
            })

    trades_frame = pd.DataFrame(all_trades)
    if trades_frame.empty:
        trades_frame = pd.DataFrame(columns=[
            "variant", "window", "ticker", "direction", "entry_time_utc",
            "exit_time_utc", "exit_reason", "pnl_r_net", "mfe_r", "mae_r",
            "duration_minutes",
        ])
    summary_rows = window_summary_rows(trades_frame, variants, windows)
    summary_frame = pd.DataFrame(summary_rows)
    scorecard = build_robust_scorecard(summary_frame, windows)
    ticker_rows = ticker_summary_rows(trades_frame, variants, symbols, windows)
    leave_one_rows = leave_one_ticker_out_rows(trades_frame, symbols, windows)
    eq_rows = equity_rows(trades_frame)

    prefix_name = args.output_prefix
    outputs = {
        "trades": Path(f"{prefix_name}_trades.csv"),
        "window_summary": Path(f"{prefix_name}_window_summary.csv"),
        "scorecard": Path(f"{prefix_name}_scorecard.csv"),
        "ticker_summary": Path(f"{prefix_name}_ticker_summary.csv"),
        "rejections": Path(f"{prefix_name}_rejections.csv"),
        "equity": Path(f"{prefix_name}_equity.csv"),
        "open": Path(f"{prefix_name}_open.csv"),
        "primary_leave_one_ticker_out": Path(f"{prefix_name}_primary_leave_one_ticker_out.csv"),
        "errors": Path(f"{prefix_name}_errors.csv"),
        "config": Path(f"{prefix_name}_config.json"),
    }

    csv_or_empty(
        all_trades,
        ["variant", "window", "ticker", "direction", "entry_time_utc", "exit_time_utc"],
        outputs["trades"],
    )
    summary_frame.to_csv(outputs["window_summary"], index=False)
    scorecard.to_csv(outputs["scorecard"], index=False)
    pd.DataFrame(ticker_rows).to_csv(outputs["ticker_summary"], index=False)
    csv_or_empty(
        rejection_rows,
        ["variant", "window", "ticker", "direction", "rejection_reason", "count"],
        outputs["rejections"],
    )
    csv_or_empty(
        eq_rows,
        ["variant", "window", "exit_time_utc", "ticker", "pnl_r_net", "cumulative_r", "drawdown_r"],
        outputs["equity"],
    )
    csv_or_empty(
        all_open,
        ["variant", "window", "ticker", "direction", "entry_time_utc"],
        outputs["open"],
    )
    pd.DataFrame(leave_one_rows).to_csv(outputs["primary_leave_one_ticker_out"], index=False)
    csv_or_empty(scan_errors, ["ticker", "window", "error"], outputs["errors"])

    config_payload = {
        "scanner_version": "MTF Pullback Quality v8 robust multi-window",
        "primary_variant": PRIMARY_VARIANT_NAME,
        "strategy_config": asdict(cfg),
        "window_days": args.window_days,
        "window_count": args.window_count,
        "windows": [
            {
                "window": window["window"],
                "start_time_utc": window["start_time_utc"].isoformat(),
                "end_time_utc": window["end_time_utc"].isoformat(),
            }
            for window in windows
        ],
        "variants": [asdict(variant) for variant in variants],
        "symbols": symbols,
        "same_candle_policy": args.same_candle_policy,
        "notes": (
            "Each window is simulated independently with reset position/cooldown/pullback state. "
            "Indicators are prepared from a shared history with warmup. Protection stops become "
            "active on the next M1 bar after trigger."
        ),
    }
    outputs["config"].write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PRIMARY CANDIDATE ===")
    primary_rows = scorecard.loc[
        (scorecard["variant"] == PRIMARY_VARIANT_NAME)
        & (scorecard["scope"].isin(["ALL", "LONG", "SHORT"]))
    ]
    primary_columns = [
        "variant", "scope", "total_trades", "total_tp", "total_sl", "total_pf",
        "total_expectancy_r", "total_net_r", "positive_windows", "all_windows_positive",
        "worst_window_net_r", "decision", "robust_score",
    ]
    print(primary_rows[primary_columns].to_string(index=False) if not primary_rows.empty else "No primary trades.")

    print("\n=== ALL VARIANTS / ALL SCOPE ===")
    if scorecard.empty:
        print("No closed trades.")
    else:
        columns = [
            "variant", "total_trades", "total_tp", "total_sl", "total_be", "total_lock",
            "total_pf", "total_expectancy_r", "total_net_r", "positive_windows",
            "all_windows_positive", "worst_window_net_r", "decision", "robust_score",
        ]
        print(scorecard.loc[scorecard["scope"] == "ALL", columns].to_string(index=False))

    print("\nSaved:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
