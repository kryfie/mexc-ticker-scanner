#!/usr/bin/env python3
"""
MEXC multi-parameter futures scanner matching the Pine:
"Analiza 3xSL - 5R - margin dynamiczny + taker fee"

One run tests:
- timeframes: Min1, Min5
- TP: 4R, 5R, 6R
- maximum SL1 ROI: 40%, 45%, 50%, 55%, 60%

For every symbol and timeframe, candles are downloaded once and reused for all
15 parameter combinations.

Outputs:
- ranking_all.csv: all configurations with at least --min-closed-trades
- ranking_best_per_ticker.csv: best eligible configuration per ticker
- parameter_summary.csv: aggregated parameter performance
- trades_debug.csv: trade-by-trade details for eligible configurations

Ranking order:
1. ending_capital_usdt DESC
2. profit_factor DESC
3. max_drawdown_pct DESC (e.g. -20% ranks above -40%)
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

BASE_URL = "https://contract.mexc.com"
DEFAULT_TIMEOUT = 30
EPSILON = 1e-10

INTERVAL_SECONDS: Dict[str, int] = {
    "Min1": 60,
    "Min5": 5 * 60,
    "Min15": 15 * 60,
    "Min30": 30 * 60,
    "Min60": 60 * 60,
    "Hour4": 4 * 60 * 60,
    "Hour8": 8 * 60 * 60,
    "Day1": 24 * 60 * 60,
}


@dataclass(frozen=True)
class ScanConfig:
    interval: str
    days: int
    tp_r: float
    max_sl1_roi_percent: float
    leverage: float
    taker_fee: float
    margin_multiplier: float
    starting_capital: float
    cooldown_minutes: int
    conservative_same_candle: bool
    check_result_on_entry_bar: bool


@dataclass
class ContractMeta:
    ticker: str
    max_leverage: Optional[int] = None
    quote_coin: Optional[str] = None
    state: Optional[int] = None
    api_allowed: bool = True


@dataclass
class ScanResult:
    ticker: str
    timeframe: str
    tp_r: float
    max_sl1_roi_limit_pct: float
    leverage: float
    taker_fee_per_side: float
    margin_multiplier: float
    max_leverage: Optional[int]
    raw_signals: int
    entries: int
    closed_trades: int
    open_trade_at_end: bool
    skipped_cooldown: int
    skipped_in_trade: int
    skipped_sl_too_large: int
    tp: int
    sl: int
    sl1: int
    sl2: int
    sl3: int
    win_rate_pct: Optional[float]
    net_r: float
    expectancy_r: Optional[float]
    avg_sl1_price_pct: Optional[float]
    median_sl1_price_pct: Optional[float]
    max_sl1_price_pct: Optional[float]
    avg_sl1_roi_pct: Optional[float]
    median_sl1_roi_pct: Optional[float]
    max_sl1_roi_pct: Optional[float]
    avg_tp_roi_pct: Optional[float]
    median_tp_roi_pct: Optional[float]
    max_tp_roi_pct: Optional[float]
    gross_profit_usdt: float
    gross_loss_usdt: float
    total_fees_usdt: float
    profit_factor: Optional[float]
    expectancy_roi_pct: Optional[float]
    starting_capital_usdt: float
    ending_capital_usdt: float
    highest_capital_usdt: float
    return_on_capital_pct: float
    max_drawdown_pct: float
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: str
    sequence_last_30: str


def mexc_get(
    path: str,
    params: Optional[dict] = None,
    retries: int = 4,
    backoff: float = 1.5,
) -> dict:
    url = f"{BASE_URL}{path}"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is False:
                raise RuntimeError(f"MEXC API error: {payload}")
            return payload
        except Exception as exc:
            last_error = exc
            print(f"Request error {attempt}/{retries}: {url} {params} -> {exc}", flush=True)
            if attempt < retries:
                time.sleep(backoff * attempt)
    assert last_error is not None
    raise last_error


def to_int(value: Any) -> Optional[int]:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def extract_max_leverage(row: Dict[str, Any]) -> Optional[int]:
    values = [
        to_int(row.get(key))
        for key in (
            "maxLeverage",
            "max_leverage",
            "maxLever",
            "leverageMax",
            "maxLongLeverage",
            "maxShortLeverage",
        )
    ]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def get_contract_meta() -> Dict[str, ContractMeta]:
    rows = mexc_get("/api/v1/contract/detail").get("data", [])
    result: Dict[str, ContractMeta] = {}
    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        result[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=extract_max_leverage(row),
            quote_coin=row.get("quoteCoin"),
            state=to_int(row.get("state")),
            api_allowed=bool(row.get("apiAllowed", True)),
        )
    return result


def get_contract_symbols(
    meta_map: Dict[str, ContractMeta],
    min_max_leverage: Optional[int],
    limit: Optional[int],
) -> List[str]:
    symbols: List[str] = []
    for symbol, meta in meta_map.items():
        if meta.quote_coin != "USDT" or meta.state != 0 or not meta.api_allowed:
            continue
        if min_max_leverage is not None and (
            meta.max_leverage is None or meta.max_leverage < min_max_leverage
        ):
            continue
        symbols.append(symbol)
    symbols = sorted(set(symbols))
    return symbols[:limit] if limit else symbols


def fetch_klines(symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    step = INTERVAL_SECONDS[interval]
    max_points = 1900
    chunk_span = step * max_points
    cursor = start
    frames: List[pd.DataFrame] = []

    while cursor < end:
        chunk_end = min(end, cursor + chunk_span - step)
        payload = mexc_get(
            f"/api/v1/contract/kline/{symbol}",
            {"interval": interval, "start": cursor, "end": chunk_end},
        ).get("data", {})
        times = payload.get("time") if payload else None
        if times:
            size = len(times)
            frames.append(
                pd.DataFrame(
                    {
                        "time": times,
                        "open": payload.get("open", [None] * size),
                        "high": payload.get("high", [None] * size),
                        "low": payload.get("low", [None] * size),
                        "close": payload.get("close", [None] * size),
                        "vol": payload.get("vol", [None] * size),
                    }
                )
            )
        cursor = chunk_end + step
        time.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "vol"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for column in ("time", "open", "high", "low", "close", "vol"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt"] = pd.to_datetime(out["time"], unit="s", utc=True)
    out["ma12"] = out["close"].rolling(12).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["ma200"] = out["close"].rolling(200).mean()
    out["ha_close"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0

    ha_open: List[float] = []
    for i, row in out.iterrows():
        if i == 0:
            value = (float(row["open"]) + float(row["close"])) / 2.0
        else:
            value = (ha_open[i - 1] + float(out.iloc[i - 1]["ha_close"])) / 2.0
        ha_open.append(value)

    out["ha_open"] = ha_open
    out["ha_high"] = out[["high", "ha_open", "ha_close"]].max(axis=1)
    out["ha_low"] = out[["low", "ha_open", "ha_close"]].min(axis=1)
    return out


def crossover(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return a_prev <= b_prev and a_now > b_now


def crossunder(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return a_prev >= b_prev and a_now < b_now


def approx_equal(a: float, b: float) -> bool:
    tolerance = EPSILON * max(1.0, abs(a), abs(b))
    return abs(a - b) <= tolerance


def empty_state() -> Dict[str, Any]:
    return {
        "in_trade": False,
        "direction": 0,
        "entry_i": None,
        "entry_time": None,
        "entry_price": None,
        "tp_price": None,
        "initial_risk": None,
        "initial_sl1_price_pct": None,
        "initial_sl1_roi_pct": None,
        "sl1_price": None,
        "sl2_price": None,
        "sl2_active": False,
        "sl3_price": None,
        "sl3_active": False,
        "margin": None,
        "capital_before": None,
    }


def calculate_streaks(sequence: Iterable[str]) -> Tuple[int, int, str]:
    current_win = current_loss = longest_win = longest_loss = 0
    current_text = "-"
    for item in sequence:
        if item == "TP":
            current_win += 1
            current_loss = 0
            longest_win = max(longest_win, current_win)
            current_text = f"W{current_win}"
        else:
            current_loss += 1
            current_win = 0
            longest_loss = max(longest_loss, current_loss)
            current_text = f"L{current_loss}"
    return longest_win, longest_loss, current_text


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    return float(pd.Series(values).median()) if values else None


def safe_max(values: Sequence[float]) -> Optional[float]:
    return float(max(values)) if values else None


def build_empty_result(
    symbol: str,
    cfg: ScanConfig,
    max_leverage: Optional[int],
) -> ScanResult:
    return ScanResult(
        ticker=symbol,
        timeframe=cfg.interval,
        tp_r=cfg.tp_r,
        max_sl1_roi_limit_pct=cfg.max_sl1_roi_percent,
        leverage=cfg.leverage,
        taker_fee_per_side=cfg.taker_fee,
        margin_multiplier=cfg.margin_multiplier,
        max_leverage=max_leverage,
        raw_signals=0,
        entries=0,
        closed_trades=0,
        open_trade_at_end=False,
        skipped_cooldown=0,
        skipped_in_trade=0,
        skipped_sl_too_large=0,
        tp=0,
        sl=0,
        sl1=0,
        sl2=0,
        sl3=0,
        win_rate_pct=None,
        net_r=0.0,
        expectancy_r=None,
        avg_sl1_price_pct=None,
        median_sl1_price_pct=None,
        max_sl1_price_pct=None,
        avg_sl1_roi_pct=None,
        median_sl1_roi_pct=None,
        max_sl1_roi_pct=None,
        avg_tp_roi_pct=None,
        median_tp_roi_pct=None,
        max_tp_roi_pct=None,
        gross_profit_usdt=0.0,
        gross_loss_usdt=0.0,
        total_fees_usdt=0.0,
        profit_factor=None,
        expectancy_roi_pct=None,
        starting_capital_usdt=cfg.starting_capital,
        ending_capital_usdt=cfg.starting_capital,
        highest_capital_usdt=cfg.starting_capital,
        return_on_capital_pct=0.0,
        max_drawdown_pct=0.0,
        longest_win_streak=0,
        longest_loss_streak=0,
        current_streak="-",
        sequence_last_30="",
    )


def backtest_prepared(
    symbol: str,
    prepared_df: pd.DataFrame,
    requested_start: int,
    cfg: ScanConfig,
    max_leverage: Optional[int],
) -> Tuple[ScanResult, List[Dict[str, Any]], List[Dict[str, Any]]]:
    start_positions = prepared_df.index[prepared_df["time"] >= requested_start].tolist()
    if len(prepared_df) < 220 or not start_positions:
        return build_empty_result(symbol, cfg, max_leverage), [], []

    first_scan_i = max(1, start_positions[0])
    state = empty_state()
    last_entry_time: Optional[int] = None
    cooldown_seconds = cfg.cooldown_minutes * 60

    raw_signals = entries = skipped_cooldown = skipped_in_trade = 0
    skipped_sl_too_large = tp_count = sl_count = 0
    sl1_count = sl2_count = sl3_count = 0

    capital = cfg.starting_capital
    highest_capital = cfg.starting_capital
    max_drawdown_pct = gross_profit = gross_loss = total_fees = 0.0
    total_margin_used = net_r = 0.0

    sl1_price_pcts: List[float] = []
    sl1_roi_pcts: List[float] = []
    tp_roi_pcts: List[float] = []
    sequence: List[str] = []
    debug_rows: List[Dict[str, Any]] = []
    signal_debug_rows: List[Dict[str, Any]] = []

    for i in range(first_scan_i, len(prepared_df)):
        row = prepared_df.iloc[i]
        prev = prepared_df.iloc[i - 1]
        needed = (
            row["ma12"], row["ma50"], row["ma200"],
            prev["ma12"], prev["ma50"], prev["ma200"],
            row["ha_close"], row["ha_open"], prev["ha_close"],
        )
        if any(pd.isna(value) for value in needed):
            continue

        cross_ma12_up = crossover(prev["ma12"], row["ma12"], prev["ma200"], row["ma200"])
        cross_ma12_down = crossunder(prev["ma12"], row["ma12"], prev["ma200"], row["ma200"])
        cross_ma50_up = crossover(prev["ma50"], row["ma50"], prev["ma200"], row["ma200"])
        cross_ma50_down = crossunder(prev["ma50"], row["ma50"], prev["ma200"], row["ma200"])
        cross_ha_up = crossover(prev["ha_close"], row["ha_close"], prev["ma200"], row["ma200"])
        cross_ha_down = crossunder(prev["ha_close"], row["ha_close"], prev["ma200"], row["ma200"])

        ha_green = row["ha_close"] > row["ha_open"]
        ha_red = row["ha_close"] < row["ha_open"]
        ha_no_lower = approx_equal(row["ha_low"], min(row["ha_open"], row["ha_close"]))
        ha_no_upper = approx_equal(row["ha_high"], max(row["ha_open"], row["ha_close"]))

        long_signal = cross_ma12_up and row["ma50"] < row["ma12"] and row["ma50"] < row["ma200"]
        short_signal = cross_ma12_down and row["ma50"] > row["ma12"] and row["ma50"] > row["ma200"]
        ma50_up = cross_ma50_up and row["ma12"] > row["ma50"] and row["ma12"] > row["ma200"]
        ma50_down = cross_ma50_down and row["ma12"] < row["ma50"] and row["ma12"] < row["ma200"]

        ma200_up_raw = (
            ha_green and ha_no_lower and cross_ha_up
            and row["ma50"] < row["ma12"] < row["ma200"]
        )
        ma200_down_raw = (
            ha_red and ha_no_upper and cross_ha_down
            and row["ma50"] > row["ma12"] > row["ma200"]
        )
        raw_signal = ma200_up_raw or ma200_down_raw
        row_time = int(row["time"])
        cooldown_ok = last_entry_time is None or row_time - last_entry_time >= cooldown_seconds

        if raw_signal:
            raw_signals += 1
            if not cooldown_ok:
                skipped_cooldown += 1
            if cooldown_ok and state["in_trade"]:
                skipped_in_trade += 1

        long_entry = float(row["high"])
        long_sl1 = float(row["ma50"])
        long_risk = long_entry - long_sl1
        long_sl1_price_pct = long_risk / long_entry * 100.0 if long_risk > 0 else math.nan
        long_sl1_roi_pct = long_sl1_price_pct * cfg.leverage if not math.isnan(long_sl1_price_pct) else math.nan
        long_valid = long_risk > 0 and long_sl1_roi_pct <= cfg.max_sl1_roi_percent

        short_entry = float(row["low"])
        short_sl1 = float(row["ma50"])
        short_risk = short_sl1 - short_entry
        short_sl1_price_pct = short_risk / short_entry * 100.0 if short_risk > 0 else math.nan
        short_sl1_roi_pct = short_sl1_price_pct * cfg.leverage if not math.isnan(short_sl1_price_pct) else math.nan
        short_valid = short_risk > 0 and short_sl1_roi_pct <= cfg.max_sl1_roi_percent

        if ma200_up_raw and cooldown_ok and not state["in_trade"] and not long_valid:
            skipped_sl_too_large += 1
        if ma200_down_raw and cooldown_ok and not state["in_trade"] and not short_valid:
            skipped_sl_too_large += 1

        ma200_up = ma200_up_raw and cooldown_ok and not state["in_trade"] and long_valid
        ma200_down = ma200_down_raw and cooldown_ok and not state["in_trade"] and short_valid

        if raw_signal:
            if not cooldown_ok:
                rejection_reason = "COOLDOWN"
            elif state["in_trade"]:
                rejection_reason = "IN_TRADE"
            elif ma200_up_raw and not long_valid:
                rejection_reason = "SL1_ROI_TOO_LARGE_LONG"
            elif ma200_down_raw and not short_valid:
                rejection_reason = "SL1_ROI_TOO_LARGE_SHORT"
            else:
                rejection_reason = "ACCEPTED"

            signal_debug_rows.append(
                {
                    "ticker": symbol,
                    "timeframe": cfg.interval,
                    "tp_r": cfg.tp_r,
                    "max_sl1_roi_limit_pct": cfg.max_sl1_roi_percent,
                    "bar_open_time_utc": pd.to_datetime(row_time, unit="s", utc=True),
                    "bar_close_time_utc": pd.to_datetime(
                        row_time + INTERVAL_SECONDS[cfg.interval], unit="s", utc=True
                    ),
                    "raw_direction": "LONG" if ma200_up_raw else "SHORT",
                    "accepted": bool(ma200_up or ma200_down),
                    "rejection_reason": rejection_reason,
                    "in_trade_before_signal": bool(state["in_trade"]),
                    "cooldown_ok": bool(cooldown_ok),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "prev_ha_close": float(prev["ha_close"]),
                    "ha_open": float(row["ha_open"]),
                    "ha_close": float(row["ha_close"]),
                    "ha_high": float(row["ha_high"]),
                    "ha_low": float(row["ha_low"]),
                    "ha_green": bool(ha_green),
                    "ha_red": bool(ha_red),
                    "ha_no_lower": bool(ha_no_lower),
                    "ha_no_upper": bool(ha_no_upper),
                    "prev_ma12": float(prev["ma12"]),
                    "ma12": float(row["ma12"]),
                    "prev_ma50": float(prev["ma50"]),
                    "ma50": float(row["ma50"]),
                    "prev_ma200": float(prev["ma200"]),
                    "ma200": float(row["ma200"]),
                    "cross_ha_up": bool(cross_ha_up),
                    "cross_ha_down": bool(cross_ha_down),
                    "cross_ma12_up": bool(cross_ma12_up),
                    "cross_ma12_down": bool(cross_ma12_down),
                    "cross_ma50_up": bool(cross_ma50_up),
                    "cross_ma50_down": bool(cross_ma50_down),
                    "candidate_entry_price": long_entry if ma200_up_raw else short_entry,
                    "candidate_sl1_price": long_sl1 if ma200_up_raw else short_sl1,
                    "candidate_sl1_price_pct": (
                        long_sl1_price_pct if ma200_up_raw else short_sl1_price_pct
                    ),
                    "candidate_sl1_roi_pct": (
                        long_sl1_roi_pct if ma200_up_raw else short_sl1_roi_pct
                    ),
                }
            )

        if ma200_up or ma200_down:
            direction = 1 if ma200_up else -1
            entry_price = long_entry if direction == 1 else short_entry
            sl1_price = long_sl1 if direction == 1 else short_sl1
            initial_risk = long_risk if direction == 1 else short_risk
            sl1_price_pct = long_sl1_price_pct if direction == 1 else short_sl1_price_pct
            sl1_roi_pct = long_sl1_roi_pct if direction == 1 else short_sl1_roi_pct
            tp_price = (
                entry_price + initial_risk * cfg.tp_r
                if direction == 1
                else entry_price - initial_risk * cfg.tp_r
            )
            margin = capital * cfg.margin_multiplier / cfg.leverage
            state = {
                "in_trade": True,
                "direction": direction,
                "entry_i": i,
                "entry_time": row_time,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "initial_risk": initial_risk,
                "initial_sl1_price_pct": sl1_price_pct,
                "initial_sl1_roi_pct": sl1_roi_pct,
                "sl1_price": sl1_price,
                "sl2_price": float(row["ha_low"] if direction == 1 else row["ha_high"]),
                "sl2_active": False,
                "sl3_price": None,
                "sl3_active": False,
                "margin": margin,
                "capital_before": capital,
            }
            entries += 1
            total_margin_used += margin
            sl1_price_pcts.append(sl1_price_pct)
            sl1_roi_pcts.append(sl1_roi_pct)
            tp_roi_pcts.append(sl1_roi_pct * cfg.tp_r)
            last_entry_time = row_time

        if state["in_trade"] and state["direction"] == 1 and long_signal and not state["sl2_active"]:
            state["sl2_active"] = True
        if state["in_trade"] and state["direction"] == -1 and short_signal and not state["sl2_active"]:
            state["sl2_active"] = True
        if state["in_trade"] and state["direction"] == 1 and ma50_up and not state["sl3_active"]:
            state["sl3_price"] = float(row["ma50"])
            state["sl3_active"] = True
        if state["in_trade"] and state["direction"] == -1 and ma50_down and not state["sl3_active"]:
            state["sl3_price"] = float(row["ma50"])
            state["sl3_active"] = True

        can_check = state["in_trade"] and (
            cfg.check_result_on_entry_bar or i > int(state["entry_i"])
        )
        if not can_check:
            continue

        direction = int(state["direction"])
        tp_hit = (
            direction == 1 and row["high"] >= state["tp_price"]
        ) or (
            direction == -1 and row["low"] <= state["tp_price"]
        )
        sl1_hit = (
            direction == 1 and row["low"] <= state["sl1_price"]
        ) or (
            direction == -1 and row["high"] >= state["sl1_price"]
        )
        sl2_hit = bool(state["sl2_active"]) and (
            (direction == 1 and row["low"] <= state["sl2_price"])
            or (direction == -1 and row["high"] >= state["sl2_price"])
        )
        sl3_hit = bool(state["sl3_active"]) and (
            (direction == 1 and row["low"] <= state["sl3_price"])
            or (direction == -1 and row["high"] >= state["sl3_price"])
        )

        sl_hit = sl1_hit or sl2_hit or sl3_hit
        result_sl = sl_hit and (cfg.conservative_same_candle or not tp_hit)
        result_tp = tp_hit and not result_sl
        if not (result_tp or result_sl):
            continue

        if result_tp:
            result = "TP"
            sl_type = ""
            exit_price = float(state["tp_price"])
            tp_count += 1
        else:
            result = "SL"
            sl_count += 1
            if sl1_hit:
                sl_type = "SL1 MA50 entry"
                exit_price = float(state["sl1_price"])
                sl1_count += 1
            elif sl2_hit:
                sl_type = "SL2 HA flat"
                exit_price = float(state["sl2_price"])
                sl2_count += 1
            else:
                sl_type = "SL3 MA50 cross"
                exit_price = float(state["sl3_price"])
                sl3_count += 1

        entry_price = float(state["entry_price"])
        initial_risk = float(state["initial_risk"])
        trade_r = (
            (exit_price - entry_price) / initial_risk
            if direction == 1
            else (entry_price - exit_price) / initial_risk
        )
        trade_price_pct = (
            (exit_price - entry_price) / entry_price * 100.0
            if direction == 1
            else (entry_price - exit_price) / entry_price * 100.0
        )
        trade_roi_pct = trade_price_pct * cfg.leverage
        margin = float(state["margin"])
        position_value = margin * cfg.leverage
        fee = position_value * cfg.taker_fee * 2.0
        pnl_gross = margin * trade_roi_pct / 100.0
        pnl_net = pnl_gross - fee

        net_r += trade_r
        total_fees += fee
        if pnl_net >= 0:
            gross_profit += pnl_net
        else:
            gross_loss += abs(pnl_net)

        capital += pnl_net
        highest_capital = max(highest_capital, capital)
        drawdown_pct = (
            (capital - highest_capital) / highest_capital * 100.0
            if highest_capital > 0 else 0.0
        )
        max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
        sequence.append(result)

        debug_rows.append(
            {
                "ticker": symbol,
                "timeframe": cfg.interval,
                "tp_r": cfg.tp_r,
                "max_sl1_roi_limit_pct": cfg.max_sl1_roi_percent,
                "leverage": cfg.leverage,
                "entry_time_utc": pd.to_datetime(state["entry_time"], unit="s", utc=True),
                "exit_time_utc": pd.to_datetime(row_time, unit="s", utc=True),
                "direction": "LONG" if direction == 1 else "SHORT",
                "result": result,
                "sl_type": sl_type,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp_price": state["tp_price"],
                "sl1_price": state["sl1_price"],
                "sl2_price": state["sl2_price"],
                "sl2_active_at_exit": state["sl2_active"],
                "sl3_price": state["sl3_price"],
                "sl3_active_at_exit": state["sl3_active"],
                "initial_risk_price": initial_risk,
                "initial_sl1_price_pct": state["initial_sl1_price_pct"],
                "initial_sl1_roi_pct": state["initial_sl1_roi_pct"],
                "trade_r": trade_r,
                "trade_price_pct": trade_price_pct,
                "trade_roi_gross_pct": trade_roi_pct,
                "capital_before_usdt": state["capital_before"],
                "margin_used_usdt": margin,
                "position_value_usdt": position_value,
                "fee_usdt": fee,
                "pnl_gross_usdt": pnl_gross,
                "pnl_net_usdt": pnl_net,
                "capital_after_usdt": capital,
            }
        )
        state = empty_state()

    closed = tp_count + sl_count
    win_rate = tp_count / closed * 100.0 if closed else None
    expectancy_r = net_r / closed if closed else None
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    expectancy_roi = (
        (gross_profit - gross_loss) / total_margin_used * 100.0
        if total_margin_used > 0 else None
    )
    return_pct = (
        (capital - cfg.starting_capital) / cfg.starting_capital * 100.0
        if cfg.starting_capital > 0 else 0.0
    )
    longest_win, longest_loss, current_streak = calculate_streaks(sequence)

    result = ScanResult(
        ticker=symbol,
        timeframe=cfg.interval,
        tp_r=cfg.tp_r,
        max_sl1_roi_limit_pct=cfg.max_sl1_roi_percent,
        leverage=cfg.leverage,
        taker_fee_per_side=cfg.taker_fee,
        margin_multiplier=cfg.margin_multiplier,
        max_leverage=max_leverage,
        raw_signals=raw_signals,
        entries=entries,
        closed_trades=closed,
        open_trade_at_end=bool(state["in_trade"]),
        skipped_cooldown=skipped_cooldown,
        skipped_in_trade=skipped_in_trade,
        skipped_sl_too_large=skipped_sl_too_large,
        tp=tp_count,
        sl=sl_count,
        sl1=sl1_count,
        sl2=sl2_count,
        sl3=sl3_count,
        win_rate_pct=win_rate,
        net_r=net_r,
        expectancy_r=expectancy_r,
        avg_sl1_price_pct=safe_mean(sl1_price_pcts),
        median_sl1_price_pct=safe_median(sl1_price_pcts),
        max_sl1_price_pct=safe_max(sl1_price_pcts),
        avg_sl1_roi_pct=safe_mean(sl1_roi_pcts),
        median_sl1_roi_pct=safe_median(sl1_roi_pcts),
        max_sl1_roi_pct=safe_max(sl1_roi_pcts),
        avg_tp_roi_pct=safe_mean(tp_roi_pcts),
        median_tp_roi_pct=safe_median(tp_roi_pcts),
        max_tp_roi_pct=safe_max(tp_roi_pcts),
        gross_profit_usdt=gross_profit,
        gross_loss_usdt=gross_loss,
        total_fees_usdt=total_fees,
        profit_factor=profit_factor,
        expectancy_roi_pct=expectancy_roi,
        starting_capital_usdt=cfg.starting_capital,
        ending_capital_usdt=capital,
        highest_capital_usdt=highest_capital,
        return_on_capital_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        current_streak=current_streak,
        sequence_last_30=" -> ".join(sequence[-30:]),
    )
    return result, debug_rows, signal_debug_rows


def add_robustness(all_results: pd.DataFrame, total_configurations: int) -> pd.DataFrame:
    if all_results.empty:
        return all_results
    summary = (
        all_results.groupby("ticker", as_index=False)
        .agg(
            profitable_config_count=("ending_capital_usdt", lambda s: int((s > all_results["starting_capital_usdt"].iloc[0]).sum())),
            tested_config_count=("ending_capital_usdt", "size"),
            eligible_config_count=("eligible_min_trades", "sum"),
        )
    )
    summary["robustness_pct"] = summary["profitable_config_count"] / total_configurations * 100.0
    return all_results.merge(summary, on="ticker", how="left")


def build_parameter_summary(eligible: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dimension", "value", "eligible_configurations", "unique_tickers",
        "avg_ending_capital_usdt", "median_ending_capital_usdt",
        "avg_return_on_capital_pct", "median_return_on_capital_pct",
        "avg_profit_factor", "avg_max_drawdown_pct", "avg_closed_trades",
        "profitable_configuration_pct",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    parts: List[pd.DataFrame] = []
    for dimension, source_column in (
        ("timeframe", "timeframe"),
        ("tp_r", "tp_r"),
        ("max_sl1_roi_limit_pct", "max_sl1_roi_limit_pct"),
    ):
        grouped = eligible.groupby(source_column, dropna=False)
        part = grouped.agg(
            eligible_configurations=("ticker", "size"),
            unique_tickers=("ticker", "nunique"),
            avg_ending_capital_usdt=("ending_capital_usdt", "mean"),
            median_ending_capital_usdt=("ending_capital_usdt", "median"),
            avg_return_on_capital_pct=("return_on_capital_pct", "mean"),
            median_return_on_capital_pct=("return_on_capital_pct", "median"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_closed_trades=("closed_trades", "mean"),
            profitable_configuration_pct=("is_profitable", "mean"),
        ).reset_index().rename(columns={source_column: "value"})
        part["profitable_configuration_pct"] *= 100.0
        part.insert(0, "dimension", dimension)
        parts.append(part[columns])

    combo = (
        eligible.groupby(["timeframe", "tp_r", "max_sl1_roi_limit_pct"], dropna=False)
        .agg(
            eligible_configurations=("ticker", "size"),
            unique_tickers=("ticker", "nunique"),
            avg_ending_capital_usdt=("ending_capital_usdt", "mean"),
            median_ending_capital_usdt=("ending_capital_usdt", "median"),
            avg_return_on_capital_pct=("return_on_capital_pct", "mean"),
            median_return_on_capital_pct=("return_on_capital_pct", "median"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_closed_trades=("closed_trades", "mean"),
            profitable_configuration_pct=("is_profitable", "mean"),
        )
        .reset_index()
    )
    combo["profitable_configuration_pct"] *= 100.0
    combo.insert(0, "dimension", "full_combination")
    combo["value"] = (
        combo["timeframe"].astype(str)
        + " | TP " + combo["tp_r"].astype(str) + "R"
        + " | SL1 " + combo["max_sl1_roi_limit_pct"].astype(str) + "%"
    )
    parts.append(combo[columns])

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(
        ["dimension", "avg_ending_capital_usdt", "avg_profit_factor", "avg_max_drawdown_pct"],
        ascending=[True, False, False, False],
        na_position="last",
    )



def scan_symbol_interval_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker process for one ticker + timeframe.

    It downloads candles once, prepares indicators once, then runs all TP/SL1
    parameter combinations locally.
    """
    symbol = str(task["symbol"])
    interval = str(task["interval"])
    max_leverage = task.get("max_leverage")
    requested_start = int(task["requested_start"])
    scan_end = int(task["scan_end"])
    tp_r_values = [float(value) for value in task["tp_r_values"]]
    max_sl1_roi_values = [float(value) for value in task["max_sl1_roi_values"]]

    warmup_start = requested_start - 500 * INTERVAL_SECONDS[interval]
    raw = fetch_klines(symbol, interval, warmup_start, scan_end)
    prepared = prepare_data(raw)

    result_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    signal_debug_rows: List[Dict[str, Any]] = []

    for tp_r in tp_r_values:
        for max_sl1_roi in max_sl1_roi_values:
            cfg = ScanConfig(
                interval=interval,
                days=int(task["days"]),
                tp_r=tp_r,
                max_sl1_roi_percent=max_sl1_roi,
                leverage=float(task["leverage"]),
                taker_fee=float(task["taker_fee"]),
                margin_multiplier=float(task["margin_multiplier"]),
                starting_capital=float(task["starting_capital"]),
                cooldown_minutes=int(task["cooldown"]),
                conservative_same_candle=bool(task["conservative_same_candle"]),
                check_result_on_entry_bar=bool(task["check_result_on_entry_bar"]),
            )
            result, trades, signals = backtest_prepared(
                symbol=symbol,
                prepared_df=prepared,
                requested_start=requested_start,
                cfg=cfg,
                max_leverage=max_leverage,
            )
            row = asdict(result)
            row["eligible_min_trades"] = (
                result.closed_trades >= int(task["min_closed_trades"])
            )
            row["is_profitable"] = (
                result.ending_capital_usdt > result.starting_capital_usdt
            )
            result_rows.append(row)

            if row["eligible_min_trades"]:
                debug_rows.extend(trades)
            signal_debug_rows.extend(signals)

    return {
        "symbol": symbol,
        "interval": interval,
        "results": result_rows,
        "debug": debug_rows,
        "signals_debug": signal_debug_rows,
    }

def parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List cannot be empty")
    return values


def parse_str_list(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List cannot be empty")
    invalid = [item for item in values if item not in INTERVAL_SECONDS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported intervals: {invalid}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MEXC multi-parameter Pine scanner")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-max-leverage", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--intervals", type=parse_str_list, default=["Min1", "Min5"])
    parser.add_argument("--tp-r-values", type=parse_float_list, default=[4.0, 5.0, 6.0])
    parser.add_argument("--max-sl1-roi-values", type=parse_float_list, default=[40.0, 45.0, 50.0, 55.0, 60.0])
    parser.add_argument("--min-closed-trades", type=int, default=30)
    parser.add_argument("--leverage", type=float, default=50.0)
    parser.add_argument("--taker-fee", type=float, default=0.0008)
    parser.add_argument("--margin-multiplier", type=float, default=10.0)
    parser.add_argument("--starting-capital", type=float, default=5.0)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel ticker+timeframe workers. On GitHub-hosted runners 2 is usually safest.",
    )
    parser.add_argument("--check-result-on-entry-bar", action="store_true")
    parser.add_argument("--tp-wins-same-candle", action="store_true")
    parser.add_argument("--ranking-all", default="ranking_all.csv")
    parser.add_argument("--ranking-best", default="ranking_best_per_ticker.csv")
    parser.add_argument("--parameter-summary", default="parameter_summary.csv")
    parser.add_argument("--debug-output", default="trades_debug.csv")
    parser.add_argument("--signals-debug-output", default="signals_debug.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days <= 0 or args.min_closed_trades < 0:
        raise SystemExit("Invalid days or minimum trade count")
    if args.leverage <= 0 or args.starting_capital <= 0 or args.margin_multiplier <= 0:
        raise SystemExit("Leverage, starting capital and margin multiplier must be > 0")
    if args.taker_fee < 0:
        raise SystemExit("Taker fee cannot be negative")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be > 0")

    print("Loading MEXC contract metadata...", flush=True)
    meta_map = get_contract_meta()

    symbols = list(args.symbols or [])
    if args.all:
        symbols = get_contract_symbols(meta_map, args.min_max_leverage, args.limit)
    else:
        symbols = [
            symbol for symbol in symbols
            if symbol in meta_map
            and (
                args.min_max_leverage is None
                or (
                    meta_map[symbol].max_leverage is not None
                    and meta_map[symbol].max_leverage >= args.min_max_leverage
                )
            )
        ]
    symbols = sorted(set(symbols))
    if not symbols:
        raise SystemExit("Use --all or provide --symbols")

    combinations = list(itertools.product(args.intervals, args.tp_r_values, args.max_sl1_roi_values))
    total_configurations = len(combinations)
    print(f"Symbols: {len(symbols)}", flush=True)
    print(f"Configurations per ticker: {total_configurations}", flush=True)
    print(f"Minimum closed trades: {args.min_closed_trades}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    signal_debug_rows: List[Dict[str, Any]] = []
    scan_end = int(time.time())
    requested_start = scan_end - args.days * 24 * 60 * 60

    max_workers = max(1, int(args.max_workers))
    tasks: List[Dict[str, Any]] = []
    for symbol in symbols:
        max_leverage = meta_map.get(symbol).max_leverage if meta_map.get(symbol) else None
        for interval in args.intervals:
            tasks.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "max_leverage": max_leverage,
                    "requested_start": requested_start,
                    "scan_end": scan_end,
                    "tp_r_values": args.tp_r_values,
                    "max_sl1_roi_values": args.max_sl1_roi_values,
                    "days": args.days,
                    "leverage": args.leverage,
                    "taker_fee": args.taker_fee,
                    "margin_multiplier": args.margin_multiplier,
                    "starting_capital": args.starting_capital,
                    "cooldown": args.cooldown,
                    "conservative_same_candle": not args.tp_wins_same_candle,
                    "check_result_on_entry_bar": args.check_result_on_entry_bar,
                    "min_closed_trades": args.min_closed_trades,
                }
            )

    print(f"Parallel workers: {max_workers}", flush=True)
    print(f"Ticker/timeframe tasks: {len(tasks)}", flush=True)

    completed_tasks = 0
    failed_tasks = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(scan_symbol_interval_worker, task): (
                task["symbol"],
                task["interval"],
            )
            for task in tasks
        }

        for future in as_completed(future_map):
            symbol, interval = future_map[future]
            try:
                payload = future.result()
                all_rows.extend(payload["results"])
                debug_rows.extend(payload["debug"])
                signal_debug_rows.extend(payload["signals_debug"])
                completed_tasks += 1
                print(
                    f"[{completed_tasks + failed_tasks}/{len(tasks)}] "
                    f"done {symbol} {interval} | "
                    f"configs={len(payload['results'])}",
                    flush=True,
                )
            except Exception as exc:
                failed_tasks += 1
                print(
                    f"[{completed_tasks + failed_tasks}/{len(tasks)}] "
                    f"FAILED {symbol} {interval}: {exc}",
                    flush=True,
                )

    print(
        f"Parallel scan finished: completed={completed_tasks}, failed={failed_tasks}",
        flush=True,
    )

    all_results = pd.DataFrame(all_rows)
    if all_results.empty:
        raise SystemExit("No scan results were generated")

    all_results = add_robustness(all_results, total_configurations)
    eligible = all_results[all_results["eligible_min_trades"]].copy()

    sort_columns = ["ending_capital_usdt", "profit_factor", "max_drawdown_pct"]
    eligible = eligible.sort_values(
        sort_columns,
        ascending=[False, False, False],
        na_position="last",
    )
    eligible.to_csv(args.ranking_all, index=False)

    if eligible.empty:
        best = eligible.copy()
    else:
        best = eligible.drop_duplicates(subset=["ticker"], keep="first")
    best.to_csv(args.ranking_best, index=False)

    parameter_summary = build_parameter_summary(eligible)
    parameter_summary.to_csv(args.parameter_summary, index=False)

    debug_df = pd.DataFrame(debug_rows)
    if debug_df.empty:
        debug_df = pd.DataFrame(
            columns=[
                "ticker", "timeframe", "tp_r", "max_sl1_roi_limit_pct",
                "entry_time_utc", "exit_time_utc", "direction", "result",
                "sl_type", "entry_price", "exit_price", "trade_r",
                "fee_usdt", "pnl_net_usdt", "capital_after_usdt",
            ]
        )
    debug_df.to_csv(args.debug_output, index=False)

    signals_debug_df = pd.DataFrame(signal_debug_rows)
    if signals_debug_df.empty:
        signals_debug_df = pd.DataFrame(
            columns=[
                "ticker", "timeframe", "tp_r", "max_sl1_roi_limit_pct",
                "bar_open_time_utc", "bar_close_time_utc", "raw_direction",
                "accepted", "rejection_reason", "open", "high", "low", "close",
                "ha_open", "ha_close", "ha_high", "ha_low",
                "ma12", "ma50", "ma200",
            ]
        )
    signals_debug_df.to_csv(args.signals_debug_output, index=False)

    print("\nTOP 30 ELIGIBLE CONFIGURATIONS:", flush=True)
    if eligible.empty:
        print("No configurations reached the minimum closed-trade threshold.", flush=True)
    else:
        display = [
            "ticker", "timeframe", "tp_r", "max_sl1_roi_limit_pct",
            "closed_trades", "tp", "sl", "win_rate_pct",
            "ending_capital_usdt", "profit_factor", "max_drawdown_pct",
            "robustness_pct",
        ]
        print(eligible[display].head(30).to_string(index=False), flush=True)

    print(f"\nSaved: {args.ranking_all}", flush=True)
    print(f"Saved: {args.ranking_best}", flush=True)
    print(f"Saved: {args.parameter_summary}", flush=True)
    print(f"Saved: {args.debug_output}", flush=True)
    print(f"Saved: {args.signals_debug_output}", flush=True)


if __name__ == "__main__":
    main()
