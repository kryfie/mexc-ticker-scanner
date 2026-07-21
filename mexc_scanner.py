#!/usr/bin/env python3
"""
MEXC futures scanner matching the Pine:
"Analiza 3xSL - 5R - margin dynamiczny + taker fee"

Main rules reproduced:
- SMA 12 / 50 / 200
- Heikin Ashi calculated from normal OHLC
- Entry MA200_UP / MA200_DOWN
- Entry price = signal candle high for LONG, low for SHORT
- SL1 = MA50 on entry candle
- Entry rejected when SL1 ROI > max_sl1_roi_percent
- TP = tp_r * initial risk
- SL2 activated after MA12/MA200 signal
- SL3 activated after MA50/MA200 signal
- Conservative same-candle handling: SL wins over TP
- Dynamic margin = current capital * margin_multiplier / leverage
- Taker fee charged on entry and exit
- Capital compounded trade by trade
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


@dataclass
class ScanConfig:
    interval: str = "Min1"
    days: int = 7
    tp_r: float = 5.0
    max_sl1_roi_percent: float = 50.0
    leverage: float = 50.0
    taker_fee: float = 0.0008
    margin_multiplier: float = 10.0
    starting_capital: float = 5.0
    cooldown_minutes: int = 0
    conservative_same_candle: bool = True
    check_result_on_entry_bar: bool = False


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
    score: Optional[float]


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
    candidates = (
        row.get("maxLeverage"),
        row.get("max_leverage"),
        row.get("maxLever"),
        row.get("leverageMax"),
        row.get("maxLongLeverage"),
        row.get("maxShortLeverage"),
    )
    values = [to_int(value) for value in candidates]
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
        if meta.quote_coin != "USDT":
            continue
        if meta.state != 0:
            continue
        if not meta.api_allowed:
            continue
        if min_max_leverage is not None:
            if meta.max_leverage is None or meta.max_leverage < min_max_leverage:
                continue
        symbols.append(symbol)

    symbols = sorted(set(symbols))
    return symbols[:limit] if limit else symbols


def fetch_klines(symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}. Choose from: {', '.join(INTERVAL_SECONDS)}")

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
            frame = pd.DataFrame(
                {
                    "time": times,
                    "open": payload.get("open", [None] * size),
                    "high": payload.get("high", [None] * size),
                    "low": payload.get("low", [None] * size),
                    "close": payload.get("close", [None] * size),
                    "vol": payload.get("vol", [None] * size),
                }
            )
            frames.append(frame)

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
    current_win = 0
    current_loss = 0
    longest_win = 0
    longest_loss = 0
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


def safe_mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def safe_median(values: List[float]) -> Optional[float]:
    return float(pd.Series(values).median()) if values else None


def safe_max(values: List[float]) -> Optional[float]:
    return float(max(values)) if values else None


def scan_symbol(
    symbol: str,
    cfg: ScanConfig,
    meta: Optional[ContractMeta],
) -> Tuple[ScanResult, List[Dict[str, Any]]]:
    end = int(time.time())
    requested_start = end - cfg.days * 24 * 60 * 60

    # Enough warm-up for MA200 and HA recursion on every supported timeframe.
    warmup_bars = 500
    warmup_start = requested_start - warmup_bars * INTERVAL_SECONDS[cfg.interval]

    raw_df = fetch_klines(symbol, cfg.interval, warmup_start, end)
    max_leverage = meta.max_leverage if meta else None

    if len(raw_df) < 220:
        return build_empty_result(symbol, cfg, max_leverage), []

    df = prepare_data(raw_df)
    start_positions = df.index[df["time"] >= requested_start].tolist()
    if not start_positions:
        return build_empty_result(symbol, cfg, max_leverage), []

    first_scan_i = max(1, start_positions[0])

    state = empty_state()
    last_entry_time: Optional[int] = None
    cooldown_seconds = cfg.cooldown_minutes * 60

    raw_signals = 0
    entries = 0
    skipped_cooldown = 0
    skipped_in_trade = 0
    skipped_sl_too_large = 0
    tp_count = 0
    sl_count = 0
    sl1_count = 0
    sl2_count = 0
    sl3_count = 0

    capital = cfg.starting_capital
    highest_capital = cfg.starting_capital
    max_drawdown_pct = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0
    total_margin_used = 0.0
    net_r = 0.0

    sl1_price_pcts: List[float] = []
    sl1_roi_pcts: List[float] = []
    tp_roi_pcts: List[float] = []
    sequence: List[str] = []
    debug_rows: List[Dict[str, Any]] = []

    for i in range(first_scan_i, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

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
            ha_green
            and ha_no_lower
            and cross_ha_up
            and row["ma50"] < row["ma12"]
            and row["ma12"] < row["ma200"]
        )
        ma200_down_raw = (
            ha_red
            and ha_no_upper
            and cross_ha_down
            and row["ma50"] > row["ma12"]
            and row["ma12"] > row["ma200"]
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

        # Pine evaluates these blocks after entry blocks on the same candle.
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
            if highest_capital > 0
            else 0.0
        )
        max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
        sequence.append(result)

        debug_rows.append(
            {
                "ticker": symbol,
                "timeframe": cfg.interval,
                "max_leverage": max_leverage,
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
                "tp_hit": tp_hit,
                "sl1_hit": sl1_hit,
                "sl2_hit": sl2_hit,
                "sl3_hit": sl3_hit,
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
                "exit_candle_open": row["open"],
                "exit_candle_high": row["high"],
                "exit_candle_low": row["low"],
                "exit_candle_close": row["close"],
            }
        )

        state = empty_state()

    closed = tp_count + sl_count
    win_rate = tp_count / closed * 100.0 if closed else None
    expectancy_r = net_r / closed if closed else None
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    expectancy_roi = (
        (gross_profit - gross_loss) / total_margin_used * 100.0
        if total_margin_used > 0
        else None
    )
    return_pct = (
        (capital - cfg.starting_capital) / cfg.starting_capital * 100.0
        if cfg.starting_capital > 0
        else 0.0
    )
    longest_win, longest_loss, current_streak = calculate_streaks(sequence)

    score: Optional[float] = None
    if closed:
        # Ranking prioritizes real compounded return and penalizes drawdown.
        pf_for_score = min(profit_factor if profit_factor is not None else 5.0, 5.0)
        score = (
            return_pct
            + pf_for_score * 5.0
            + (win_rate or 0.0) * 0.15
            + min(closed, 50) * 0.20
            + max_drawdown_pct * 0.50
            - longest_loss * 0.75
        )

    result = ScanResult(
        ticker=symbol,
        timeframe=cfg.interval,
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
        score=score,
    )
    return result, debug_rows


def build_empty_result(
    symbol: str,
    cfg: ScanConfig,
    max_leverage: Optional[int],
) -> ScanResult:
    return ScanResult(
        ticker=symbol,
        timeframe=cfg.interval,
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
        score=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEXC scanner matching Pine 3xSL/5R with dynamic margin and taker fee."
    )
    parser.add_argument("--symbols", nargs="*", help="Example: SUI_USDT ONDO_USDT BTC_USDT")
    parser.add_argument("--all", action="store_true", help="Scan all enabled USDT perpetual contracts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-max-leverage", type=int, default=None)
    parser.add_argument("--interval", choices=sorted(INTERVAL_SECONDS), default="Min1")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--tp-r", type=float, default=5.0)
    parser.add_argument("--max-sl1-roi", type=float, default=50.0)
    parser.add_argument("--leverage", type=float, default=50.0)
    parser.add_argument("--taker-fee", type=float, default=0.0008)
    parser.add_argument("--margin-multiplier", type=float, default=10.0)
    parser.add_argument("--starting-capital", type=float, default=5.0)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument(
        "--check-result-on-entry-bar",
        action="store_true",
        help="Match Pine when this checkbox is enabled.",
    )
    parser.add_argument(
        "--tp-wins-same-candle",
        action="store_true",
        help="By default SL wins when TP and SL occur on the same candle.",
    )
    parser.add_argument("--output", default="ranking.csv")
    parser.add_argument("--debug-output", default="trades_debug.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be > 0")
    if args.leverage <= 0:
        raise SystemExit("--leverage must be > 0")
    if args.starting_capital <= 0:
        raise SystemExit("--starting-capital must be > 0")
    if args.margin_multiplier <= 0:
        raise SystemExit("--margin-multiplier must be > 0")
    if args.taker_fee < 0:
        raise SystemExit("--taker-fee cannot be negative")

    cfg = ScanConfig(
        interval=args.interval,
        days=args.days,
        tp_r=args.tp_r,
        max_sl1_roi_percent=args.max_sl1_roi,
        leverage=args.leverage,
        taker_fee=args.taker_fee,
        margin_multiplier=args.margin_multiplier,
        starting_capital=args.starting_capital,
        cooldown_minutes=args.cooldown,
        conservative_same_candle=not args.tp_wins_same_candle,
        check_result_on_entry_bar=args.check_result_on_entry_bar,
    )

    print("Loading MEXC contract metadata...", flush=True)
    meta_map = get_contract_meta()

    symbols = list(args.symbols or [])
    if args.all:
        symbols = get_contract_symbols(
            meta_map=meta_map,
            min_max_leverage=args.min_max_leverage,
            limit=args.limit,
        )
    elif args.min_max_leverage is not None:
        symbols = [
            symbol
            for symbol in symbols
            if symbol in meta_map
            and meta_map[symbol].max_leverage is not None
            and meta_map[symbol].max_leverage >= args.min_max_leverage
        ]

    symbols = sorted(set(symbols))
    if not symbols:
        raise SystemExit("Use --all or provide --symbols SUI_USDT BTC_USDT")

    print(f"Symbols: {len(symbols)}", flush=True)
    print(f"Config: {cfg}", flush=True)

    ranking_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for index, symbol in enumerate(symbols, start=1):
        max_lev = meta_map.get(symbol).max_leverage if meta_map.get(symbol) else None
        print(f"[{index}/{len(symbols)}] {symbol} | max leverage={max_lev}", flush=True)

        completed = False
        for attempt in range(1, 4):
            try:
                result, trades = scan_symbol(symbol, cfg, meta_map.get(symbol))
                ranking_rows.append(asdict(result))
                debug_rows.extend(trades)
                completed = True
                break
            except Exception as exc:
                print(f"ERROR {symbol} attempt {attempt}/3: {exc}", flush=True)
                if attempt < 3:
                    time.sleep(3 * attempt)

        if not completed:
            print(f"FAILED {symbol}", flush=True)

        time.sleep(0.10)

    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["score", "return_on_capital_pct", "profit_factor", "closed_trades"],
            ascending=[False, False, False, False],
            na_position="last",
        )

    ranking_df.to_csv(args.output, index=False)

    debug_df = pd.DataFrame(debug_rows)
    if debug_df.empty:
        debug_df = pd.DataFrame(
            columns=[
                "ticker", "timeframe", "entry_time_utc", "exit_time_utc",
                "direction", "result", "sl_type", "entry_price", "exit_price",
                "trade_r", "trade_roi_gross_pct", "fee_usdt", "pnl_net_usdt",
                "capital_before_usdt", "capital_after_usdt",
            ]
        )
    debug_df.to_csv(args.debug_output, index=False)

    print("\nTOP 30:", flush=True)
    if ranking_df.empty:
        print("No results.", flush=True)
    else:
        display_columns = [
            "ticker", "timeframe", "entries", "tp", "sl", "win_rate_pct",
            "profit_factor", "ending_capital_usdt", "return_on_capital_pct",
            "max_drawdown_pct", "total_fees_usdt", "score",
        ]
        print(ranking_df[display_columns].head(30).to_string(index=False), flush=True)

    print(f"\nSaved: {args.output}", flush=True)
    print(f"Saved: {args.debug_output}", flush=True)


if __name__ == "__main__":
    main()
