#!/usr/bin/env python3
"""
MEXC FINAL VALIDATION SCANNER
=============================

This is the NEXT validation stage after the broad 90-day MEXC scan.

It does NOT optimize parameters again.

Frozen candidates / frozen profiles:
- RIF_USDT         -> SELECTIVE_32R
- COHRSTOCK_USDT   -> MOMENTUM_28R
- ICX_USDT         -> TREND_30R
- ESPORTS_USDT     -> MOMENTUM_28R

For every candidate it stress-tests:
- BOTH
- LONG_ONLY
- SHORT_ONLY

against adverse slippage per side:
- 0.03%
- 0.05%
- 0.08%
- 0.10%

and independent windows:
- OLD_30D      = oldest 30 days of the 90-day sample
- MID_30D      = middle 30 days
- RECENT_30D   = most recent 30 days
- FULL90       = whole 90-day sample

Each window starts again from 5 USDT so the 30-day periods are directly comparable.
FULL90 compounds continuously from 5 USDT.

Execution model is frozen:
- 5m base candles
- 15m higher-timeframe trend
- 10x leverage
- 4% target equity risk per trade
- max 35% equity as isolated margin
- MEXC taker fee = 0.08% per side
- one position at a time per ticker
- next-candle-open entry
- SL wins if SL and TP touch in the same candle
- time exit after profile-specific max holding bars

IMPORTANT HTF FIX
-----------------
MEXC kline timestamps are treated as candle START timestamps.
For higher-timeframe construction we first convert every 5m candle to its CLOSE
timestamp (+5 minutes), and only then build 15m candles. This prevents an
accidental future 5m candle from entering a supposedly closed 15m bar.

The broad scanner did not explicitly make that start-time -> close-time shift.
Therefore this final validation can produce different results. That is intentional:
the live bot must be based only on information genuinely available at decision time.

Outputs:
- validation_ranking.csv
- validation_summary.csv
- validation_trades.csv
- validation_errors.csv

Research tool only. Historical profitability does not guarantee future profit.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


BASE_URL = "https://contract.mexc.com"
TIMEOUT = 30

BASE_INTERVAL = "Min5"
BASE_SECONDS = 5 * 60
CHUNK_POINTS = 1800

DAYS = 90
STARTING_CAPITAL = 5.0
LEVERAGE = 10.0
RISK_PCT = 0.04
MAX_MARGIN_PCT = 0.35
TAKER_FEE = 0.0008

SLIPPAGE_SCENARIOS = [0.0003, 0.0005, 0.0008, 0.0010]
DIRECTION_MODES = ["BOTH", "LONG_ONLY", "SHORT_ONLY"]


@dataclass(frozen=True)
class Profile:
    name: str
    ht_fast: int
    ht_slow: int
    adx_min: float
    volume_mult: float
    pullback_atr: float
    deep_pullback_atr: float
    stop_atr: float
    swing_bars: int
    rr: float
    min_stop_pct: float
    max_stop_pct: float
    min_atr_pct: float
    max_atr_pct: float
    cooldown_bars: int
    max_hold_bars: int


MOMENTUM_28R = Profile(
    "MOMENTUM_28R",
    ht_fast=30,
    ht_slow=100,
    adx_min=23,
    volume_mult=1.10,
    pullback_atr=0.45,
    deep_pullback_atr=0.60,
    stop_atr=1.35,
    swing_bars=3,
    rr=2.8,
    min_stop_pct=0.0035,
    max_stop_pct=0.022,
    min_atr_pct=0.0025,
    max_atr_pct=0.050,
    cooldown_bars=4,
    max_hold_bars=42,
)

TREND_30R = Profile(
    "TREND_30R",
    ht_fast=50,
    ht_slow=150,
    adx_min=22,
    volume_mult=1.00,
    pullback_atr=0.60,
    deep_pullback_atr=0.75,
    stop_atr=1.55,
    swing_bars=4,
    rr=3.0,
    min_stop_pct=0.0040,
    max_stop_pct=0.027,
    min_atr_pct=0.0020,
    max_atr_pct=0.045,
    cooldown_bars=4,
    max_hold_bars=60,
)

SELECTIVE_32R = Profile(
    "SELECTIVE_32R",
    ht_fast=30,
    ht_slow=120,
    adx_min=26,
    volume_mult=1.20,
    pullback_atr=0.40,
    deep_pullback_atr=0.55,
    stop_atr=1.30,
    swing_bars=3,
    rr=3.2,
    min_stop_pct=0.0035,
    max_stop_pct=0.020,
    min_atr_pct=0.0025,
    max_atr_pct=0.045,
    cooldown_bars=5,
    max_hold_bars=36,
)

CANDIDATES: Dict[str, Profile] = {
    "RIF_USDT": SELECTIVE_32R,
    "COHRSTOCK_USDT": MOMENTUM_28R,
    "ICX_USDT": TREND_30R,
    "ESPORTS_USDT": MOMENTUM_28R,
}


@dataclass
class ContractMeta:
    ticker: str
    max_leverage: Optional[int]
    contract_size: float
    min_vol: float
    state: Optional[int]
    quote_coin: str
    api_allowed: bool


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    time_exits: int
    win_rate_pct: Optional[float]

    gross_profit_usdt: float
    gross_loss_usdt: float
    profit_factor: Optional[float]

    fees_usdt: float
    net_pnl_usdt: float
    start_equity_usdt: float
    end_equity_usdt: float
    return_pct: float

    max_drawdown_pct: float
    min_equity_usdt: float

    expectancy_usdt: Optional[float]
    avg_trade_return_pct: Optional[float]
    median_trade_return_pct: Optional[float]
    avg_stop_pct: Optional[float]
    avg_rr_realized_after_fee: Optional[float]

    longest_loss_streak: int
    longest_win_streak: int
    avg_hold_bars: Optional[float]

    long_trades: int
    long_net_pnl_usdt: float
    long_profit_factor: Optional[float]

    short_trades: int
    short_net_pnl_usdt: float
    short_profit_factor: Optional[float]

    blown_up: bool


def mexc_get(path: str, params: Optional[dict] = None, retries: int = 4) -> dict:
    url = BASE_URL + path
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and payload.get("success") is False:
                raise RuntimeError(f"MEXC success=false: {payload}")

            return payload
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(attempt)

    assert last_exc is not None
    raise last_exc


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_candidate_contracts() -> Dict[str, ContractMeta]:
    rows = mexc_get("/api/v1/contract/detail").get("data", [])
    wanted = set(CANDIDATES)
    result: Dict[str, ContractMeta] = {}

    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        if symbol not in wanted:
            continue

        max_lev_candidates = [
            as_int(row.get("maxLeverage")),
            as_int(row.get("maxLongLeverage")),
            as_int(row.get("maxShortLeverage")),
            as_int(row.get("maxLever")),
        ]
        max_lev_values = [x for x in max_lev_candidates if x is not None]
        max_lev = max(max_lev_values) if max_lev_values else None

        result[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=max_lev,
            contract_size=max(as_float(row.get("contractSize"), 1.0), 1e-12),
            min_vol=max(as_float(row.get("minVol"), 1.0), 1e-12),
            state=as_int(row.get("state")),
            quote_coin=str(row.get("quoteCoin") or "").upper(),
            api_allowed=bool(row.get("apiAllowed", True)),
        )

    return result


def fetch_5m(symbol: str, days: int = DAYS) -> pd.DataFrame:
    now = int(time.time())
    requested_start = now - days * 86400

    # Indicator warmup. Results are still strictly limited to requested 90 days.
    fetch_start = requested_start - 12 * 86400

    cursor = fetch_start
    frames: List[pd.DataFrame] = []
    chunk_span = CHUNK_POINTS * BASE_SECONDS

    while cursor < now:
        chunk_end = min(now, cursor + chunk_span - BASE_SECONDS)

        data = mexc_get(
            f"/api/v1/contract/kline/{symbol}",
            {
                "interval": BASE_INTERVAL,
                "start": cursor,
                "end": chunk_end,
            },
        ).get("data", {})

        if isinstance(data, dict) and data.get("time"):
            n = len(data["time"])
            volume = data.get("vol") or data.get("amount") or [0.0] * n
            if len(volume) != n:
                volume = [0.0] * n

            frames.append(
                pd.DataFrame(
                    {
                        "time": data["time"],
                        "open": data.get("open", [None] * n),
                        "high": data.get("high", [None] * n),
                        "low": data.get("low", [None] * n),
                        "close": data.get("close", [None] * n),
                        "volume": volume,
                    }
                )
            )

        cursor = chunk_end + BASE_SECONDS

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    for column in ["time", "open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df["time"] = df["time"].astype("int64")

    # only completed 5m candles
    df = df[(df["time"] + BASE_SECONDS) <= now].reset_index(drop=True)

    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr_value = atr(df, period)

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_value
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_value
    )

    denominator = (plus_di + minus_di).replace(0, math.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def enrich(df5: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """
    Build 5m indicators and CLOSED 15m indicators without lookahead.

    MEXC 5m timestamp = bar start.
    Example:
       timestamp 10:00 = candle 10:00-10:05
       timestamp 10:05 = candle 10:05-10:10
       timestamp 10:10 = candle 10:10-10:15

    Their CLOSE timestamps are 10:05, 10:10, 10:15.
    Those three bars form the 15m candle CLOSED at 10:15.
    """
    out = df5.copy()
    out["bar_start_dt"] = pd.to_datetime(out["time"], unit="s", utc=True)
    out["bar_close_dt"] = out["bar_start_dt"] + pd.Timedelta(seconds=BASE_SECONDS)

    # 5m indicators
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"]
    out["vol_avg20"] = out["volume"].rolling(20, min_periods=20).mean()

    # Build HTF from BAR CLOSE timestamps.
    htf_source = out.set_index("bar_close_dt")

    df15 = (
        htf_source[["open", "high", "low", "close", "volume"]]
        .resample(
            "15min",
            label="right",
            closed="right",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    df15["ht_fast"] = ema(df15["close"], profile.ht_fast)
    df15["ht_slow"] = ema(df15["close"], profile.ht_slow)
    df15["ht_adx14"] = adx(df15, 14)

    htf = df15[["close", "ht_fast", "ht_slow", "ht_adx14"]].copy()
    htf = htf.rename(columns={"close": "ht_close"}).reset_index()

    # At the close of a 5m signal candle, only HTF bars closed at or before that
    # exact close time are available.
    out = pd.merge_asof(
        out.sort_values("bar_close_dt"),
        htf.sort_values("bar_close_dt"),
        on="bar_close_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    return out.sort_values("time").reset_index(drop=True)


def adverse_entry(raw_open: float, direction: int, slippage: float) -> float:
    if direction == 1:
        return raw_open * (1 + slippage)
    return raw_open * (1 - slippage)


def adverse_exit(raw_exit: float, direction: int, slippage: float) -> float:
    if direction == 1:
        return raw_exit * (1 - slippage)
    return raw_exit * (1 + slippage)


def profit_factor(gross_profit: float, gross_loss: float) -> Optional[float]:
    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return math.inf
    return None


def longest_streak(sequence: List[str], wanted: str) -> int:
    best = 0
    current = 0

    for result in sequence:
        if result == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def simulate(
    df: pd.DataFrame,
    profile: Profile,
    meta: ContractMeta,
    start_ts: int,
    end_ts: int,
    window_name: str,
    direction_mode: str,
    slippage: float,
) -> Tuple[Metrics, List[Dict[str, Any]]]:
    equity = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    minimum_equity = STARTING_CAPITAL
    max_drawdown = 0.0

    position: Optional[Dict[str, Any]] = None
    last_exit_i: Optional[int] = None

    wins = 0
    losses = 0
    time_exits = 0

    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0

    trade_returns: List[float] = []
    stop_pct_values: List[float] = []
    rr_values: List[float] = []
    holding_values: List[int] = []
    result_sequence: List[str] = []

    long_n = 0
    long_gp = 0.0
    long_gl = 0.0
    long_net = 0.0

    short_n = 0
    short_gp = 0.0
    short_gl = 0.0
    short_net = 0.0

    trades: List[Dict[str, Any]] = []

    start_indices = df.index[df["time"] >= start_ts].tolist()
    end_indices = df.index[df["time"] <= end_ts].tolist()

    if not start_indices or not end_indices:
        return empty_metrics(), []

    first_i = max(
        start_indices[0],
        55,
        profile.swing_bars + 2,
    )
    last_i = min(
        end_indices[-1],
        len(df) - 2,
    )

    i = first_i

    while i <= last_i:
        row = df.iloc[i]

        # -------------------------------------------------------------
        # Manage existing position.
        # Entry was at the open of its entry candle, so SL/TP may be hit
        # during that same candle.
        # -------------------------------------------------------------
        if position is not None:
            direction = int(position["direction"])

            hit_sl = (
                float(row["low"]) <= float(position["sl"])
                if direction == 1
                else float(row["high"]) >= float(position["sl"])
            )
            hit_tp = (
                float(row["high"]) >= float(position["tp"])
                if direction == 1
                else float(row["low"]) <= float(position["tp"])
            )

            exit_reason: Optional[str] = None
            raw_exit: Optional[float] = None

            # Conservative intrabar rule: SL wins conflict.
            if hit_sl:
                exit_reason = "SL"
                raw_exit = float(position["sl"])
            elif hit_tp:
                exit_reason = "TP"
                raw_exit = float(position["tp"])
            elif i - int(position["entry_i"]) >= profile.max_hold_bars:
                exit_reason = "TIME"
                raw_exit = float(row["close"])

            if exit_reason is not None and raw_exit is not None:
                exit_price = adverse_exit(raw_exit, direction, slippage)

                qty = float(position["qty"])
                gross = qty * (exit_price - float(position["entry"])) * direction

                exit_fee = abs(qty * exit_price) * TAKER_FEE
                trade_fee = float(position["entry_fee"]) + exit_fee
                net = gross - trade_fee

                equity_before = equity
                equity += net

                total_fees += trade_fee
                if gross > 0:
                    gross_profit += gross
                elif gross < 0:
                    gross_loss += gross

                trade_return = (
                    net / equity_before * 100.0
                    if equity_before > 0
                    else -100.0
                )
                trade_returns.append(trade_return)

                hold_bars = i - int(position["entry_i"])
                holding_values.append(hold_bars)

                risk_dollars = (
                    float(position["notional"])
                    * float(position["stop_pct"])
                )
                realized_r = (
                    net / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )
                rr_values.append(realized_r)

                if exit_reason == "TP":
                    wins += 1
                    result_sequence.append("W")
                elif exit_reason == "SL":
                    losses += 1
                    result_sequence.append("L")
                else:
                    time_exits += 1
                    result_sequence.append("W" if net > 0 else "L")

                if direction == 1:
                    long_n += 1
                    long_net += net
                    if gross > 0:
                        long_gp += gross
                    elif gross < 0:
                        long_gl += gross
                else:
                    short_n += 1
                    short_net += net
                    if gross > 0:
                        short_gp += gross
                    elif gross < 0:
                        short_gl += gross

                peak = max(peak, equity)
                minimum_equity = min(minimum_equity, equity)

                drawdown = (
                    (peak - equity) / peak * 100.0
                    if peak > 0
                    else 100.0
                )
                max_drawdown = max(max_drawdown, drawdown)

                trades.append(
                    {
                        "ticker": meta.ticker,
                        "profile": profile.name,
                        "direction_mode": direction_mode,
                        "slippage_per_side_pct": slippage * 100.0,
                        "window": window_name,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "signal_time_utc": pd.Timestamp(
                            int(position["signal_time"]),
                            unit="s",
                            tz="UTC",
                        ),
                        "entry_time_utc": pd.Timestamp(
                            int(position["entry_time"]),
                            unit="s",
                            tz="UTC",
                        ),
                        "exit_time_utc": pd.Timestamp(
                            int(row["time"]),
                            unit="s",
                            tz="UTC",
                        ),
                        "entry_price": position["entry"],
                        "sl": position["sl"],
                        "tp": position["tp"],
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "stop_pct": float(position["stop_pct"]) * 100.0,
                        "target_rr": profile.rr,
                        "realized_r_after_fee": realized_r,
                        "equity_before_usdt": equity_before,
                        "notional_usdt": position["notional"],
                        "margin_used_usdt": position["margin"],
                        "gross_pnl_usdt": gross,
                        "fees_usdt": trade_fee,
                        "net_pnl_usdt": net,
                        "equity_after_usdt": equity,
                        "trade_return_pct": trade_return,
                        "hold_bars": hold_bars,
                    }
                )

                position = None
                last_exit_i = i

                # Practical near-zero stop for a 5 USDT experiment.
                if equity <= 0.05:
                    break

        # -------------------------------------------------------------
        # Search for a new signal.
        # -------------------------------------------------------------
        if position is None:
            if (
                last_exit_i is not None
                and i - last_exit_i < profile.cooldown_bars
            ):
                i += 1
                continue

            # Need next candle OPEN for entry.
            if i + 1 > last_i:
                break

            previous = df.iloc[i - 1]
            signal = row
            next_bar = df.iloc[i + 1]

            required = [
                "ema20",
                "ema50",
                "atr14",
                "atr_pct",
                "vol_avg20",
                "ht_close",
                "ht_fast",
                "ht_slow",
                "ht_adx14",
            ]
            if any(pd.isna(signal.get(column)) for column in required):
                i += 1
                continue

            if pd.isna(previous.get("ema20")) or pd.isna(previous.get("atr14")):
                i += 1
                continue

            atr_pct_value = float(signal["atr_pct"])
            if not (
                profile.min_atr_pct
                <= atr_pct_value
                <= profile.max_atr_pct
            ):
                i += 1
                continue

            volume_average = float(signal["vol_avg20"])
            if volume_average <= 0:
                i += 1
                continue

            if float(signal["volume"]) < volume_average * profile.volume_mult:
                i += 1
                continue

            ht_fast = float(signal["ht_fast"])
            ht_slow = float(signal["ht_slow"])
            ht_close = float(signal["ht_close"])
            ht_adx = float(signal["ht_adx14"])

            previous_ema20 = float(previous["ema20"])
            previous_atr = float(previous["atr14"])
            signal_ema20 = float(signal["ema20"])

            trend_long = (
                ht_fast > ht_slow
                and ht_close > ht_fast
                and ht_adx >= profile.adx_min
            )
            trend_short = (
                ht_fast < ht_slow
                and ht_close < ht_fast
                and ht_adx >= profile.adx_min
            )

            pullback_long = (
                float(previous["low"])
                <= previous_ema20
                + previous_atr * profile.pullback_atr
                and float(previous["low"])
                >= float(previous["ema50"])
                - previous_atr * profile.deep_pullback_atr
            )
            pullback_short = (
                float(previous["high"])
                >= previous_ema20
                - previous_atr * profile.pullback_atr
                and float(previous["high"])
                <= float(previous["ema50"])
                + previous_atr * profile.deep_pullback_atr
            )

            trigger_long = (
                float(signal["close"]) > signal_ema20
                and float(signal["close"]) > float(previous["high"])
                and float(signal["close"]) > float(signal["open"])
            )
            trigger_short = (
                float(signal["close"]) < signal_ema20
                and float(signal["close"]) < float(previous["low"])
                and float(signal["close"]) < float(signal["open"])
            )

            long_signal = trend_long and pullback_long and trigger_long
            short_signal = trend_short and pullback_short and trigger_short

            if direction_mode == "LONG_ONLY":
                short_signal = False
            elif direction_mode == "SHORT_ONLY":
                long_signal = False

            direction = 0
            if long_signal:
                direction = 1
            elif short_signal:
                direction = -1

            if direction != 0:
                raw_entry = float(next_bar["open"])
                entry = adverse_entry(raw_entry, direction, slippage)
                atr_now = float(signal["atr14"])

                if direction == 1:
                    swing = float(
                        df.iloc[
                            max(0, i - profile.swing_bars + 1): i + 1
                        ]["low"].min()
                    )
                    atr_stop = entry - profile.stop_atr * atr_now

                    # closer valid stop
                    stop = max(swing, atr_stop)
                    stop_distance = entry - stop
                else:
                    swing = float(
                        df.iloc[
                            max(0, i - profile.swing_bars + 1): i + 1
                        ]["high"].max()
                    )
                    atr_stop = entry + profile.stop_atr * atr_now

                    # closer valid stop
                    stop = min(swing, atr_stop)
                    stop_distance = stop - entry

                if entry <= 0 or stop_distance <= 0:
                    i += 1
                    continue

                stop_pct = stop_distance / entry

                if not (
                    profile.min_stop_pct
                    <= stop_pct
                    <= profile.max_stop_pct
                ):
                    i += 1
                    continue

                # Risk-based sizing.
                risk_budget = equity * RISK_PCT
                notional_by_risk = risk_budget / stop_pct

                # Margin safety cap.
                notional_by_margin = (
                    equity
                    * MAX_MARGIN_PCT
                    * LEVERAGE
                )

                notional = min(
                    notional_by_risk,
                    notional_by_margin,
                )

                min_notional = (
                    meta.min_vol
                    * meta.contract_size
                    * entry
                )

                if notional < min_notional:
                    i += 1
                    continue

                margin = notional / LEVERAGE
                qty = notional / entry
                entry_fee = notional * TAKER_FEE

                if direction == 1:
                    take_profit = entry + profile.rr * stop_distance
                else:
                    take_profit = entry - profile.rr * stop_distance

                position = {
                    "direction": direction,
                    "signal_time": int(signal["time"]),
                    "entry_time": int(next_bar["time"]),
                    "entry_i": i + 1,
                    "entry": entry,
                    "sl": stop,
                    "tp": take_profit,
                    "stop_pct": stop_pct,
                    "notional": notional,
                    "margin": margin,
                    "qty": qty,
                    "entry_fee": entry_fee,
                }

                stop_pct_values.append(stop_pct * 100.0)

                # Next loop processes the actual entry candle.
                i += 1
                continue

        i += 1

    trade_count = len(trades)
    profit_factor_value = profit_factor(gross_profit, gross_loss)
    net_pnl = equity - STARTING_CAPITAL
    return_pct = (
        net_pnl / STARTING_CAPITAL * 100.0
        if STARTING_CAPITAL > 0
        else 0.0
    )

    metrics = Metrics(
        trades=trade_count,
        wins=wins,
        losses=losses,
        time_exits=time_exits,
        win_rate_pct=(
            wins / trade_count * 100.0
            if trade_count
            else None
        ),
        gross_profit_usdt=gross_profit,
        gross_loss_usdt=gross_loss,
        profit_factor=profit_factor_value,
        fees_usdt=total_fees,
        net_pnl_usdt=net_pnl,
        start_equity_usdt=STARTING_CAPITAL,
        end_equity_usdt=equity,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown,
        min_equity_usdt=minimum_equity,
        expectancy_usdt=(
            net_pnl / trade_count
            if trade_count
            else None
        ),
        avg_trade_return_pct=(
            float(pd.Series(trade_returns).mean())
            if trade_returns
            else None
        ),
        median_trade_return_pct=(
            float(pd.Series(trade_returns).median())
            if trade_returns
            else None
        ),
        avg_stop_pct=(
            float(pd.Series(stop_pct_values).mean())
            if stop_pct_values
            else None
        ),
        avg_rr_realized_after_fee=(
            float(pd.Series(rr_values).mean())
            if rr_values
            else None
        ),
        longest_loss_streak=longest_streak(
            result_sequence,
            "L",
        ),
        longest_win_streak=longest_streak(
            result_sequence,
            "W",
        ),
        avg_hold_bars=(
            float(pd.Series(holding_values).mean())
            if holding_values
            else None
        ),
        long_trades=long_n,
        long_net_pnl_usdt=long_net,
        long_profit_factor=profit_factor(long_gp, long_gl),
        short_trades=short_n,
        short_net_pnl_usdt=short_net,
        short_profit_factor=profit_factor(short_gp, short_gl),
        blown_up=equity <= 0.05,
    )

    return metrics, trades


def empty_metrics() -> Metrics:
    return Metrics(
        trades=0,
        wins=0,
        losses=0,
        time_exits=0,
        win_rate_pct=None,
        gross_profit_usdt=0.0,
        gross_loss_usdt=0.0,
        profit_factor=None,
        fees_usdt=0.0,
        net_pnl_usdt=0.0,
        start_equity_usdt=STARTING_CAPITAL,
        end_equity_usdt=STARTING_CAPITAL,
        return_pct=0.0,
        max_drawdown_pct=0.0,
        min_equity_usdt=STARTING_CAPITAL,
        expectancy_usdt=None,
        avg_trade_return_pct=None,
        median_trade_return_pct=None,
        avg_stop_pct=None,
        avg_rr_realized_after_fee=None,
        longest_loss_streak=0,
        longest_win_streak=0,
        avg_hold_bars=None,
        long_trades=0,
        long_net_pnl_usdt=0.0,
        long_profit_factor=None,
        short_trades=0,
        short_net_pnl_usdt=0.0,
        short_profit_factor=None,
        blown_up=False,
    )


def safe_pf(value: Any) -> float:
    try:
        number = float(value)
        if math.isinf(number):
            return 10.0
        if math.isnan(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ticker + direction_mode.

    Primary stress scenario = 0.05% adverse slippage / side.
    Hard stress            = 0.08%
    Extreme stress         = 0.10%

    No strategy parameters are optimized here.
    """
    rows: List[Dict[str, Any]] = []

    for (ticker, direction_mode), group in summary.groupby(
        ["ticker", "direction_mode"],
        sort=False,
    ):
        def one(window: str, slip: float) -> Optional[pd.Series]:
            match = group[
                (group["window"] == window)
                & (group["slippage_per_side_pct"].round(6) == round(slip * 100.0, 6))
            ]
            if match.empty:
                return None
            return match.iloc[0]

        full003 = one("FULL90", 0.0003)
        full005 = one("FULL90", 0.0005)
        full008 = one("FULL90", 0.0008)
        full010 = one("FULL90", 0.0010)

        recent005 = one("RECENT_30D", 0.0005)

        monthly005 = [
            one("OLD_30D", 0.0005),
            one("MID_30D", 0.0005),
            one("RECENT_30D", 0.0005),
        ]
        monthly005 = [x for x in monthly005 if x is not None]

        positive_months_005 = sum(
            1
            for row in monthly005
            if float(row["return_pct"]) > 0
        )

        baseline_positive = (
            full003 is not None
            and float(full003["return_pct"]) > 0
        )
        medium_positive = (
            full005 is not None
            and float(full005["return_pct"]) > 0
        )
        hard_positive = (
            full008 is not None
            and float(full008["return_pct"]) > 0
        )
        extreme_positive = (
            full010 is not None
            and float(full010["return_pct"]) > 0
        )

        # Descriptive robustness grade, NOT an optimization target.
        grade = "D"

        if full005 is not None and recent005 is not None:
            full005_pf = safe_pf(full005["profit_factor"])
            recent005_pf = safe_pf(recent005["profit_factor"])

            common_ok = (
                int(full005["trades"]) >= 30
                and float(full005["return_pct"]) > 0
                and full005_pf >= 1.15
                and float(full005["max_drawdown_pct"]) <= 40.0
                and positive_months_005 >= 2
                and int(recent005["trades"]) >= 5
                and float(recent005["return_pct"]) > 0
                and recent005_pf >= 1.05
            )

            if common_ok:
                grade = "B"

                if (
                    hard_positive
                    and positive_months_005 == 3
                    and float(full005["max_drawdown_pct"]) <= 35.0
                ):
                    grade = "A"

            elif baseline_positive:
                grade = "C"

        # Ranking score is only for ordering after all frozen tests.
        score = -999999.0

        if full005 is not None and recent005 is not None:
            score = (
                float(full005["return_pct"]) * 0.35
                + safe_pf(full005["profit_factor"]) * 20.0
                - float(full005["max_drawdown_pct"]) * 0.55
                + float(recent005["return_pct"]) * 0.50
                + safe_pf(recent005["profit_factor"]) * 10.0
                + positive_months_005 * 12.0
                + (15.0 if hard_positive else 0.0)
                + (10.0 if extreme_positive else 0.0)
                - int(full005["longest_loss_streak"]) * 0.8
            )

        profile = group.iloc[0]["profile"]
        actual_days = group.iloc[0]["actual_data_days"]

        result = {
            "ticker": ticker,
            "profile": profile,
            "direction_mode": direction_mode,
            "robustness_grade": grade,
            "ranking_score": score,
            "actual_data_days": actual_days,
            "positive_30d_windows_at_0_05_slippage": positive_months_005,
            "full90_positive_at_0_03": baseline_positive,
            "full90_positive_at_0_05": medium_positive,
            "full90_positive_at_0_08": hard_positive,
            "full90_positive_at_0_10": extreme_positive,
        }

        for label, row in [
            ("full90_slip003", full003),
            ("full90_slip005", full005),
            ("full90_slip008", full008),
            ("full90_slip010", full010),
            ("recent30_slip005", recent005),
        ]:
            if row is None:
                continue

            for metric in [
                "trades",
                "wins",
                "losses",
                "time_exits",
                "win_rate_pct",
                "profit_factor",
                "fees_usdt",
                "net_pnl_usdt",
                "end_equity_usdt",
                "return_pct",
                "max_drawdown_pct",
                "longest_loss_streak",
                "avg_trade_return_pct",
                "avg_rr_realized_after_fee",
            ]:
                result[f"{label}_{metric}"] = row[metric]

        rows.append(result)

    ranking = pd.DataFrame(rows)

    grade_order = {
        "A": 4,
        "B": 3,
        "C": 2,
        "D": 1,
    }

    ranking["_grade_sort"] = ranking["robustness_grade"].map(grade_order).fillna(0)

    ranking = ranking.sort_values(
        by=[
            "_grade_sort",
            "ranking_score",
            "full90_slip005_return_pct",
        ],
        ascending=[False, False, False],
        na_position="last",
    ).drop(columns=["_grade_sort"])

    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-profile final validation for 4 MEXC candidates"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Kept for explicit workflow logging; validation expects 90.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.days != 90:
        raise SystemExit("Final validation is intentionally frozen to 90 days.")

    contracts = load_candidate_contracts()

    summary_rows: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print("=" * 80)
    print("MEXC FINAL VALIDATION")
    print("=" * 80)
    print(f"Capital: {STARTING_CAPITAL} USDT")
    print(f"Leverage: {LEVERAGE}x")
    print(f"Risk target: {RISK_PCT*100:.1f}% equity/trade")
    print(f"Max margin: {MAX_MARGIN_PCT*100:.1f}% equity")
    print(f"Taker fee: {TAKER_FEE*100:.3f}% / side")
    print("Profiles are FROZEN. No optimization.")
    print()

    for ticker, profile in CANDIDATES.items():
        print(f"\n--- {ticker} | {profile.name} ---", flush=True)

        try:
            meta = contracts.get(ticker)

            if meta is None:
                raise RuntimeError("Ticker not found in active MEXC contract detail")

            if meta.quote_coin != "USDT":
                raise RuntimeError(f"Unexpected quote coin: {meta.quote_coin}")

            if meta.state != 0:
                raise RuntimeError(f"Contract is not active: state={meta.state}")

            if not meta.api_allowed:
                raise RuntimeError("MEXC API trading not allowed for this contract")

            if meta.max_leverage is not None and meta.max_leverage < int(LEVERAGE):
                raise RuntimeError(
                    f"Contract max leverage {meta.max_leverage} < required {LEVERAGE}x"
                )

            raw = fetch_5m(ticker, DAYS)

            if raw.empty:
                raise RuntimeError("No 5m candles")

            requested_end = int(raw["time"].max())
            requested_start = requested_end - DAYS * 86400

            actual_90 = raw[raw["time"] >= requested_start].copy()
            if actual_90.empty:
                raise RuntimeError("No candles inside requested 90d")

            actual_start = int(actual_90["time"].min())
            actual_end = int(actual_90["time"].max())
            actual_days = (actual_end - actual_start) / 86400.0

            # We now demand a genuinely long history.
            if actual_days < 88.0:
                raise RuntimeError(
                    f"Insufficient real history for final validation: {actual_days:.1f} days"
                )

            data = enrich(raw, profile)

            # Use exact 90-day boundaries based on the newest completed MEXC bar.
            full_end = int(data["time"].max())
            full_start = full_end - 90 * 86400

            old_start = full_start
            old_end = full_start + 30 * 86400 - BASE_SECONDS

            mid_start = full_start + 30 * 86400
            mid_end = full_start + 60 * 86400 - BASE_SECONDS

            recent_start = full_start + 60 * 86400
            recent_end = full_end

            windows = [
                ("OLD_30D", old_start, old_end),
                ("MID_30D", mid_start, mid_end),
                ("RECENT_30D", recent_start, recent_end),
                ("FULL90", full_start, full_end),
            ]

            for direction_mode in DIRECTION_MODES:
                for slippage in SLIPPAGE_SCENARIOS:
                    for window_name, start_ts, end_ts in windows:
                        metrics, trades = simulate(
                            df=data,
                            profile=profile,
                            meta=meta,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            window_name=window_name,
                            direction_mode=direction_mode,
                            slippage=slippage,
                        )

                        row = {
                            "ticker": ticker,
                            "profile": profile.name,
                            "direction_mode": direction_mode,
                            "slippage_per_side_pct": slippage * 100.0,
                            "window": window_name,
                            "actual_data_days": round(actual_days, 2),
                            "window_start_utc": pd.Timestamp(
                                start_ts,
                                unit="s",
                                tz="UTC",
                            ),
                            "window_end_utc": pd.Timestamp(
                                end_ts,
                                unit="s",
                                tz="UTC",
                            ),
                            "starting_capital_usdt": STARTING_CAPITAL,
                            "leverage": LEVERAGE,
                            "risk_per_trade_pct": RISK_PCT * 100.0,
                            "max_margin_pct_of_equity": MAX_MARGIN_PCT * 100.0,
                            "taker_fee_per_side_pct": TAKER_FEE * 100.0,
                            "target_rr": profile.rr,
                            **asdict(metrics),
                        }

                        summary_rows.append(row)
                        all_trades.extend(trades)

                        print(
                            f"{direction_mode:10s} | "
                            f"slip={slippage*100:.2f}% | "
                            f"{window_name:10s} | "
                            f"n={metrics.trades:3d} | "
                            f"PF={metrics.profit_factor} | "
                            f"ret={metrics.return_pct:+7.2f}% | "
                            f"DD={metrics.max_drawdown_pct:6.2f}%",
                            flush=True,
                        )

        except Exception as exc:
            errors.append(
                {
                    "ticker": ticker,
                    "profile": profile.name,
                    "error": repr(exc),
                }
            )
            print(f"ERROR {ticker}: {exc}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("validation_summary.csv", index=False)

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv("validation_trades.csv", index=False)

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv("validation_errors.csv", index=False)

    if summary.empty:
        ranking = pd.DataFrame()
    else:
        ranking = build_ranking(summary)

    ranking.to_csv("validation_ranking.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL ROBUSTNESS RANKING")
    print("=" * 80)

    if ranking.empty:
        print("No ranking rows.")
    else:
        show = [
            "ticker",
            "profile",
            "direction_mode",
            "robustness_grade",
            "ranking_score",
            "positive_30d_windows_at_0_05_slippage",
            "full90_positive_at_0_03",
            "full90_positive_at_0_05",
            "full90_positive_at_0_08",
            "full90_positive_at_0_10",
            "full90_slip005_trades",
            "full90_slip005_profit_factor",
            "full90_slip005_return_pct",
            "full90_slip005_max_drawdown_pct",
            "recent30_slip005_trades",
            "recent30_slip005_profit_factor",
            "recent30_slip005_return_pct",
        ]

        available = [c for c in show if c in ranking.columns]
        print(ranking[available].to_string(index=False))

    print("\nSaved:")
    print("  validation_ranking.csv")
    print("  validation_summary.csv")
    print("  validation_trades.csv")
    print("  validation_errors.csv")


if __name__ == "__main__":
    main()
