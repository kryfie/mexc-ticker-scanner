#!/usr/bin/env python3
"""
MEXC ROBUST EDGE SCANNER
========================

Purpose
-------
Scan active MEXC USDT perpetual contracts over the most recent 90 days and find
candidates for a live bot using a deliberately conservative, non-lookahead
trend-pullback-breakout strategy.

This scanner is designed for RESEARCH, not to guarantee profit.

Core design
-----------
Base timeframe: 5m
Higher timeframe: 15m, derived from 5m candles.

LONG:
1. 15m trend: EMA fast > EMA slow, close > EMA fast, ADX >= threshold.
2. Previous 5m candle pulls back toward EMA20 without breaking too deeply.
3. Current 5m candle closes bullish, above EMA20, and above previous high.
4. Volume is at least rolling-volume-average * profile threshold.
5. ATR% is inside a reasonable liquidity/volatility band.
6. Entry is NEXT candle OPEN (prevents lookahead).
7. Stop = closer of recent swing low and ATR stop cap.
8. TP = fixed R multiple.
9. Position sizing targets fixed % equity risk, capped by max isolated margin.

SHORT is exactly mirrored.

Execution assumptions
---------------------
- Starting capital: 5 USDT by default.
- Isolated-style sizing model.
- Default leverage: 10x.
- Target risk: 4% of current equity per trade.
- Max margin committed: 35% of current equity.
- Taker fee: 0.08% per side (0.0008).
- Default modeled slippage: 0.03% per side.
- One position at a time per ticker.
- If TP and SL are both touched in the same candle, SL wins (conservative).
- Maximum holding time: profile-specific number of 5m bars.
- Time exit uses adverse modeled slippage.

Anti-overfitting / validation
-----------------------------
The 90-day period is split chronologically:
- First 60 days: TRAIN. Multiple predefined strategy profiles are compared.
- Last 30 days: OUT-OF-SAMPLE (OOS). The best TRAIN profile is frozen and tested.
- Ranking prioritizes OOS performance and stability, not full-period return.

Outputs
-------
ranking_robust_edge.csv
    One row per MEXC ticker with train/OOS/full metrics and chosen profile.

trades_robust_edge.csv
    Every simulated trade for the chosen profile, with split=train/oos.

profile_train_results.csv
    Training result for every tested profile and ticker.

errors_robust_edge.csv
    Symbols that could not be evaluated.

No API key is required: only public MEXC futures endpoints are used.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


BASE_URL = "https://contract.mexc.com"
TIMEOUT = 30
BASE_INTERVAL = "Min5"
BASE_SECONDS = 5 * 60

# MEXC public kline endpoint returns at most a limited number of points per call.
CHUNK_POINTS = 1800


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


# Small, intentional profile set. This is not a giant brute-force optimizer.
# The point is to test nearby sensible variants while limiting overfit.
PROFILES: List[Profile] = [
    Profile(
        "BALANCED_22R",
        ht_fast=40, ht_slow=120, adx_min=18,
        volume_mult=1.00, pullback_atr=0.55, deep_pullback_atr=0.70,
        stop_atr=1.50, swing_bars=3, rr=2.2,
        min_stop_pct=0.0035, max_stop_pct=0.025,
        min_atr_pct=0.0020, max_atr_pct=0.050,
        cooldown_bars=3, max_hold_bars=48,
    ),
    Profile(
        "BALANCED_25R",
        ht_fast=40, ht_slow=120, adx_min=20,
        volume_mult=1.05, pullback_atr=0.50, deep_pullback_atr=0.65,
        stop_atr=1.45, swing_bars=3, rr=2.5,
        min_stop_pct=0.0035, max_stop_pct=0.023,
        min_atr_pct=0.0020, max_atr_pct=0.045,
        cooldown_bars=3, max_hold_bars=48,
    ),
    Profile(
        "MOMENTUM_28R",
        ht_fast=30, ht_slow=100, adx_min=23,
        volume_mult=1.10, pullback_atr=0.45, deep_pullback_atr=0.60,
        stop_atr=1.35, swing_bars=3, rr=2.8,
        min_stop_pct=0.0035, max_stop_pct=0.022,
        min_atr_pct=0.0025, max_atr_pct=0.050,
        cooldown_bars=4, max_hold_bars=42,
    ),
    Profile(
        "TREND_30R",
        ht_fast=50, ht_slow=150, adx_min=22,
        volume_mult=1.00, pullback_atr=0.60, deep_pullback_atr=0.75,
        stop_atr=1.55, swing_bars=4, rr=3.0,
        min_stop_pct=0.0040, max_stop_pct=0.027,
        min_atr_pct=0.0020, max_atr_pct=0.045,
        cooldown_bars=4, max_hold_bars=60,
    ),
    Profile(
        "SMOOTH_20R",
        ht_fast=50, ht_slow=150, adx_min=18,
        volume_mult=0.95, pullback_atr=0.65, deep_pullback_atr=0.80,
        stop_atr=1.75, swing_bars=4, rr=2.0,
        min_stop_pct=0.0040, max_stop_pct=0.030,
        min_atr_pct=0.0018, max_atr_pct=0.040,
        cooldown_bars=3, max_hold_bars=60,
    ),
    Profile(
        "SELECTIVE_32R",
        ht_fast=30, ht_slow=120, adx_min=26,
        volume_mult=1.20, pullback_atr=0.40, deep_pullback_atr=0.55,
        stop_atr=1.30, swing_bars=3, rr=3.2,
        min_stop_pct=0.0035, max_stop_pct=0.020,
        min_atr_pct=0.0025, max_atr_pct=0.045,
        cooldown_bars=5, max_hold_bars=36,
    ),
]


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
    expectancy_usdt: Optional[float]
    expectancy_pct_of_start: Optional[float]
    avg_trade_return_pct: Optional[float]
    median_trade_return_pct: Optional[float]
    avg_stop_pct: Optional[float]
    avg_rr_realized: Optional[float]
    longest_loss_streak: int
    longest_win_streak: int
    avg_hold_bars: Optional[float]
    long_trades: int
    long_net_pnl_usdt: float
    long_profit_factor: Optional[float]
    short_trades: int
    short_net_pnl_usdt: float
    short_profit_factor: Optional[float]
    min_equity_usdt: float
    blown_up: bool


def mexc_get(path: str, params: Optional[dict] = None, retries: int = 4) -> dict:
    url = BASE_URL + path
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                raise RuntimeError(f"MEXC API returned success=false: {payload}")
            return payload
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 * attempt)
    assert last_exc is not None
    raise last_exc


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_contracts(min_leverage: int) -> Dict[str, ContractMeta]:
    payload = mexc_get("/api/v1/contract/detail")
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    out: Dict[str, ContractMeta] = {}

    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue

        quote = str(row.get("quoteCoin") or "").upper()
        state = inum(row.get("state"))
        api_allowed = bool(row.get("apiAllowed", True))
        max_lev = inum(row.get("maxLeverage"))
        if max_lev is None:
            max_lev = max(
                [
                    x for x in [
                        inum(row.get("maxLongLeverage")),
                        inum(row.get("maxShortLeverage")),
                        inum(row.get("maxLever")),
                    ] if x is not None
                ] or [0]
            )

        if quote != "USDT":
            continue
        if state != 0:
            continue
        if not api_allowed:
            continue
        if max_lev < min_leverage:
            continue

        out[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=max_lev,
            contract_size=max(fnum(row.get("contractSize"), 1.0), 1e-12),
            min_vol=max(fnum(row.get("minVol"), 1.0), 1e-12),
            state=state,
            quote_coin=quote,
            api_allowed=api_allowed,
        )
    return out


def fetch_klines(symbol: str, days: int) -> pd.DataFrame:
    end = int(time.time())
    start = end - days * 86400
    cursor = start
    chunk_span = CHUNK_POINTS * BASE_SECONDS
    frames: List[pd.DataFrame] = []

    while cursor < end:
        chunk_end = min(end, cursor + chunk_span - BASE_SECONDS)
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
            vol = data.get("vol") or data.get("amount") or [0.0] * n
            if len(vol) != n:
                vol = [0.0] * n

            frames.append(
                pd.DataFrame(
                    {
                        "time": data["time"],
                        "open": data.get("open", [None] * n),
                        "high": data.get("high", [None] * n),
                        "low": data.get("low", [None] * n),
                        "close": data.get("close", [None] * n),
                        "volume": vol,
                    }
                )
            )
        cursor = chunk_end + BASE_SECONDS

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for c in ["time", "open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df["time"] = df["time"].astype("int64")

    # Remove still-forming candle.
    now = int(time.time())
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
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    a = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / a

    denom = (plus_di + minus_di).replace(0, math.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def enrich(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty:
        return df5

    out = df5.copy()
    out["dt"] = pd.to_datetime(out["time"], unit="s", utc=True)

    # 5m indicators
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"]
    out["vol_avg20"] = out["volume"].rolling(20, min_periods=20).mean()

    # 15m bars, closed-bar values only.
    base = out.set_index("dt")
    df15 = (
        base[["open", "high", "low", "close", "volume"]]
        .resample("15min", label="right", closed="right")
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

    # Compute the longest indicator values we need once.
    for span in sorted({p.ht_fast for p in PROFILES} | {p.ht_slow for p in PROFILES}):
        df15[f"ema_{span}"] = ema(df15["close"], span)
    df15["adx14"] = adx(df15, 14)

    ht_cols = ["close", "adx14"] + [f"ema_{s}" for s in sorted({p.ht_fast for p in PROFILES} | {p.ht_slow for p in PROFILES})]
    ht = df15[ht_cols].copy()
    ht.columns = [f"ht_{c}" for c in ht.columns]

    out = pd.merge_asof(
        out.sort_values("dt"),
        ht.sort_index().reset_index(),
        on="dt",
        direction="backward",
        allow_exact_matches=True,
    )
    return out.reset_index(drop=True)


def adverse_entry_price(raw_open: float, direction: int, slippage: float) -> float:
    return raw_open * (1 + slippage) if direction == 1 else raw_open * (1 - slippage)


def adverse_exit_price(raw_exit: float, direction: int, slippage: float) -> float:
    return raw_exit * (1 - slippage) if direction == 1 else raw_exit * (1 + slippage)


def pf(gross_profit: float, gross_loss: float) -> Optional[float]:
    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return math.inf
    return None


def longest_streak(sequence: List[str], wanted: str) -> int:
    best = current = 0
    for item in sequence:
        if item == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def simulate(
    df: pd.DataFrame,
    profile: Profile,
    meta: ContractMeta,
    start_equity: float,
    leverage: float,
    risk_pct: float,
    max_margin_pct: float,
    fee: float,
    slippage: float,
    start_ts: int,
    end_ts: int,
    split_name: str,
) -> Tuple[Metrics, List[Dict[str, Any]]]:
    """
    Independent simulation for one time window.
    Equity resets to start_equity at the beginning of each window.
    """
    equity = start_equity
    peak = start_equity
    min_equity = start_equity
    max_dd = 0.0
    position: Optional[Dict[str, Any]] = None
    last_exit_i: Optional[int] = None

    wins = losses = time_exits = 0
    gp = gl = fees_total = 0.0
    trade_returns: List[float] = []
    stop_pcts: List[float] = []
    rr_realized_vals: List[float] = []
    hold_bars_vals: List[int] = []
    seq: List[str] = []

    long_gp = long_gl = long_net = 0.0
    short_gp = short_gl = short_net = 0.0
    long_n = short_n = 0

    trades: List[Dict[str, Any]] = []

    ht_fast_col = f"ht_ema_{profile.ht_fast}"
    ht_slow_col = f"ht_ema_{profile.ht_slow}"

    start_indices = df.index[df["time"] >= start_ts].tolist()
    end_indices = df.index[df["time"] <= end_ts].tolist()
    if not start_indices or not end_indices:
        return Metrics(
            0,0,0,0,None,0.0,0.0,None,0.0,0.0,
            start_equity,start_equity,0.0,0.0,None,None,None,None,None,None,
            0,0,None,0,0.0,None,0,0.0,None,start_equity,False
        ), []

    first_i = max(55, start_indices[0], profile.swing_bars + 2)
    last_i = min(len(df) - 2, end_indices[-1])

    i = first_i
    while i <= last_i:
        row = df.iloc[i]

        # --------------------------------------------------------------
        # Manage open trade
        # --------------------------------------------------------------
        if position is not None:
            d = position["direction"]
            hit_sl = row["low"] <= position["sl"] if d == 1 else row["high"] >= position["sl"]
            hit_tp = row["high"] >= position["tp"] if d == 1 else row["low"] <= position["tp"]

            exit_reason = None
            raw_exit = None

            if hit_sl:
                exit_reason = "SL"
                raw_exit = position["sl"]
            elif hit_tp:
                exit_reason = "TP"
                raw_exit = position["tp"]
            elif i - position["entry_i"] >= profile.max_hold_bars:
                exit_reason = "TIME"
                raw_exit = row["close"]

            if exit_reason is not None:
                exit_px = adverse_exit_price(float(raw_exit), d, slippage)
                qty = position["qty"]
                gross = qty * (exit_px - position["entry"]) * d
                exit_fee = abs(qty * exit_px) * fee
                total_fee = position["entry_fee"] + exit_fee
                net = gross - total_fee

                equity_before = equity
                equity += net
                fees_total += total_fee
                gp += gross if gross > 0 else 0.0
                gl += gross if gross < 0 else 0.0

                trade_ret = (net / equity_before * 100.0) if equity_before > 0 else -100.0
                trade_returns.append(trade_ret)
                hold_bars = i - position["entry_i"]
                hold_bars_vals.append(hold_bars)

                risk_dollars = position["notional"] * position["stop_pct"]
                rr_realized = net / risk_dollars if risk_dollars > 0 else 0.0
                rr_realized_vals.append(rr_realized)

                if exit_reason == "TP":
                    wins += 1
                    seq.append("W")
                elif exit_reason == "SL":
                    losses += 1
                    seq.append("L")
                else:
                    time_exits += 1
                    seq.append("W" if net > 0 else "L")

                if d == 1:
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
                min_equity = min(min_equity, equity)
                dd = (peak - equity) / peak * 100.0 if peak > 0 else 100.0
                max_dd = max(max_dd, dd)

                trades.append(
                    {
                        "ticker": meta.ticker,
                        "split": split_name,
                        "profile": profile.name,
                        "direction": "LONG" if d == 1 else "SHORT",
                        "signal_time_utc": pd.Timestamp(position["signal_time"], unit="s", tz="UTC"),
                        "entry_time_utc": pd.Timestamp(position["entry_time"], unit="s", tz="UTC"),
                        "exit_time_utc": pd.Timestamp(int(row["time"]), unit="s", tz="UTC"),
                        "entry_price": position["entry"],
                        "sl": position["sl"],
                        "tp": position["tp"],
                        "exit_price": exit_px,
                        "exit_reason": exit_reason,
                        "stop_pct": position["stop_pct"] * 100.0,
                        "rr_target": profile.rr,
                        "rr_realized_after_fee": rr_realized,
                        "equity_before_usdt": equity_before,
                        "notional_usdt": position["notional"],
                        "margin_used_usdt": position["margin"],
                        "gross_pnl_usdt": gross,
                        "fees_usdt": total_fee,
                        "net_pnl_usdt": net,
                        "equity_after_usdt": equity,
                        "trade_return_pct": trade_ret,
                        "hold_bars": hold_bars,
                    }
                )

                position = None
                last_exit_i = i

                if equity <= 0.05:
                    break

        # --------------------------------------------------------------
        # Entry signal (if flat)
        # --------------------------------------------------------------
        if position is None:
            if last_exit_i is not None and (i - last_exit_i) < profile.cooldown_bars:
                i += 1
                continue

            # Need next candle for non-lookahead entry.
            if i + 1 > last_i:
                break

            prev = df.iloc[i - 1]
            sig = row
            nxt = df.iloc[i + 1]

            required = [
                ht_fast_col, ht_slow_col, "ht_close", "ht_adx14",
                "ema20", "ema50", "atr14", "atr_pct", "vol_avg20",
            ]
            if any(pd.isna(sig.get(c)) for c in required):
                i += 1
                continue
            if pd.isna(prev.get("ema20")) or pd.isna(prev.get("atr14")):
                i += 1
                continue

            atr_pct = float(sig["atr_pct"])
            if not (profile.min_atr_pct <= atr_pct <= profile.max_atr_pct):
                i += 1
                continue

            vol_avg = float(sig["vol_avg20"])
            if vol_avg <= 0:
                i += 1
                continue
            if float(sig["volume"]) < vol_avg * profile.volume_mult:
                i += 1
                continue

            ht_fast = float(sig[ht_fast_col])
            ht_slow = float(sig[ht_slow_col])
            ht_close = float(sig["ht_close"])
            ht_adx = float(sig["ht_adx14"])

            prev_ema20 = float(prev["ema20"])
            prev_atr = float(prev["atr14"])
            sig_ema20 = float(sig["ema20"])
            sig_ema50 = float(sig["ema50"])

            trend_long = ht_fast > ht_slow and ht_close > ht_fast and ht_adx >= profile.adx_min
            trend_short = ht_fast < ht_slow and ht_close < ht_fast and ht_adx >= profile.adx_min

            pull_long = (
                float(prev["low"]) <= prev_ema20 + prev_atr * profile.pullback_atr
                and float(prev["low"]) >= float(prev["ema50"]) - prev_atr * profile.deep_pullback_atr
            )
            pull_short = (
                float(prev["high"]) >= prev_ema20 - prev_atr * profile.pullback_atr
                and float(prev["high"]) <= float(prev["ema50"]) + prev_atr * profile.deep_pullback_atr
            )

            trigger_long = (
                float(sig["close"]) > sig_ema20
                and float(sig["close"]) > float(prev["high"])
                and float(sig["close"]) > float(sig["open"])
            )
            trigger_short = (
                float(sig["close"]) < sig_ema20
                and float(sig["close"]) < float(prev["low"])
                and float(sig["close"]) < float(sig["open"])
            )

            direction = 0
            if trend_long and pull_long and trigger_long:
                direction = 1
            elif trend_short and pull_short and trigger_short:
                direction = -1

            if direction != 0:
                raw_entry = float(nxt["open"])
                entry = adverse_entry_price(raw_entry, direction, slippage)
                atr_now = float(sig["atr14"])

                if direction == 1:
                    swing = float(df.iloc[max(0, i - profile.swing_bars + 1): i + 1]["low"].min())
                    atr_stop = entry - profile.stop_atr * atr_now
                    sl = max(swing, atr_stop)
                    stop_dist = entry - sl
                else:
                    swing = float(df.iloc[max(0, i - profile.swing_bars + 1): i + 1]["high"].max())
                    atr_stop = entry + profile.stop_atr * atr_now
                    sl = min(swing, atr_stop)
                    stop_dist = sl - entry

                if stop_dist <= 0 or entry <= 0:
                    i += 1
                    continue

                stop_pct = stop_dist / entry
                if not (profile.min_stop_pct <= stop_pct <= profile.max_stop_pct):
                    i += 1
                    continue

                # Risk-based sizing, then margin cap.
                risk_budget = equity * risk_pct
                notional_by_risk = risk_budget / stop_pct
                notional_by_margin = equity * max_margin_pct * leverage
                notional = min(notional_by_risk, notional_by_margin)

                # MEXC minimum theoretical notional for 1 minimum-volume contract.
                min_notional = meta.min_vol * meta.contract_size * entry
                if notional < min_notional:
                    i += 1
                    continue

                margin = notional / leverage
                qty = notional / entry
                entry_fee = notional * fee

                if direction == 1:
                    tp = entry + profile.rr * stop_dist
                else:
                    tp = entry - profile.rr * stop_dist

                position = {
                    "direction": direction,
                    "signal_time": int(sig["time"]),
                    "entry_time": int(nxt["time"]),
                    "entry_i": i + 1,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "stop_pct": stop_pct,
                    "notional": notional,
                    "margin": margin,
                    "qty": qty,
                    "entry_fee": entry_fee,
                }

                stop_pcts.append(stop_pct * 100.0)

                # Start evaluating from entry candle.
                i += 1
                continue

        i += 1

    trades_n = len(trades)
    win_rate = wins / trades_n * 100.0 if trades_n else None
    profit_factor = pf(gp, gl)
    net_pnl = equity - start_equity
    ret = net_pnl / start_equity * 100.0 if start_equity > 0 else 0.0
    expectancy = net_pnl / trades_n if trades_n else None
    expectancy_pct = expectancy / start_equity * 100.0 if expectancy is not None and start_equity > 0 else None

    metrics = Metrics(
        trades=trades_n,
        wins=wins,
        losses=losses,
        time_exits=time_exits,
        win_rate_pct=win_rate,
        gross_profit_usdt=gp,
        gross_loss_usdt=gl,
        profit_factor=profit_factor,
        fees_usdt=fees_total,
        net_pnl_usdt=net_pnl,
        start_equity_usdt=start_equity,
        end_equity_usdt=equity,
        return_pct=ret,
        max_drawdown_pct=max_dd,
        expectancy_usdt=expectancy,
        expectancy_pct_of_start=expectancy_pct,
        avg_trade_return_pct=float(pd.Series(trade_returns).mean()) if trade_returns else None,
        median_trade_return_pct=float(pd.Series(trade_returns).median()) if trade_returns else None,
        avg_stop_pct=float(pd.Series(stop_pcts).mean()) if stop_pcts else None,
        avg_rr_realized=float(pd.Series(rr_realized_vals).mean()) if rr_realized_vals else None,
        longest_loss_streak=longest_streak(seq, "L"),
        longest_win_streak=longest_streak(seq, "W"),
        avg_hold_bars=float(pd.Series(hold_bars_vals).mean()) if hold_bars_vals else None,
        long_trades=long_n,
        long_net_pnl_usdt=long_net,
        long_profit_factor=pf(long_gp, long_gl),
        short_trades=short_n,
        short_net_pnl_usdt=short_net,
        short_profit_factor=pf(short_gp, short_gl),
        min_equity_usdt=min_equity,
        blown_up=equity <= 0.05,
    )
    return metrics, trades


def metric_score(m: Metrics, min_trades: int) -> float:
    """
    Training profile selection score.
    Rewards positive expectancy and PF, penalizes drawdown and tiny sample sizes.
    """
    if m.trades < min_trades:
        return -1e9
    if m.blown_up:
        return -1e9
    if m.net_pnl_usdt <= 0:
        return -1e6 + m.net_pnl_usdt

    pf_value = m.profit_factor
    if pf_value is None:
        pf_value = 0.0
    elif math.isinf(pf_value):
        pf_value = 4.0
    pf_value = min(float(pf_value), 4.0)

    return (
        m.return_pct * 0.30
        + pf_value * 18.0
        - m.max_drawdown_pct * 0.45
        + min(m.trades, 80) * 0.25
        + (m.expectancy_pct_of_start or 0.0) * 1.5
        - m.longest_loss_streak * 0.8
    )


def oos_rank_score(train: Metrics, oos: Metrics, full: Metrics, min_oos_trades: int) -> float:
    """
    Final candidate score.
    OOS dominates. Training only provides a small stability bonus.
    """
    if oos.trades < min_oos_trades:
        return -1e9
    if oos.net_pnl_usdt <= 0 or oos.blown_up:
        return -1e8 + oos.net_pnl_usdt

    oos_pf = oos.profit_factor
    if oos_pf is None:
        oos_pf = 0.0
    elif math.isinf(oos_pf):
        oos_pf = 4.0
    oos_pf = min(float(oos_pf), 4.0)

    train_pf = train.profit_factor or 0.0
    if math.isinf(train_pf):
        train_pf = 4.0

    full_pf = full.profit_factor or 0.0
    if math.isinf(full_pf):
        full_pf = 4.0

    stability_bonus = 0.0
    if train.net_pnl_usdt > 0 and float(train_pf) > 1.0:
        stability_bonus += 12.0
    if full.net_pnl_usdt > 0 and float(full_pf) > 1.0:
        stability_bonus += 8.0

    return (
        oos.return_pct * 0.55
        + oos_pf * 25.0
        - oos.max_drawdown_pct * 0.70
        + min(oos.trades, 50) * 0.50
        + (oos.expectancy_pct_of_start or 0.0) * 2.0
        - oos.longest_loss_streak * 1.2
        + stability_bonus
    )


def flatten_metrics(prefix: str, m: Metrics) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in asdict(m).items()}


def run_symbol(
    symbol: str,
    meta: ContractMeta,
    days: int,
    start_equity: float,
    leverage: float,
    risk_pct: float,
    max_margin_pct: float,
    fee: float,
    slippage: float,
    train_days: int,
    min_train_trades: int,
    min_oos_trades: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = fetch_klines(symbol, days)
    if raw.empty or len(raw) < 1000:
        raise RuntimeError("Not enough 5m candles")

    data = enrich(raw)
    actual_start = int(data["time"].min())
    actual_end = int(data["time"].max())
    actual_days = (actual_end - actual_start) / 86400.0

    # Require substantial real history; requested 90 days can be shorter for new listings.
    if actual_days < max(45.0, days * 0.65):
        raise RuntimeError(f"Insufficient real history: {actual_days:.1f} days")

    split_ts = actual_end - (days - train_days) * 86400
    train_start = actual_end - days * 86400
    train_end = split_ts - BASE_SECONDS
    oos_start = split_ts
    oos_end = actual_end

    profile_rows: List[Dict[str, Any]] = []
    profile_results: List[Tuple[Profile, Metrics]] = []

    for profile in PROFILES:
        train_m, _ = simulate(
            data, profile, meta,
            start_equity=start_equity,
            leverage=leverage,
            risk_pct=risk_pct,
            max_margin_pct=max_margin_pct,
            fee=fee,
            slippage=slippage,
            start_ts=train_start,
            end_ts=train_end,
            split_name="TRAIN",
        )
        score = metric_score(train_m, min_train_trades)
        row = {
            "ticker": symbol,
            "profile": profile.name,
            "train_score": score,
            **flatten_metrics("train", train_m),
        }
        profile_rows.append(row)
        profile_results.append((profile, train_m))

    profile_results.sort(key=lambda x: metric_score(x[1], min_train_trades), reverse=True)
    chosen_profile, chosen_train = profile_results[0]

    # If no profile satisfies the training threshold, keep the best but candidate
    # will naturally rank very poorly.
    oos_m, oos_trades = simulate(
        data, chosen_profile, meta,
        start_equity=start_equity,
        leverage=leverage,
        risk_pct=risk_pct,
        max_margin_pct=max_margin_pct,
        fee=fee,
        slippage=slippage,
        start_ts=oos_start,
        end_ts=oos_end,
        split_name="OOS",
    )

    full_m, full_trades = simulate(
        data, chosen_profile, meta,
        start_equity=start_equity,
        leverage=leverage,
        risk_pct=risk_pct,
        max_margin_pct=max_margin_pct,
        fee=fee,
        slippage=slippage,
        start_ts=train_start,
        end_ts=actual_end,
        split_name="FULL90",
    )

    rank_score = oos_rank_score(chosen_train, oos_m, full_m, min_oos_trades)

    candidate = (
        chosen_train.trades >= min_train_trades
        and chosen_train.net_pnl_usdt > 0
        and (chosen_train.profit_factor or 0) > 1.05
        and oos_m.trades >= min_oos_trades
        and oos_m.net_pnl_usdt > 0
        and (oos_m.profit_factor or 0) > 1.15
        and oos_m.max_drawdown_pct <= 45.0
        and full_m.net_pnl_usdt > 0
        and not oos_m.blown_up
    )

    row = {
        "ticker": symbol,
        "candidate": candidate,
        "rank_score": rank_score,
        "chosen_profile": chosen_profile.name,
        "requested_days": days,
        "actual_data_days": round(actual_days, 2),
        "data_start_utc": pd.Timestamp(actual_start, unit="s", tz="UTC"),
        "data_end_utc": pd.Timestamp(actual_end, unit="s", tz="UTC"),
        "train_days": train_days,
        "oos_days": days - train_days,
        "starting_capital_usdt": start_equity,
        "leverage": leverage,
        "risk_per_trade_pct": risk_pct * 100.0,
        "max_margin_pct_of_equity": max_margin_pct * 100.0,
        "fee_per_side_pct": fee * 100.0,
        "modeled_slippage_per_side_pct": slippage * 100.0,
        "max_leverage_mexc": meta.max_leverage,
        **flatten_metrics("train", chosen_train),
        **flatten_metrics("oos", oos_m),
        **flatten_metrics("full90", full_m),
    }

    # Keep only FULL90 trade log to make artifact smaller; split is still marked.
    # Add OOS separately so the analyst can inspect untouched recent trades easily.
    trades = full_trades + oos_trades
    return row, trades, profile_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust 90-day MEXC edge scanner")

    p.add_argument("--days", type=int, default=90)
    p.add_argument("--train-days", type=int, default=60)

    p.add_argument("--starting-capital", type=float, default=5.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--risk-pct", type=float, default=0.04)
    p.add_argument("--max-margin-pct", type=float, default=0.35)
    p.add_argument("--fee", type=float, default=0.0008)
    p.add_argument("--slippage", type=float, default=0.0003)

    p.add_argument("--min-train-trades", type=int, default=18)
    p.add_argument("--min-oos-trades", type=int, default=6)
    p.add_argument("--min-max-leverage", type=int, default=10)

    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--symbols", nargs="*", default=None)

    p.add_argument("--ranking-output", default="ranking_robust_edge.csv")
    p.add_argument("--trades-output", default="trades_robust_edge.csv")
    p.add_argument("--profiles-output", default="profile_train_results.csv")
    p.add_argument("--errors-output", default="errors_robust_edge.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.days != 90:
        print(f"WARNING: designed for 90 days, requested {args.days}")
    if not (0 < args.train_days < args.days):
        raise SystemExit("--train-days must be between 1 and days-1")
    if args.starting_capital <= 0:
        raise SystemExit("--starting-capital must be > 0")
    if args.leverage <= 0:
        raise SystemExit("--leverage must be > 0")
    if not (0 < args.risk_pct < 1):
        raise SystemExit("--risk-pct must be decimal, e.g. 0.04")
    if not (0 < args.max_margin_pct <= 1):
        raise SystemExit("--max-margin-pct must be decimal, e.g. 0.35")
    if args.fee < 0 or args.slippage < 0:
        raise SystemExit("fee/slippage cannot be negative")

    print("Loading MEXC USDT perpetual contracts...", flush=True)
    contracts = get_contracts(args.min_max_leverage)

    if args.symbols:
        symbols = [s for s in args.symbols if s in contracts]
    else:
        symbols = sorted(contracts)

    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]

    if not symbols:
        raise SystemExit("No symbols selected")

    print(
        f"Scanning {len(symbols)} tickers | "
        f"{args.days}d | train={args.train_days}d | OOS={args.days-args.train_days}d | "
        f"capital={args.starting_capital} | lev={args.leverage}x | "
        f"risk={args.risk_pct*100:.1f}% | max_margin={args.max_margin_pct*100:.1f}% | "
        f"fee={args.fee*100:.3f}%/side | slippage={args.slippage*100:.3f}%/side",
        flush=True,
    )

    ranking_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_symbol,
                symbol,
                contracts[symbol],
                args.days,
                args.starting_capital,
                args.leverage,
                args.risk_pct,
                args.max_margin_pct,
                args.fee,
                args.slippage,
                args.train_days,
                args.min_train_trades,
                args.min_oos_trades,
            ): symbol
            for symbol in symbols
        }

        done = 0
        for future in as_completed(future_map):
            symbol = future_map[future]
            done += 1
            try:
                row, trades, profiles = future.result()
                ranking_rows.append(row)
                trade_rows.extend(trades)
                profile_rows.extend(profiles)

                print(
                    f"[{done}/{len(symbols)}] {symbol} | "
                    f"profile={row['chosen_profile']} | "
                    f"OOS trades={row['oos_trades']} | "
                    f"OOS PF={row['oos_profit_factor']} | "
                    f"OOS ret={row['oos_return_pct']:.1f}% | "
                    f"OOS DD={row['oos_max_drawdown_pct']:.1f}% | "
                    f"candidate={row['candidate']}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"ticker": symbol, "error": repr(exc)})
                print(f"[{done}/{len(symbols)}] ERROR {symbol}: {exc}", flush=True)

    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            by=["candidate", "rank_score", "oos_return_pct", "oos_profit_factor", "oos_trades"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )
    ranking.to_csv(args.ranking_output, index=False)

    pd.DataFrame(trade_rows).to_csv(args.trades_output, index=False)

    profile_df = pd.DataFrame(profile_rows)
    if not profile_df.empty:
        profile_df = profile_df.sort_values(
            by=["ticker", "train_score"],
            ascending=[True, False],
            na_position="last",
        )
    profile_df.to_csv(args.profiles_output, index=False)

    pd.DataFrame(errors).to_csv(args.errors_output, index=False)

    print("\n================ TOP CANDIDATES ================", flush=True)
    if ranking.empty:
        print("No ranking rows.", flush=True)
    else:
        show_cols = [
            "ticker",
            "candidate",
            "rank_score",
            "chosen_profile",
            "actual_data_days",
            "train_trades",
            "train_profit_factor",
            "train_return_pct",
            "train_max_drawdown_pct",
            "oos_trades",
            "oos_profit_factor",
            "oos_return_pct",
            "oos_max_drawdown_pct",
            "oos_longest_loss_streak",
            "full90_trades",
            "full90_profit_factor",
            "full90_return_pct",
            "full90_max_drawdown_pct",
            "full90_end_equity_usdt",
        ]
        print(ranking[show_cols].head(30).to_string(index=False), flush=True)

    print(f"\nSaved: {args.ranking_output}")
    print(f"Saved: {args.trades_output}")
    print(f"Saved: {args.profiles_output}")
    print(f"Saved: {args.errors_output}")


if __name__ == "__main__":
    main()
