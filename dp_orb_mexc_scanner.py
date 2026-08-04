#!/usr/bin/env python3
"""
MEXC USDT perpetual scanner for the BOT6 DP_ORB Pine strategy.

Mirrors the Pine rules:
- ORB: 09:30-09:45 America/New_York
- Regular session: 09:30-16:00 America/New_York
- LONG: close > ORB High and high > high[1]
- SHORT: close < ORB Low and low < low[1]
- Entry = signal candle close
- LONG SL = signal candle low; SHORT SL = signal candle high
- Reject when SL ROI > configured limit
- TP = configured R multiple
- One position at a time
- TP/SL checked from the next candle; SL wins same-candle conflict
- Notional = current capital * margin multiplier
- Taker fee on entry and exit
- Capital compounds trade by trade

Outputs ranking, trade debug and errors CSV files.
No API key is required; only public MEXC futures endpoints are used.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE_URL = "https://contract.mexc.com"
NY_TZ = ZoneInfo("America/New_York")
DEFAULT_TIMEOUT = 30

INTERVAL_SECONDS: Dict[str, int] = {
    "Min1": 60,
    "Min5": 300,
    "Min15": 900,
    "Min30": 1800,
    "Min60": 3600,
    "Hour4": 14400,
}


@dataclass
class ScanConfig:
    interval: str = "Min5"
    days: int = 90
    tp_r: float = 5.0
    leverage: float = 20.0
    max_sl_roi_percent: float = 95.0
    starting_capital: float = 1.0
    margin_multiplier: float = 10.0
    taker_fee: float = 0.0008
    cooldown_bars: int = 0
    session_days: str = "23456"
    min_closed_trades: int = 20


@dataclass
class ContractMeta:
    ticker: str
    max_leverage: Optional[int]
    price_unit: Optional[float]
    quote_coin: Optional[str]
    state: Optional[int]
    api_allowed: bool


@dataclass
class ScanResult:
    ticker: str
    timeframe: str
    days: int
    max_leverage: Optional[int]
    eligible_min_trades: bool
    entries: int
    rejected_sl_roi: int
    rejected_cooldown: int
    tp: int
    sl: int
    win_rate_pct: Optional[float]
    net_r: float
    expectancy_r: Optional[float]
    expectancy_roi_pct_per_trade: Optional[float]
    avg_sl_price_pct: Optional[float]
    median_sl_price_pct: Optional[float]
    max_sl_price_pct: Optional[float]
    avg_sl_roi_pct: Optional[float]
    median_sl_roi_pct: Optional[float]
    max_sl_roi_pct_seen: Optional[float]
    avg_tp_roi_pct: Optional[float]
    median_tp_roi_pct: Optional[float]
    max_tp_roi_pct: Optional[float]
    gross_profit_usdt: float
    gross_loss_usdt: float
    profit_factor: Optional[float]
    starting_capital_usdt: float
    ending_capital_usdt: float
    highest_capital_usdt: float
    return_on_capital_pct: float
    max_drawdown_pct: float
    current_margin_per_trade_usdt: float
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: str
    cooldown_bars: int
    cooldown_remaining_bars: int
    status: str
    taker_fee_per_side_pct: float
    avg_fee_per_closed_trade_usdt: float
    total_fees_usdt: float
    net_pnl_usdt: float
    closed_trades: int
    open_trade_at_end: bool

    # Direction split from the SAME combined backtest.
    long_closed_trades: int
    long_tp: int
    long_sl: int
    long_win_rate_pct: Optional[float]
    long_net_r: float
    long_expectancy_r: Optional[float]
    long_gross_profit_usdt: float
    long_gross_loss_usdt: float
    long_profit_factor: Optional[float]
    long_fees_usdt: float
    long_net_pnl_usdt: float
    long_avg_sl_roi_pct: Optional[float]

    short_closed_trades: int
    short_tp: int
    short_sl: int
    short_win_rate_pct: Optional[float]
    short_net_r: float
    short_expectancy_r: Optional[float]
    short_gross_profit_usdt: float
    short_gross_loss_usdt: float
    short_profit_factor: Optional[float]
    short_fees_usdt: float
    short_net_pnl_usdt: float
    short_avg_sl_roi_pct: Optional[float]

    scanner_score: Optional[float]


def mexc_get(path: str, params: Optional[dict] = None, retries: int = 4) -> dict:
    url = BASE_URL + path
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                raise RuntimeError(f"MEXC API error: {payload}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.25 * attempt)
    raise last_error


def to_int(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def max_leverage(row: Dict[str, Any]) -> Optional[int]:
    vals = [
        to_int(row.get(k))
        for k in (
            "maxLeverage", "max_leverage", "maxLever", "leverageMax",
            "maxLongLeverage", "maxShortLeverage",
        )
    ]
    vals = [x for x in vals if x is not None]
    return max(vals) if vals else None


def get_contract_meta() -> Dict[str, ContractMeta]:
    rows = mexc_get("/api/v1/contract/detail").get("data", [])
    out: Dict[str, ContractMeta] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        out[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=max_leverage(row),
            price_unit=to_float(row.get("priceUnit")),
            quote_coin=row.get("quoteCoin"),
            state=to_int(row.get("state")),
            api_allowed=bool(row.get("apiAllowed", True)),
        )
    return out


def get_symbols(meta: Dict[str, ContractMeta], min_lev: Optional[int], limit: Optional[int]) -> List[str]:
    symbols = []
    for symbol, m in meta.items():
        if m.quote_coin != "USDT" or m.state != 0 or not m.api_allowed:
            continue
        if min_lev is not None and (m.max_leverage is None or m.max_leverage < min_lev):
            continue
        symbols.append(symbol)
    symbols = sorted(set(symbols))
    return symbols[:limit] if limit and limit > 0 else symbols


def fetch_klines(symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
    step = INTERVAL_SECONDS[interval]
    max_points = 1900
    chunk_span = step * max_points
    cursor = start
    frames = []

    while cursor < end:
        chunk_end = min(end, cursor + chunk_span - step)
        data = mexc_get(
            f"/api/v1/contract/kline/{symbol}",
            {"interval": interval, "start": cursor, "end": chunk_end},
        ).get("data", {})
        times = data.get("time") if isinstance(data, dict) else None
        if times:
            n = len(times)
            frames.append(pd.DataFrame({
                "time": times,
                "open": data.get("open", [None] * n),
                "high": data.get("high", [None] * n),
                "low": data.get("low", [None] * n),
                "close": data.get("close", [None] * n),
            }))
        cursor = chunk_end + step

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for c in ["time", "open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def add_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt_utc"] = pd.to_datetime(out["time"], unit="s", utc=True)
    out["dt_ny"] = out["dt_utc"].dt.tz_convert(NY_TZ)
    out["local_date"] = out["dt_ny"].dt.date
    out["minute"] = out["dt_ny"].dt.hour * 60 + out["dt_ny"].dt.minute
    out["pine_day"] = ((out["dt_ny"].dt.weekday + 1) % 7) + 1
    return out


def sm(values: List[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def s_med(values: List[float]) -> Optional[float]:
    return float(median(values)) if values else None


def s_max(values: List[float]) -> Optional[float]:
    return float(max(values)) if values else None


def streaks(seq: Iterable[str]) -> Tuple[int, int, str]:
    cw = cl = lw = ll = 0
    text = "-"
    for item in seq:
        if item == "TP":
            cw += 1
            cl = 0
            lw = max(lw, cw)
            text = f"W{cw}"
        else:
            cl += 1
            cw = 0
            ll = max(ll, cl)
            text = f"L{cl}"
    return lw, ll, text


def empty_result(symbol: str, cfg: ScanConfig, meta: Optional[ContractMeta]) -> ScanResult:
    return ScanResult(
        ticker=symbol,
        timeframe=cfg.interval,
        days=cfg.days,
        max_leverage=meta.max_leverage if meta else None,
        eligible_min_trades=False,
        entries=0,
        rejected_sl_roi=0,
        rejected_cooldown=0,
        tp=0,
        sl=0,
        win_rate_pct=None,
        net_r=0.0,
        expectancy_r=None,
        expectancy_roi_pct_per_trade=None,
        avg_sl_price_pct=None,
        median_sl_price_pct=None,
        max_sl_price_pct=None,
        avg_sl_roi_pct=None,
        median_sl_roi_pct=None,
        max_sl_roi_pct_seen=None,
        avg_tp_roi_pct=None,
        median_tp_roi_pct=None,
        max_tp_roi_pct=None,
        gross_profit_usdt=0.0,
        gross_loss_usdt=0.0,
        profit_factor=None,
        starting_capital_usdt=cfg.starting_capital,
        ending_capital_usdt=cfg.starting_capital,
        highest_capital_usdt=cfg.starting_capital,
        return_on_capital_pct=0.0,
        max_drawdown_pct=0.0,
        current_margin_per_trade_usdt=cfg.starting_capital * cfg.margin_multiplier / cfg.leverage,
        longest_win_streak=0,
        longest_loss_streak=0,
        current_streak="-",
        cooldown_bars=cfg.cooldown_bars,
        cooldown_remaining_bars=0,
        status="Brak",
        taker_fee_per_side_pct=cfg.taker_fee * 100.0,
        avg_fee_per_closed_trade_usdt=0.0,
        total_fees_usdt=0.0,
        net_pnl_usdt=0.0,
        closed_trades=0,
        open_trade_at_end=False,

        long_closed_trades=0,
        long_tp=0,
        long_sl=0,
        long_win_rate_pct=None,
        long_net_r=0.0,
        long_expectancy_r=None,
        long_gross_profit_usdt=0.0,
        long_gross_loss_usdt=0.0,
        long_profit_factor=None,
        long_fees_usdt=0.0,
        long_net_pnl_usdt=0.0,
        long_avg_sl_roi_pct=None,

        short_closed_trades=0,
        short_tp=0,
        short_sl=0,
        short_win_rate_pct=None,
        short_net_r=0.0,
        short_expectancy_r=None,
        short_gross_profit_usdt=0.0,
        short_gross_loss_usdt=0.0,
        short_profit_factor=None,
        short_fees_usdt=0.0,
        short_net_pnl_usdt=0.0,
        short_avg_sl_roi_pct=None,

        scanner_score=None,
    )


def scan_symbol(symbol: str, cfg: ScanConfig, meta: Optional[ContractMeta]) -> Tuple[ScanResult, List[Dict[str, Any]]]:
    now = int(time.time())
    step = INTERVAL_SECONDS[cfg.interval]
    start = now - cfg.days * 86400
    raw = fetch_klines(symbol, cfg.interval, start - 86400, now)
    if len(raw) < 10:
        return empty_result(symbol, cfg, meta), []

    df = add_sessions(raw)
    df = df[(df["time"] + step) <= now].reset_index(drop=True)
    scan_idx = df.index[df["time"] >= start].tolist()
    if not scan_idx:
        return empty_result(symbol, cfg, meta), []
    first_i = max(1, scan_idx[0])

    allowed = {int(x) for x in cfg.session_days if x.isdigit() and 1 <= int(x) <= 7}
    if not allowed:
        raise ValueError("Invalid session_days; use e.g. 23456 or 1234567")

    orb_start, orb_end = 570, 585
    reg_start, reg_end = 570, 960
    orb_mask = df["pine_day"].isin(allowed) & (df["minute"] >= orb_start) & (df["minute"] < orb_end)
    orb = df.loc[orb_mask].groupby("local_date").agg(
        orb_high=("high", "max"), orb_low=("low", "min")
    ).to_dict("index")

    position = None
    last_entry_i = None
    capital = cfg.starting_capital
    highest = cfg.starting_capital
    max_dd = 0.0
    entries = rejected_sl = rejected_cd = wins = losses = closed = 0
    gp = gl = fees = sum_roi = 0.0

    # Directional attribution of the exact same combined backtest.
    dir_stats = {
        "LONG": {
            "closed": 0, "tp": 0, "sl": 0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "fees": 0.0, "net_pnl": 0.0,
            "sl_roi_vals": [],
        },
        "SHORT": {
            "closed": 0, "tp": 0, "sl": 0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "fees": 0.0, "net_pnl": 0.0,
            "sl_roi_vals": [],
        },
    }

    sl_price_vals: List[float] = []
    sl_roi_vals: List[float] = []
    tp_roi_vals: List[float] = []
    seq: List[str] = []
    debug: List[Dict[str, Any]] = []
    min_tick = meta.price_unit if meta and meta.price_unit and meta.price_unit > 0 else 0.0

    for i in range(first_i, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        row_time = int(row["time"])
        had_position_at_bar_open = position is not None

        # Exit block. Pine checks only from the candle AFTER entry.
        if position is not None and i > position["entry_i"]:
            d = position["direction"]
            hit_sl = (d == 1 and row["low"] <= position["sl"]) or (d == -1 and row["high"] >= position["sl"])
            hit_tp = (d == 1 and row["high"] >= position["tp"]) or (d == -1 and row["low"] <= position["tp"])
            if hit_sl or hit_tp:
                won = False if hit_sl else True
                exit_price = position["sl"] if hit_sl else position["tp"]
                qty = position["qty"]
                entry = position["entry"]
                cap_before = position["capital_before"]
                exit_notional = qty * exit_price
                exit_fee = exit_notional * cfg.taker_fee
                gross = qty * (exit_price - entry) if d == 1 else qty * (entry - exit_price)
                trade_fee = position["entry_fee"] + exit_fee
                net = gross - trade_fee
                trade_roi = net / cap_before * 100.0 if cap_before else 0.0

                capital = cap_before + net
                fees += trade_fee
                sum_roi += trade_roi
                closed += 1

                direction_name = "LONG" if d == 1 else "SHORT"
                ds = dir_stats[direction_name]
                ds["closed"] += 1
                ds["fees"] += trade_fee
                ds["net_pnl"] += net
                ds["sl_roi_vals"].append(float(position["sl_roi_pct"]))

                if won:
                    wins += 1
                    gp += gross
                    ds["tp"] += 1
                    ds["gross_profit"] += gross
                    seq.append("TP")
                    result = "TP"
                else:
                    losses += 1
                    gl += gross
                    ds["sl"] += 1
                    ds["gross_loss"] += gross
                    seq.append("SL")
                    result = "SL"

                highest = max(highest, capital)
                dd = (highest - capital) / highest * 100.0 if highest > 0 else 0.0
                max_dd = max(max_dd, dd)

                debug.append({
                    "ticker": symbol,
                    "timeframe": cfg.interval,
                    "entry_time_utc": pd.Timestamp(position["entry_time"], unit="s", tz="UTC"),
                    "entry_time_ny": pd.Timestamp(position["entry_time"], unit="s", tz="UTC").tz_convert(NY_TZ),
                    "exit_time_utc": pd.Timestamp(row_time, unit="s", tz="UTC"),
                    "exit_time_ny": pd.Timestamp(row_time, unit="s", tz="UTC").tz_convert(NY_TZ),
                    "direction": "LONG" if d == 1 else "SHORT",
                    "result": result,
                    "entry_price": entry,
                    "sl_price": position["sl"],
                    "tp_price": position["tp"],
                    "exit_price": exit_price,
                    "sl_price_pct": position["sl_price_pct"],
                    "sl_roi_pct": position["sl_roi_pct"],
                    "tp_roi_pct": position["tp_roi_pct"],
                    "capital_before_usdt": cap_before,
                    "position_notional_usdt": position["notional"],
                    "margin_used_usdt": position["margin"],
                    "gross_pnl_usdt": gross,
                    "fee_usdt": trade_fee,
                    "net_pnl_usdt": net,
                    "trade_roi_pct_on_capital": trade_roi,
                    "capital_after_usdt": capital,
                })
                position = None

        # Pine entry section ran BEFORE its exit section, so a bar that started
        # with an open position cannot open a new trade after closing it.
        if had_position_at_bar_open:
            continue
        if position is not None or capital <= 0:
            continue

        pine_day, minute = int(row["pine_day"]), int(row["minute"])
        if pine_day not in allowed or not (reg_start <= minute < reg_end) or minute < orb_end:
            continue
        levels = orb.get(row["local_date"])
        if not levels:
            continue

        long_sig = row["close"] > levels["orb_high"] and row["high"] > prev["high"]
        short_sig = row["close"] < levels["orb_low"] and row["low"] < prev["low"]
        if not (long_sig or short_sig):
            continue

        cd_ready = last_entry_i is None or cfg.cooldown_bars <= 0 or (i - last_entry_i) >= cfg.cooldown_bars
        if not cd_ready:
            rejected_cd += 1
            continue

        if long_sig:
            d = 1
            entry = float(row["close"])
            stop = float(row["low"])
            risk = entry - stop
        else:
            d = -1
            entry = float(row["close"])
            stop = float(row["high"])
            risk = stop - entry

        sl_price_pct = risk / entry * 100.0 if entry > 0 and risk > 0 else math.nan
        sl_roi = sl_price_pct * cfg.leverage if not math.isnan(sl_price_pct) else math.nan
        valid = risk > max(min_tick * 0.1, 0.0) and risk > 0 and not math.isnan(sl_roi) and sl_roi <= cfg.max_sl_roi_percent
        if not valid:
            rejected_sl += 1
            continue

        notional = capital * cfg.margin_multiplier
        margin = notional / cfg.leverage
        qty = notional / entry
        entry_fee = notional * cfg.taker_fee
        target = entry + risk * cfg.tp_r if d == 1 else entry - risk * cfg.tp_r
        tp_price_pct = abs(target - entry) / entry * 100.0
        tp_roi = tp_price_pct * cfg.leverage

        position = {
            "direction": d, "entry_i": i, "entry_time": row_time,
            "entry": entry, "sl": stop, "tp": target, "qty": qty,
            "capital_before": capital, "notional": notional, "margin": margin,
            "entry_fee": entry_fee, "sl_price_pct": sl_price_pct,
            "sl_roi_pct": sl_roi, "tp_roi_pct": tp_roi,
        }
        last_entry_i = i
        entries += 1
        sl_price_vals.append(sl_price_pct)
        sl_roi_vals.append(sl_roi)
        tp_roi_vals.append(tp_roi)

    wr = wins / closed * 100.0 if closed else None
    net_r = wins * cfg.tp_r - losses
    exp_r = net_r / closed if closed else None
    exp_roi = sum_roi / closed if closed else None
    pf = gp / abs(gl) if gl < 0 else (math.inf if gp > 0 else None)
    return_pct = (capital / cfg.starting_capital - 1.0) * 100.0
    net_pnl = capital - cfg.starting_capital
    avg_fee = fees / closed if closed else 0.0
    next_margin = max(capital, 0.0) * cfg.margin_multiplier / cfg.leverage
    lw, ll, current = streaks(seq)
    cooldown_remaining = 0 if last_entry_i is None or cfg.cooldown_bars <= 0 else max(cfg.cooldown_bars - ((len(df) - 1) - last_entry_i), 0)
    status = "Brak" if position is None else ("LONG" if position["direction"] == 1 else "SHORT")
    eligible = closed >= cfg.min_closed_trades

    def direction_metrics(name: str) -> Dict[str, Any]:
        ds = dir_stats[name]
        d_closed = int(ds["closed"])
        d_tp = int(ds["tp"])
        d_sl = int(ds["sl"])
        d_wr = d_tp / d_closed * 100.0 if d_closed else None
        d_net_r = d_tp * cfg.tp_r - d_sl
        d_exp_r = d_net_r / d_closed if d_closed else None
        d_gp = float(ds["gross_profit"])
        d_gl = float(ds["gross_loss"])
        d_pf = d_gp / abs(d_gl) if d_gl < 0 else (math.inf if d_gp > 0 else None)

        return {
            "closed": d_closed,
            "tp": d_tp,
            "sl": d_sl,
            "win_rate": d_wr,
            "net_r": d_net_r,
            "expectancy_r": d_exp_r,
            "gross_profit": d_gp,
            "gross_loss": d_gl,
            "profit_factor": d_pf,
            "fees": float(ds["fees"]),
            "net_pnl": float(ds["net_pnl"]),
            "avg_sl_roi": sm(ds["sl_roi_vals"]),
        }

    long_m = direction_metrics("LONG")
    short_m = direction_metrics("SHORT")

    score = None
    if closed:
        pf_score = min(pf if pf is not None and math.isfinite(pf) else 5.0, 5.0)
        score = (exp_r or 0.0) * 20 + pf_score * 5 + return_pct * 0.10 - max_dd * 0.10 + min(closed, 100) * 0.05 - ll * 0.25

    result = ScanResult(
        ticker=symbol, timeframe=cfg.interval, days=cfg.days,
        max_leverage=meta.max_leverage if meta else None,
        eligible_min_trades=eligible,
        entries=entries, rejected_sl_roi=rejected_sl, rejected_cooldown=rejected_cd,
        tp=wins, sl=losses, win_rate_pct=wr, net_r=net_r, expectancy_r=exp_r,
        expectancy_roi_pct_per_trade=exp_roi,
        avg_sl_price_pct=sm(sl_price_vals), median_sl_price_pct=s_med(sl_price_vals), max_sl_price_pct=s_max(sl_price_vals),
        avg_sl_roi_pct=sm(sl_roi_vals), median_sl_roi_pct=s_med(sl_roi_vals), max_sl_roi_pct_seen=s_max(sl_roi_vals),
        avg_tp_roi_pct=sm(tp_roi_vals), median_tp_roi_pct=s_med(tp_roi_vals), max_tp_roi_pct=s_max(tp_roi_vals),
        gross_profit_usdt=gp, gross_loss_usdt=gl, profit_factor=pf,
        starting_capital_usdt=cfg.starting_capital, ending_capital_usdt=capital,
        highest_capital_usdt=highest, return_on_capital_pct=return_pct, max_drawdown_pct=max_dd,
        current_margin_per_trade_usdt=next_margin, longest_win_streak=lw, longest_loss_streak=ll,
        current_streak=current, cooldown_bars=cfg.cooldown_bars,
        cooldown_remaining_bars=cooldown_remaining, status=status,
        taker_fee_per_side_pct=cfg.taker_fee * 100.0,
        avg_fee_per_closed_trade_usdt=avg_fee, total_fees_usdt=fees, net_pnl_usdt=net_pnl,
        closed_trades=closed, open_trade_at_end=position is not None,

        long_closed_trades=long_m["closed"],
        long_tp=long_m["tp"],
        long_sl=long_m["sl"],
        long_win_rate_pct=long_m["win_rate"],
        long_net_r=long_m["net_r"],
        long_expectancy_r=long_m["expectancy_r"],
        long_gross_profit_usdt=long_m["gross_profit"],
        long_gross_loss_usdt=long_m["gross_loss"],
        long_profit_factor=long_m["profit_factor"],
        long_fees_usdt=long_m["fees"],
        long_net_pnl_usdt=long_m["net_pnl"],
        long_avg_sl_roi_pct=long_m["avg_sl_roi"],

        short_closed_trades=short_m["closed"],
        short_tp=short_m["tp"],
        short_sl=short_m["sl"],
        short_win_rate_pct=short_m["win_rate"],
        short_net_r=short_m["net_r"],
        short_expectancy_r=short_m["expectancy_r"],
        short_gross_profit_usdt=short_m["gross_profit"],
        short_gross_loss_usdt=short_m["gross_loss"],
        short_profit_factor=short_m["profit_factor"],
        short_fees_usdt=short_m["fees"],
        short_net_pnl_usdt=short_m["net_pnl"],
        short_avg_sl_roi_pct=short_m["avg_sl_roi"],

        scanner_score=score,
    )
    return result, debug


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEXC BOT6 DP_ORB ticker scanner")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-max-leverage", type=int, default=20)
    p.add_argument("--interval", choices=sorted(INTERVAL_SECONDS), default="Min5")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--tp-r", type=float, default=5.0)
    p.add_argument("--leverage", type=float, default=20.0)
    p.add_argument("--max-sl-roi", type=float, default=95.0)
    p.add_argument("--starting-capital", type=float, default=1.0)
    p.add_argument("--margin-multiplier", type=float, default=10.0)
    p.add_argument("--taker-fee", type=float, default=0.0008)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--session-days", default="23456")
    p.add_argument("--min-closed-trades", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default="ranking_dp_orb.csv")
    p.add_argument("--debug-output", default="trades_dp_orb_debug.csv")
    p.add_argument("--errors-output", default="errors_dp_orb.csv")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.days <= 0 or a.tp_r <= 0 or a.leverage <= 0 or a.max_sl_roi <= 0:
        raise SystemExit("days/tp-r/leverage/max-sl-roi must be > 0")
    if a.starting_capital <= 0 or a.margin_multiplier <= 0 or a.taker_fee < 0:
        raise SystemExit("invalid capital/margin/fee")
    if a.cooldown_bars < 0 or a.workers <= 0:
        raise SystemExit("cooldown/workers invalid")

    cfg = ScanConfig(
        interval=a.interval, days=a.days, tp_r=a.tp_r, leverage=a.leverage,
        max_sl_roi_percent=a.max_sl_roi, starting_capital=a.starting_capital,
        margin_multiplier=a.margin_multiplier, taker_fee=a.taker_fee,
        cooldown_bars=a.cooldown_bars, session_days=a.session_days,
        min_closed_trades=a.min_closed_trades,
    )

    print("Loading MEXC contract metadata...", flush=True)
    meta = get_contract_meta()
    symbols = get_symbols(meta, a.min_max_leverage, a.limit) if a.all else sorted(set(a.symbols or []))
    if not symbols:
        raise SystemExit("Use --all or --symbols RIF_USDT BTC_USDT")

    print(f"Tickers: {len(symbols)} | Config: {cfg}", flush=True)
    results, debug, errors = [], [], []

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(scan_symbol, s, cfg, meta.get(s)): s for s in symbols}
        for n, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result, trades = future.result()
                results.append(asdict(result))
                debug.extend(trades)
                print(f"[{n}/{len(symbols)}] {symbol} trades={result.closed_trades} PF={result.profit_factor} return={result.return_on_capital_pct:.2f}% DD={result.max_drawdown_pct:.2f}%", flush=True)
            except Exception as exc:
                errors.append({"ticker": symbol, "error": repr(exc)})
                print(f"[{n}/{len(symbols)}] ERROR {symbol}: {exc}", flush=True)

    ranking = pd.DataFrame(results)
    if not ranking.empty:
        ranking["pf_sort"] = pd.to_numeric(ranking["profit_factor"], errors="coerce").replace([math.inf], 999.0)
        ranking = ranking.sort_values(
            ["eligible_min_trades", "scanner_score", "return_on_capital_pct", "pf_sort", "closed_trades"],
            ascending=[False, False, False, False, False], na_position="last"
        ).drop(columns=["pf_sort"])
    ranking.to_csv(a.output, index=False)
    pd.DataFrame(debug).to_csv(a.debug_output, index=False)
    pd.DataFrame(errors).to_csv(a.errors_output, index=False)

    print("\nTOP 30:", flush=True)
    if not ranking.empty:
        cols = [
            "ticker", "timeframe", "entries", "tp", "sl", "win_rate_pct", "net_r",
            "expectancy_r", "expectancy_roi_pct_per_trade", "profit_factor",
            "ending_capital_usdt", "return_on_capital_pct", "max_drawdown_pct",

            "long_closed_trades", "long_tp", "long_sl", "long_win_rate_pct",
            "long_net_r", "long_expectancy_r", "long_profit_factor", "long_net_pnl_usdt",

            "short_closed_trades", "short_tp", "short_sl", "short_win_rate_pct",
            "short_net_r", "short_expectancy_r", "short_profit_factor", "short_net_pnl_usdt",

            "longest_loss_streak", "total_fees_usdt", "net_pnl_usdt",
            "closed_trades", "scanner_score",
        ]
        print(ranking[cols].head(30).to_string(index=False), flush=True)
    print(f"Saved: {a.output}, {a.debug_output}, {a.errors_output}", flush=True)


if __name__ == "__main__":
    main()
