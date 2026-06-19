#!/usr/bin/env python3
"""
MEXC Futures scalping strategy scanner/backtester.

Port logiki PineScript "Scalping Strategy - TREND FOLLOWING BOT - 5 CANDLE ENTRY":
- SMA12/SMA50/SMA200
- Heikin Ashi
- trend following: MA12 najwyzej dla LONG, MA12 najnizej dla SHORT
- S1 pullback do MA12
- wejscie max 5 swiec po S1 na plaskiej HA
- SL = plaska krawedz swiecy wejscia
- symulacja TP/SL z zasada preferSLifBothHit

Wazne: aktualny Pine NIE wysyla TP w webhooku i NIE ma SL2.

Dodatkowo:
- skanuje tylko kontrakty USDT z maxLeverage >= min-symbol-leverage, domyslnie x100
- analizuje ostatnie N dni, domyslnie 7
- zapisuje tylko 2 pliki: latest_scalp_ranking.csv oraz scalp_trades_<interval>_<timestamp>.csv

To NIE sklada zlecen. To tylko analiza/backtest/sygnaly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://api.mexc.com"
MAX_KLINES_PER_REQUEST = 2000


@dataclasses.dataclass
class Params:
    interval: str = "Min1"
    days: int = 7
    tp_roi_percent: float = 20.0
    leverage: float = 75.0
    min_dist_12_50: float = 0.30
    min_dist_12_200: float = 0.60
    min_spike_pct: float = 0.50
    max_bars_after_spike: int = 5
    prefer_sl_if_both_hit: bool = True
    quote_coin: str = "USDT"
    max_workers: int = 6
    request_sleep: float = 0.05
    min_trades: int = 1
    min_symbol_leverage: int = 100


@dataclasses.dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: int
    entry_price: float
    sl_price: float
    tp_price: float
    exit_time: Optional[int] = None
    exit_type: Optional[str] = None
    bars_held: Optional[int] = None
    dist_12_50: Optional[float] = None
    dist_12_200: Optional[float] = None


def get_json(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Dict[str, Any]:
    url = BASE_URL + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", False):
        raise RuntimeError(f"MEXC API error for {url}: {data}")
    return data


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fetch_symbols(quote_coin: str = "USDT", min_symbol_leverage: int = 100) -> List[str]:
    data = get_json("/api/v1/contract/detail/country")
    rows = data.get("data", [])
    if isinstance(rows, dict):
        rows = [rows]

    symbols: List[str] = []
    skipped_low_leverage = 0

    for item in rows:
        max_leverage = safe_int(item.get("countryConfigContractMaxLeverage"), 0) or safe_int(item.get("maxLeverage"), 0)
        if (
            item.get("quoteCoin") == quote_coin
            and item.get("state") == 0
            and item.get("apiAllowed", True)
            and not item.get("isHidden", False)
            and item.get("symbol")
        ):
            if max_leverage >= min_symbol_leverage:
                symbols.append(item["symbol"])
            else:
                skipped_low_leverage += 1

    print(f"Symbols after leverage filter x{min_symbol_leverage}+: {len(symbols)} | skipped lower leverage: {skipped_low_leverage}")
    return sorted(set(symbols))


def interval_to_seconds(interval: str) -> int:
    mapping = {
        "Min1": 60,
        "Min5": 300,
        "Min15": 900,
        "Min30": 1800,
        "Min60": 3600,
        "Hour4": 14400,
        "Hour8": 28800,
        "Day1": 86400,
        "Week1": 604800,
        "Month1": 2592000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}. Use one of: {', '.join(mapping)}")
    return mapping[interval]



def normalize_symbol(symbol: str, quote_coin: str = "USDT") -> str:
    """Allow workflow input like BEAT, BEAT_USDT, beat/usdt."""
    s = symbol.strip().upper().replace("/", "_").replace("-", "_")
    if not s:
        return s
    if "_" not in s:
        return f"{s}_{quote_coin}"
    return s

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch last N days. MEXC returns max ~2000 candles per request, so we page by time."""
    seconds_per_bar = interval_to_seconds(interval)
    end_all = int(time.time())
    start_all = end_all - int(days * 86400)
    max_span = seconds_per_bar * MAX_KLINES_PER_REQUEST

    frames: List[pd.DataFrame] = []
    start = start_all

    while start < end_all:
        end = min(start + max_span, end_all)
        data = get_json(
            f"/api/v1/contract/kline/{symbol}",
            params={"interval": interval, "start": start, "end": end},
        ).get("data", {})

        if data and data.get("time"):
            frames.append(
                pd.DataFrame(
                    {
                        "time": data["time"],
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "vol": data.get("vol", [math.nan] * len(data["time"])),
                    }
                )
            )
        start = end + seconds_per_bar

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma12"] = out["close"].rolling(12).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["ma200"] = out["close"].rolling(200).mean()

    out["haClose"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    ha_open: List[float] = []
    for i, row in out.iterrows():
        if i == 0:
            ha_open.append((row["open"] + row["close"]) / 2.0)
        else:
            ha_open.append((ha_open[-1] + out.at[i - 1, "haClose"]) / 2.0)
    out["haOpen"] = ha_open
    out["haHigh"] = out[["high", "haOpen", "haClose"]].max(axis=1)
    out["haLow"] = out[["low", "haOpen", "haClose"]].min(axis=1)

    out["haGreen"] = out["haClose"] > out["haOpen"]
    out["haRed"] = out["haClose"] < out["haOpen"]
    eps = 1e-12
    out["haNoLowerWick"] = (out["haLow"] - out[["haOpen", "haClose"]].min(axis=1)).abs() <= eps
    out["haNoUpperWick"] = (out["haHigh"] - out[["haOpen", "haClose"]].max(axis=1)).abs() <= eps
    return out


def run_strategy(symbol: str, raw_df: pd.DataFrame, p: Params) -> Tuple[Dict[str, Any], List[Trade]]:
    df = add_indicators(raw_df)
    tp_move_pct = p.tp_roi_percent / p.leverage / 100.0

    short_spike_bar: Optional[int] = None
    long_spike_bar: Optional[int] = None

    in_trade = False
    direction = 0
    entry_price = math.nan
    sl_price = math.nan
    tp_price = math.nan
    entry_i = -1

    trades: List[Trade] = []
    tp_count = 0
    sl_count = 0
    long_count = 0
    short_count = 0
    spike_short_count = 0
    spike_long_count = 0
    latest_signal = ""

    for i, r in df.iterrows():
        if pd.isna(r["ma200"]):
            continue

        dist_12_50 = abs(r["ma12"] - r["ma50"]) / r["close"] * 100.0
        dist_12_200 = abs(r["ma12"] - r["ma200"]) / r["close"] * 100.0
        ma_distance_ok = dist_12_50 >= p.min_dist_12_50 and dist_12_200 >= p.min_dist_12_200

        # NEW PINE: TREND FOLLOWING LOGIC
        # SHORT: MA12 is lowest (downtrend), S1 is green pullback reaching MA12.
        # LONG:  MA12 is highest (uptrend), S1 is red pullback reaching MA12.
        # Pine currently keeps minSpikePct as input, but the active condition is >= 0 after reaching MA12.
        short_trend_stretch = r["ma12"] < r["ma50"] and r["ma50"] < r["ma200"] and ma_distance_ok
        long_trend_stretch = r["ma12"] > r["ma50"] and r["ma50"] > r["ma200"] and ma_distance_ok

        short_spike_candle = bool(short_trend_stretch and r["haGreen"] and r["haHigh"] >= r["ma12"] and ((r["haHigh"] - r["ma12"]) / r["ma12"] * 100.0 >= 0.0))
        long_spike_candle = bool(long_trend_stretch and r["haRed"] and r["haLow"] <= r["ma12"] and ((r["ma12"] - r["haLow"]) / r["ma12"] * 100.0 >= 0.0))

        if short_spike_candle:
            short_spike_bar = i
            spike_short_count += 1
            latest_signal = "SHORT_SPIKE"
        if long_spike_candle:
            long_spike_bar = i
            spike_long_count += 1
            latest_signal = "LONG_SPIKE"

        short_waiting = short_spike_bar is not None and i > short_spike_bar and i - short_spike_bar <= p.max_bars_after_spike
        long_waiting = long_spike_bar is not None and i > long_spike_bar and i - long_spike_bar <= p.max_bars_after_spike

        short_scalp = bool(short_waiting and r["haRed"] and r["haNoUpperWick"])
        long_scalp = bool(long_waiting and r["haGreen"] and r["haNoLowerWick"])

        new_long_entry = long_scalp and not in_trade
        new_short_entry = short_scalp and not in_trade

        if new_long_entry:
            in_trade = True
            direction = 1
            entry_i = i
            entry_price = float(r["close"])
            sl_price = float(r["haLow"])
            tp_price = entry_price * (1.0 + tp_move_pct)
            long_count += 1
            latest_signal = "SCALP_LONG_ENTRY"
            trades.append(Trade(symbol, "LONG", int(r["time"]), entry_price, sl_price, tp_price, dist_12_50=dist_12_50, dist_12_200=dist_12_200))


        if new_short_entry:
            in_trade = True
            direction = -1
            entry_i = i
            entry_price = float(r["close"])
            sl_price = float(r["haHigh"])
            tp_price = entry_price * (1.0 - tp_move_pct)
            short_count += 1
            latest_signal = "SCALP_SHORT_ENTRY"
            trades.append(Trade(symbol, "SHORT", int(r["time"]), entry_price, sl_price, tp_price, dist_12_50=dist_12_50, dist_12_200=dist_12_200))


        if short_scalp or (short_spike_bar is not None and i - short_spike_bar > p.max_bars_after_spike):
            short_spike_bar = None
        if long_scalp or (long_spike_bar is not None and i - long_spike_bar > p.max_bars_after_spike):
            long_spike_bar = None

        exit_tp = False
        exit_sl = False
        if in_trade and direction == 1:
            long_tp_hit = r["high"] >= tp_price
            long_sl_hit = r["low"] <= sl_price
            if long_tp_hit and long_sl_hit:
                exit_sl = p.prefer_sl_if_both_hit
                exit_tp = not p.prefer_sl_if_both_hit
            elif long_tp_hit:
                exit_tp = True
            elif long_sl_hit:
                exit_sl = True
        elif in_trade and direction == -1:
            short_tp_hit = r["low"] <= tp_price
            short_sl_hit = r["high"] >= sl_price
            if short_tp_hit and short_sl_hit:
                exit_sl = p.prefer_sl_if_both_hit
                exit_tp = not p.prefer_sl_if_both_hit
            elif short_tp_hit:
                exit_tp = True
            elif short_sl_hit:
                exit_sl = True

        if exit_tp or exit_sl:
            if trades:
                trades[-1].exit_time = int(r["time"])
                trades[-1].exit_type = "TP" if exit_tp else "SL"
                trades[-1].bars_held = int(i - entry_i)
            if exit_tp:
                tp_count += 1
                latest_signal = "SCALP_TP"
            if exit_sl:
                sl_count += 1
                latest_signal = "SCALP_SL"
            in_trade = False
            direction = 0
            entry_price = math.nan
            sl_price = math.nan
            tp_price = math.nan
            entry_i = -1

    total_trades = long_count + short_count
    closed_trades = tp_count + sl_count
    wr = (tp_count / closed_trades * 100.0) if closed_trades else 0.0
    pf_like = (tp_count * p.tp_roi_percent / max(sl_count, 1)) if sl_count else (tp_count * p.tp_roi_percent if tp_count else 0.0)

    last = df.iloc[-1]
    result = {
        "symbol": symbol,
        "interval": p.interval,
        "days": p.days,
        "candles": len(df),
        "min_symbol_leverage": p.min_symbol_leverage,
        "total_trades": total_trades,
        "closed_trades": closed_trades,
        "tp": tp_count,
        "sl": sl_count,
        "winrate_pct": round(wr, 2),
        "long_count": long_count,
        "short_count": short_count,
        "short_spikes": spike_short_count,
        "long_spikes": spike_long_count,
        "open_trade": "LONG" if in_trade and direction == 1 else "SHORT" if in_trade and direction == -1 else "",
        "latest_signal": latest_signal,
        "last_close": float(last["close"]),
        "last_dist_12_50_pct": round(float(abs(last["ma12"] - last["ma50"]) / last["close"] * 100.0), 4) if not pd.isna(last["ma200"]) else math.nan,
        "last_dist_12_200_pct": round(float(abs(last["ma12"] - last["ma200"]) / last["close"] * 100.0), 4) if not pd.isna(last["ma200"]) else math.nan,
        "score": round(wr * math.log1p(max(closed_trades, 0)), 4),
        "pf_like": round(pf_like, 4),
        "last_time_utc": datetime.fromtimestamp(int(last["time"]), tz=timezone.utc).isoformat(),
    }
    return result, trades


def scan_one(symbol: str, p: Params) -> Tuple[Optional[Dict[str, Any]], List[Trade], Optional[str]]:
    try:
        time.sleep(p.request_sleep)
        df = fetch_klines(symbol, p.interval, p.days)
        if df.empty or len(df) < 210:
            return None, [], f"{symbol}: not enough candles"
        result, trades = run_strategy(symbol, df, p)
        return result, trades, None
    except Exception as exc:  # noqa: BLE001
        return None, [], f"{symbol}: {exc}"


def trades_to_df(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [dataclasses.asdict(t) for t in trades]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col + "_utc"] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
    return df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan MEXC Futures tickers with Scalping Strategy")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols, e.g. BTC_USDT,ETH_USDT. Empty = all MEXC USDT futures.")
    parser.add_argument("--interval", default="Min1")
    parser.add_argument("--days", type=int, default=7, help="How many recent days to analyze. Default: 7.")
    parser.add_argument("--tp-roi-percent", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=75.0)
    parser.add_argument("--min-dist-12-50", type=float, default=0.30)
    parser.add_argument("--min-dist-12-200", type=float, default=0.60)
    parser.add_argument("--min-spike-pct", type=float, default=0.50, help="Kept for compatibility; current new Pine S1 uses pullback to MA12 with >=0 spike.")
    parser.add_argument("--max-bars-after-spike", type=int, default=5)
    parser.add_argument("--prefer-tp-if-both-hit", action="store_true", help="Opposite of Pine default. If same candle hits TP and SL, count TP.")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--min-symbol-leverage", type=int, default=100, help="Only scan contracts with maxLeverage >= this value. Default: 100.")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args(argv)

    p = Params(
        interval=args.interval,
        days=max(1, args.days),
        tp_roi_percent=args.tp_roi_percent,
        leverage=args.leverage,
        min_dist_12_50=args.min_dist_12_50,
        min_dist_12_200=args.min_dist_12_200,
        min_spike_pct=args.min_spike_pct,
        max_bars_after_spike=args.max_bars_after_spike,
        prefer_sl_if_both_hit=not args.prefer_tp_if_both_hit,
        max_workers=args.max_workers,
        min_trades=args.min_trades,
        min_symbol_leverage=args.min_symbol_leverage,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols.strip():
        symbols = [normalize_symbol(s, p.quote_coin) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = fetch_symbols(p.quote_coin, p.min_symbol_leverage)

    print(f"Scanning {len(symbols)} symbols | interval={p.interval} days={p.days} min_leverage=x{p.min_symbol_leverage}")

    results: List[Dict[str, Any]] = []
    all_trades: List[Trade] = []
    errors: List[str] = []

    with futures.ThreadPoolExecutor(max_workers=p.max_workers) as executor:
        fut_map = {executor.submit(scan_one, sym, p): sym for sym in symbols}
        for fut in futures.as_completed(fut_map):
            result, trades, err = fut.result()
            if err:
                errors.append(err)
            if result:
                results.append(result)
                all_trades.extend(trades)

    if not results:
        print("No results.")
        for e in errors[:20]:
            print("ERR", e)
        return 2

    ranking = pd.DataFrame(results)
    ranking = ranking[ranking["closed_trades"] >= p.min_trades].copy()
    ranking = ranking.sort_values(["score", "winrate_pct", "closed_trades"], ascending=[False, False, False])

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    latest_path = out_dir / "latest_scalp_ranking.csv"
    trades_path = out_dir / f"scalp_trades_{p.interval}_{ts}.csv"

    ranking.to_csv(latest_path, index=False)
    trades_df = trades_to_df(all_trades)
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)

    print("\nTOP 30:")
    cols = ["symbol", "closed_trades", "tp", "sl", "winrate_pct", "long_count", "short_count", "latest_signal", "open_trade", "score"]
    if ranking.empty:
        print("No symbols passed min_trades filter.")
    else:
        print(ranking[cols].head(30).to_string(index=False))

    print(f"\nSaved: {latest_path}")
    if not trades_df.empty:
        print(f"Saved: {trades_path}")
    if errors:
        # Nie wrzucamy trzeciego pliku raportowego. Pokazujemy tylko pierwsze bledy w logu Actions.
        print(f"\nErrors: {len(errors)}")
        for e in errors[:30]:
            print("ERR", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
