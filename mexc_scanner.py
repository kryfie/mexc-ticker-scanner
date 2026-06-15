import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import pandas as pd
import requests

BASE_URL = "https://api.mexc.com"


@dataclass
class ScanConfig:
    tp_percent: float = 1.2
    cooldown_minutes: int = 180
    conservative_same_candle: bool = True
    check_result_on_entry_bar: bool = False
    days: int = 14


@dataclass
class ScanResult:
    ticker: str
    entries: int
    tp: int
    sl: int
    wr: Optional[float]
    sl_rate: Optional[float]
    sl1: int
    sl2: int
    sl3: int
    ma200_raw: int
    skipped_cooldown: int
    skipped_in_trade: int
    sequence_last_30: str
    score: Optional[float]


def mexc_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("success") is False:
        raise RuntimeError(f"MEXC error: {data}")
    return data


def get_contract_symbols(limit: Optional[int] = None) -> List[str]:
    data = mexc_get("/api/v1/contract/detail")
    rows = data.get("data", [])
    symbols = []
    for x in rows:
        if x.get("state") == 0 and x.get("quoteCoin") == "USDT" and x.get("apiAllowed", True):
            symbols.append(x["symbol"])
    symbols = sorted(set(symbols))
    return symbols[:limit] if limit else symbols


def fetch_klines(symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
    step_seconds = 60 if interval == "Min1" else 15 * 60
    max_points = 2000
    chunk = step_seconds * max_points
    frames = []
    cursor = start

    while cursor < end:
        chunk_end = min(end, cursor + chunk - step_seconds)
        data = mexc_get(
            f"/api/v1/contract/kline/{symbol}",
            {
                "interval": interval,
                "start": cursor,
                "end": chunk_end,
            },
        ).get("data", {})

        if not data or not data.get("time"):
            cursor = chunk_end + step_seconds
            continue

        df = pd.DataFrame(
            {
                "time": data["time"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "vol": data.get("vol", [None] * len(data["time"])),
            }
        )
        frames.append(df)
        cursor = chunk_end + step_seconds
        time.sleep(0.06)

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "vol"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("time").sort_values("time")

    for col in ["open", "high", "low", "close", "vol"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.reset_index(drop=True)


def crossover(a_prev, a, b_prev, b) -> bool:
    return a_prev <= b_prev and a > b


def crossunder(a_prev, a, b_prev, b) -> bool:
    return a_prev >= b_prev and a < b


def prepare_data(df1: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    df = df1.copy()
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df["ma12"] = df["close"].rolling(12).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = []

    for i, row in df.iterrows():
        if i == 0:
            ha_open.append((row["open"] + row["close"]) / 2)
        else:
            ha_open.append((ha_open[i - 1] + ha_close.iloc[i - 1]) / 2)

    df["ha_close"] = ha_close
    df["ha_open"] = ha_open
    df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"] = df[["low", "ha_open", "ha_close"]].min(axis=1)

    m15 = df15.copy()
    m15["dt15"] = pd.to_datetime(m15["time"], unit="s", utc=True)
    m15["m15_close"] = m15["close"]
    m15["m15_ma12"] = m15["close"].rolling(12).mean()
    m15["m15_ma200"] = m15["close"].rolling(200).mean()
    m15 = m15[["dt15", "m15_close", "m15_ma12", "m15_ma200"]].dropna()

    df = pd.merge_asof(
        df.sort_values("dt"),
        m15.sort_values("dt15"),
        left_on="dt",
        right_on="dt15",
        direction="backward",
    )

    return df.reset_index(drop=True)


def reset_trade_state() -> Dict[str, Any]:
    return {
        "in_trade": False,
        "direction": 0,
        "entry_i": None,
        "entry_time": None,
        "entry_price": None,
        "tp_price": None,
        "sl1_price": None,
        "sl2_price": None,
        "sl2_active": False,
        "sl3_price": None,
        "sl3_active": False,
    }


def scan_symbol(symbol: str, cfg: ScanConfig) -> tuple[ScanResult, List[Dict[str, Any]]]:
    end = int(time.time())
    start = end - cfg.days * 24 * 60 * 60

    # Long warmup for MA and HA stability.
    warmup_start = start - 30 * 24 * 60 * 60

    df1 = fetch_klines(symbol, "Min1", warmup_start, end)
    df15 = fetch_klines(symbol, "Min15", warmup_start, end)

    if len(df1) < 250 or len(df15) < 210:
        empty_result = ScanResult(symbol, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, "", None)
        return empty_result, []

    df = prepare_data(df1, df15)
    df = df[df["time"] >= start].reset_index(drop=True)

    state = reset_trade_state()

    last_entry_time = None
    total = 0
    tp = 0
    sl = 0
    sl1 = 0
    sl2 = 0
    sl3 = 0

    ma200_raw = 0
    skipped_cooldown = 0
    skipped_in_trade = 0

    seq = []
    debug_trades: List[Dict[str, Any]] = []

    cooldown_seconds = cfg.cooldown_minutes * 60

    for i in range(1, len(df)):
        r = df.iloc[i]
        p = df.iloc[i - 1]

        needed = [
            r.ma12,
            r.ma50,
            r.ma200,
            p.ma12,
            p.ma50,
            p.ma200,
            r.m15_close,
            r.m15_ma12,
            r.m15_ma200,
        ]

        if any(pd.isna(x) for x in needed):
            continue

        ha_green = r.ha_close > r.ha_open
        ha_red = r.ha_close < r.ha_open
        ha_no_lower = abs(r.ha_low - min(r.ha_open, r.ha_close)) < 1e-12
        ha_no_upper = abs(r.ha_high - max(r.ha_open, r.ha_close)) < 1e-12

        m15_long_trend = r.m15_close > r.m15_ma200
        m15_short_trend = r.m15_close < r.m15_ma200
        m15_dist = abs(r.m15_ma12 - r.m15_ma200) / r.m15_close * 100
        m15_strong = m15_dist > 7.5

        allow_long = m15_long_trend or (m15_short_trend and m15_strong)
        allow_short = m15_short_trend or (m15_long_trend and m15_strong)

        cross_ma12_up = crossover(p.ma12, r.ma12, p.ma200, r.ma200)
        cross_ma12_down = crossunder(p.ma12, r.ma12, p.ma200, r.ma200)
        cross_ma50_up = crossover(p.ma50, r.ma50, p.ma200, r.ma200)
        cross_ma50_down = crossunder(p.ma50, r.ma50, p.ma200, r.ma200)
        cross_ha_up = crossover(p.ha_close, r.ha_close, p.ma200, r.ma200)
        cross_ha_down = crossunder(p.ha_close, r.ha_close, p.ma200, r.ma200)

        long_signal = allow_long and cross_ma12_up and r.ma50 < r.ma12 and r.ma50 < r.ma200
        short_signal = allow_short and cross_ma12_down and r.ma50 > r.ma12 and r.ma50 > r.ma200

        ma50_up = allow_long and cross_ma50_up and r.ma12 > r.ma50 and r.ma12 > r.ma200
        ma50_down = allow_short and cross_ma50_down and r.ma12 < r.ma50 and r.ma12 < r.ma200

        ma200_up_raw = (
            allow_long
            and ha_green
            and ha_no_lower
            and cross_ha_up
            and r.ma50 < r.ma12
            and r.ma12 < r.ma200
        )

        ma200_down_raw = (
            allow_short
            and ha_red
            and ha_no_upper
            and cross_ha_down
            and r.ma50 > r.ma12
            and r.ma12 > r.ma200
        )

        ma200_raw_signal = ma200_up_raw or ma200_down_raw

        cooldown_ok = last_entry_time is None or (int(r.time) - last_entry_time) >= cooldown_seconds

        if ma200_raw_signal:
            ma200_raw += 1

            if not cooldown_ok:
                skipped_cooldown += 1

            if cooldown_ok and state["in_trade"]:
                skipped_in_trade += 1

        ma200_up = ma200_up_raw and cooldown_ok and not state["in_trade"]
        ma200_down = ma200_down_raw and cooldown_ok and not state["in_trade"]

        if ma200_up:
            state["in_trade"] = True
            state["direction"] = 1
            state["entry_i"] = i
            state["entry_time"] = int(r.time)
            state["entry_price"] = r.high
            state["tp_price"] = state["entry_price"] * (1 + cfg.tp_percent / 100)
            state["sl1_price"] = r.ma50
            state["sl2_price"] = r.ha_low
            state["sl2_active"] = False
            state["sl3_price"] = None
            state["sl3_active"] = False

            total += 1
            last_entry_time = int(r.time)

        elif ma200_down:
            state["in_trade"] = True
            state["direction"] = -1
            state["entry_i"] = i
            state["entry_time"] = int(r.time)
            state["entry_price"] = r.low
            state["tp_price"] = state["entry_price"] * (1 - cfg.tp_percent / 100)
            state["sl1_price"] = r.ma50
            state["sl2_price"] = r.ha_high
            state["sl2_active"] = False
            state["sl3_price"] = None
            state["sl3_active"] = False

            total += 1
            last_entry_time = int(r.time)

        if state["in_trade"] and state["direction"] == 1 and long_signal and not state["sl2_active"]:
            state["sl2_active"] = True

        if state["in_trade"] and state["direction"] == -1 and short_signal and not state["sl2_active"]:
            state["sl2_active"] = True

        if state["in_trade"] and state["direction"] == 1 and ma50_up and not state["sl3_active"]:
            state["sl3_price"] = r.ma50
            state["sl3_active"] = True

        if state["in_trade"] and state["direction"] == -1 and ma50_down and not state["sl3_active"]:
            state["sl3_price"] = r.ma50
            state["sl3_active"] = True

        can_check = state["in_trade"] and (
            cfg.check_result_on_entry_bar or i > state["entry_i"]
        )

        if not can_check:
            continue

        direction = state["direction"]

        tp_hit = (
            direction == 1 and r.high >= state["tp_price"]
        ) or (
            direction == -1 and r.low <= state["tp_price"]
        )

        sl1_hit = (
            direction == 1 and r.low <= state["sl1_price"]
        ) or (
            direction == -1 and r.high >= state["sl1_price"]
        )

        sl2_hit = state["sl2_active"] and (
            (direction == 1 and r.low <= state["sl2_price"])
            or (direction == -1 and r.high >= state["sl2_price"])
        )

        sl3_hit = state["sl3_active"] and (
            (direction == 1 and r.low <= state["sl3_price"])
            or (direction == -1 and r.high >= state["sl3_price"])
        )

        sl_hit = sl1_hit or sl2_hit or sl3_hit

        result_sl = sl_hit and (cfg.conservative_same_candle or not tp_hit)
        result_tp = tp_hit and not result_sl

        if result_tp or result_sl:
            result = "TP" if result_tp else "SL"
            sl_type = ""

            if result_sl:
                if sl1_hit:
                    sl_type = "SL1 MA50 entry"
                elif sl2_hit:
                    sl_type = "SL2 HA flat"
                elif sl3_hit:
                    sl_type = "SL3 MA50 cross"

            debug_trades.append(
                {
                    "ticker": symbol,
                    "entry_time_utc": pd.to_datetime(state["entry_time"], unit="s", utc=True),
                    "exit_time_utc": pd.to_datetime(int(r.time), unit="s", utc=True),
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": state["entry_price"],
                    "exit_candle_open": r.open,
                    "exit_candle_high": r.high,
                    "exit_candle_low": r.low,
                    "exit_candle_close": r.close,
                    "result": result,
                    "sl_type": sl_type,
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
                }
            )

            if result_tp:
                tp += 1
                seq.append("TP")
            else:
                sl += 1
                seq.append("SL")

                if sl1_hit:
                    sl1 += 1
                elif sl2_hit:
                    sl2 += 1
                elif sl3_hit:
                    sl3 += 1

            state = reset_trade_state()

    closed = tp + sl
    wr = tp / closed * 100 if closed else None
    sl_rate = sl / closed * 100 if closed else None

    score = None
    if closed:
        score = wr + min(total, 40) * 0.5 - sl1 * 2.0

    result = ScanResult(
        ticker=symbol,
        entries=total,
        tp=tp,
        sl=sl,
        wr=wr,
        sl_rate=sl_rate,
        sl1=sl1,
        sl2=sl2,
        sl3=sl3,
        ma200_raw=ma200_raw,
        skipped_cooldown=skipped_cooldown,
        skipped_in_trade=skipped_in_trade,
        sequence_last_30=" -> ".join(seq[-30:]),
        score=score,
    )

    return result, debug_trades


def main():
    parser = argparse.ArgumentParser(description="MEXC futures scanner based on TradingView Pine logic")

    parser.add_argument("--symbols", nargs="*", help="Examples: SUI_USDT ONDO_USDT VELVET_USDT")
    parser.add_argument("--all", action="store_true", help="Scan all enabled USDT futures")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols when using --all")

    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--tp", type=float, default=1.2)
    parser.add_argument("--cooldown", type=int, default=180)

    parser.add_argument("--output", default="ranking.csv")
    parser.add_argument("--debug-output", default="trades_debug.csv")

    args = parser.parse_args()

    cfg = ScanConfig(
        tp_percent=args.tp,
        cooldown_minutes=args.cooldown,
        days=args.days,
    )

    symbols = args.symbols or []

    if args.all:
        symbols = get_contract_symbols(args.limit)

    if not symbols:
        raise SystemExit("Podaj --symbols SUI_USDT ONDO_USDT albo użyj --all")

    results = []
    all_debug_trades = []

    for n, symbol in enumerate(symbols, start=1):
        print(f"[{n}/{len(symbols)}] scanning {symbol}...")

        try:
            result, debug_trades = scan_symbol(symbol, cfg)
            results.append(asdict(result))
            all_debug_trades.extend(debug_trades)

        except Exception as e:
            print(f"ERROR {symbol}: {e}")

        time.sleep(0.12)

    ranking_df = pd.DataFrame(results)

    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            ["score", "entries"],
            ascending=[False, False],
            na_position="last",
        )

    ranking_df.to_csv(args.output, index=False)

    debug_df = pd.DataFrame(all_debug_trades)

    if not debug_df.empty:
        debug_df.to_csv(args.debug_output, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ticker",
                "entry_time_utc",
                "exit_time_utc",
                "direction",
                "entry_price",
                "result",
                "sl_type",
            ]
        ).to_csv(args.debug_output, index=False)

    print(ranking_df.head(30).to_string(index=False))
    print(f"\nSaved ranking: {args.output}")
    print(f"Saved debug trades: {args.debug_output}")


if __name__ == "__main__":
    main()
