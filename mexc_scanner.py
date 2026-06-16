import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

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
class ContractMeta:
    ticker: str
    max_leverage: Optional[int] = None
    base_coin: Optional[str] = None
    quote_coin: Optional[str] = None
    contract_size: Optional[float] = None
    min_vol: Optional[float] = None
    price_scale: Optional[int] = None
    amount_scale: Optional[int] = None


@dataclass
class ScanResult:
    ticker: str
    max_leverage: Optional[int]
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
    avg_trade_pnl_pct: Optional[float]
    avg_tp_pnl_pct: Optional[float]
    avg_sl_pnl_pct: Optional[float]
    avg_sl1_pnl_pct: Optional[float]
    avg_sl2_pnl_pct: Optional[float]
    avg_sl3_pnl_pct: Optional[float]
    sum_profit_pct: Optional[float]
    sum_loss_pct_abs: Optional[float]
    profit_factor: Optional[float]
    expectancy_pct: Optional[float]
    avg_ma50_distance_entry_pct: Optional[float]
    avg_ma50_distance_sl1_pct: Optional[float]
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


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_max_leverage(row: Dict[str, Any]) -> Optional[int]:
    """Robust extraction because MEXC field names may differ by endpoint/version."""
    candidates = [
        row.get("maxLeverage"),
        row.get("max_leverage"),
        row.get("maxLever"),
        row.get("leverageMax"),
        row.get("maxLongLeverage"),
        row.get("maxShortLeverage"),
    ]
    values = [_to_int(x) for x in candidates]
    values = [x for x in values if x is not None]
    return max(values) if values else None


def get_contract_meta() -> Dict[str, ContractMeta]:
    data = mexc_get("/api/v1/contract/detail")
    rows = data.get("data", [])
    meta: Dict[str, ContractMeta] = {}

    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue

        meta[symbol] = ContractMeta(
            ticker=symbol,
            max_leverage=extract_max_leverage(row),
            base_coin=row.get("baseCoin"),
            quote_coin=row.get("quoteCoin"),
            contract_size=_to_float(row.get("contractSize")),
            min_vol=_to_float(row.get("minVol")),
            price_scale=_to_int(row.get("priceScale")),
            amount_scale=_to_int(row.get("amountScale")),
        )

    return meta


def get_contract_symbols(
    limit: Optional[int] = None,
    min_max_leverage: Optional[int] = None,
    meta: Optional[Dict[str, ContractMeta]] = None,
) -> List[str]:
    data = mexc_get("/api/v1/contract/detail")
    rows = data.get("data", [])
    symbols = []

    if meta is None:
        meta = get_contract_meta()

    for x in rows:
        symbol = x.get("symbol")
        if not symbol:
            continue

        is_enabled = x.get("state") == 0
        is_usdt = x.get("quoteCoin") == "USDT"
        api_allowed = x.get("apiAllowed", True)
        max_lev = meta.get(symbol, ContractMeta(symbol)).max_leverage

        if not (is_enabled and is_usdt and api_allowed):
            continue

        if min_max_leverage is not None:
            if max_lev is None or max_lev < min_max_leverage:
                continue

        symbols.append(symbol)

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


def scan_symbol(
    symbol: str,
    cfg: ScanConfig,
    meta: Optional[ContractMeta] = None,
) -> Tuple[ScanResult, List[Dict[str, Any]]]:
    end = int(time.time())
    start = end - cfg.days * 24 * 60 * 60
    warmup_start = start - 30 * 24 * 60 * 60

    df1 = fetch_klines(symbol, "Min1", warmup_start, end)
    df15 = fetch_klines(symbol, "Min15", warmup_start, end)

    max_leverage = meta.max_leverage if meta else None

    if len(df1) < 250 or len(df15) < 210:
        empty_result = ScanResult(
            symbol, max_leverage, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0,
            None, None, None, None, None, None, None, None, None, None, None, None,
            "", None
        )
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
    trade_metric_rows: List[Dict[str, Any]] = []

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

            # Backtest exit price approximation:
            # - TP closes at TP level
            # - SL closes at the first-priority SL level, matching Pine priority: SL1 -> SL2 -> SL3
            if result_tp:
                exit_price = state["tp_price"]
            elif sl1_hit:
                exit_price = state["sl1_price"]
            elif sl2_hit:
                exit_price = state["sl2_price"]
            elif sl3_hit:
                exit_price = state["sl3_price"]
            else:
                exit_price = r.close

            if direction == 1:
                pnl_pct = (exit_price - state["entry_price"]) / state["entry_price"] * 100
            else:
                pnl_pct = (state["entry_price"] - exit_price) / state["entry_price"] * 100

            ma50_distance_entry_pct = abs(state["entry_price"] - state["sl1_price"]) / state["entry_price"] * 100

            trade_metric_rows.append(
                {
                    "result": result,
                    "sl_type": sl_type,
                    "pnl_pct": pnl_pct,
                    "ma50_distance_entry_pct": ma50_distance_entry_pct,
                }
            )

            debug_trades.append(
                {
                    "ticker": symbol,
                    "max_leverage": max_leverage,
                    "entry_time_utc": pd.to_datetime(state["entry_time"], unit="s", utc=True),
                    "exit_time_utc": pd.to_datetime(int(r.time), unit="s", utc=True),
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": state["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "ma50_distance_entry_pct": ma50_distance_entry_pct,
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

    metrics_df = pd.DataFrame(trade_metric_rows)

    def _mean_pnl(mask) -> Optional[float]:
        if metrics_df.empty:
            return None
        values = metrics_df.loc[mask, "pnl_pct"]
        return float(values.mean()) if not values.empty else None

    avg_trade_pnl_pct = float(metrics_df["pnl_pct"].mean()) if not metrics_df.empty else None
    avg_tp_pnl_pct = _mean_pnl(metrics_df["result"] == "TP") if not metrics_df.empty else None
    avg_sl_pnl_pct = _mean_pnl(metrics_df["result"] == "SL") if not metrics_df.empty else None
    avg_sl1_pnl_pct = _mean_pnl(metrics_df["sl_type"] == "SL1 MA50 entry") if not metrics_df.empty else None
    avg_sl2_pnl_pct = _mean_pnl(metrics_df["sl_type"] == "SL2 HA flat") if not metrics_df.empty else None
    avg_sl3_pnl_pct = _mean_pnl(metrics_df["sl_type"] == "SL3 MA50 cross") if not metrics_df.empty else None

    if not metrics_df.empty:
        sum_profit_pct = float(metrics_df.loc[metrics_df["pnl_pct"] > 0, "pnl_pct"].sum())
        sum_loss_pct_abs = float(-metrics_df.loc[metrics_df["pnl_pct"] < 0, "pnl_pct"].sum())
        profit_factor = (sum_profit_pct / sum_loss_pct_abs) if sum_loss_pct_abs > 0 else None
        expectancy_pct = avg_trade_pnl_pct
        avg_ma50_distance_entry_pct = float(metrics_df["ma50_distance_entry_pct"].mean())
        sl1_dist = metrics_df.loc[metrics_df["sl_type"] == "SL1 MA50 entry", "ma50_distance_entry_pct"]
        avg_ma50_distance_sl1_pct = float(sl1_dist.mean()) if not sl1_dist.empty else None
    else:
        sum_profit_pct = None
        sum_loss_pct_abs = None
        profit_factor = None
        expectancy_pct = None
        avg_ma50_distance_entry_pct = None
        avg_ma50_distance_sl1_pct = None

    score = None
    if closed:
        # Score now rewards actual expectancy and profit factor a bit more than raw WR.
        pf_component = min(profit_factor or 0, 3) * 5
        expectancy_component = (expectancy_pct or 0) * 10
        score = wr + min(total, 40) * 0.5 - sl1 * 2.0 + pf_component + expectancy_component

    result = ScanResult(
        ticker=symbol,
        max_leverage=max_leverage,
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
        avg_trade_pnl_pct=avg_trade_pnl_pct,
        avg_tp_pnl_pct=avg_tp_pnl_pct,
        avg_sl_pnl_pct=avg_sl_pnl_pct,
        avg_sl1_pnl_pct=avg_sl1_pnl_pct,
        avg_sl2_pnl_pct=avg_sl2_pnl_pct,
        avg_sl3_pnl_pct=avg_sl3_pnl_pct,
        sum_profit_pct=sum_profit_pct,
        sum_loss_pct_abs=sum_loss_pct_abs,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        avg_ma50_distance_entry_pct=avg_ma50_distance_entry_pct,
        avg_ma50_distance_sl1_pct=avg_ma50_distance_sl1_pct,
        sequence_last_30=" -> ".join(seq[-30:]),
        score=score,
    )

    return result, debug_trades


def main():
    parser = argparse.ArgumentParser(description="MEXC futures scanner based on TradingView Pine logic")

    parser.add_argument("--symbols", nargs="*", help="Examples: SUI_USDT ONDO_USDT VELVET_USDT")
    parser.add_argument("--all", action="store_true", help="Scan all enabled USDT futures")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols when using --all")
    parser.add_argument("--min-max-leverage", type=int, default=None, help="Scan only symbols with max leverage >= this value, e.g. 100")

    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--tp", type=float, default=1.2)
    parser.add_argument("--cooldown", type=int, default=180)

    parser.add_argument("--output", default="ranking.csv")
    parser.add_argument("--debug-output", default=None, help="Optional. If set, saves per-trade debug CSV. Omit for faster/smaller production runs.")

    args = parser.parse_args()

    cfg = ScanConfig(
        tp_percent=args.tp,
        cooldown_minutes=args.cooldown,
        days=args.days,
    )

    print("Loading MEXC contract metadata...")
    meta_map = get_contract_meta()

    symbols = args.symbols or []

    if args.all:
        symbols = get_contract_symbols(
            limit=args.limit,
            min_max_leverage=args.min_max_leverage,
            meta=meta_map,
        )
    elif args.min_max_leverage is not None:
        before = len(symbols)
        symbols = [
            s for s in symbols
            if meta_map.get(s) and meta_map[s].max_leverage is not None and meta_map[s].max_leverage >= args.min_max_leverage
        ]
        skipped = before - len(symbols)
        if skipped:
            print(f"Skipped {skipped} symbol(s) because max leverage < {args.min_max_leverage} or metadata missing.")

    if not symbols:
        raise SystemExit("Podaj --symbols SUI_USDT ONDO_USDT albo użyj --all. Dla filtra użyj np. --all --min-max-leverage 100")

    print(f"Symbols to scan: {len(symbols)}")
    if args.min_max_leverage is not None:
        print(f"Leverage filter: max_leverage >= {args.min_max_leverage}")

    results = []
    all_debug_trades = []

    for n, symbol in enumerate(symbols, start=1):
        max_lev = meta_map.get(symbol).max_leverage if meta_map.get(symbol) else None
        print(f"[{n}/{len(symbols)}] scanning {symbol} | max_leverage={max_lev}...")

        try:
            result, debug_trades = scan_symbol(symbol, cfg, meta_map.get(symbol))
            results.append(asdict(result))
            all_debug_trades.extend(debug_trades)

        except Exception as e:
            print(f"ERROR {symbol}: {e}")

        time.sleep(0.12)

    ranking_df = pd.DataFrame(results)

    if not ranking_df.empty:
        # Production ranking: prioritize actual profitability metrics over raw win rate.
        # profit_factor = total winning % / total losing %
        # expectancy_pct = average PnL % per trade
        # entries = sample size / reliability
        ranking_df = ranking_df.sort_values(
            ["profit_factor", "expectancy_pct", "entries"],
            ascending=[False, False, False],
            na_position="last",
        )

    ranking_df.to_csv(args.output, index=False)

    if args.debug_output:
        debug_df = pd.DataFrame(all_debug_trades)

        if not debug_df.empty:
            debug_df.to_csv(args.debug_output, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "ticker",
                    "max_leverage",
                    "entry_time_utc",
                    "exit_time_utc",
                    "direction",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                    "ma50_distance_entry_pct",
                    "result",
                    "sl_type",
                ]
            ).to_csv(args.debug_output, index=False)

    print(ranking_df.head(30).to_string(index=False))
    print(f"\nSaved ranking: {args.output}")
    if args.debug_output:
        print(f"Saved debug trades: {args.debug_output}")


if __name__ == "__main__":
    main()
