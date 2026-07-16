# MTF Regime Diagnostic Scanner

Skaner odtwarza strategię wielointerwałową na danych kontraktów futures MEXC:

- **M15** — wybór reżimu LONG/SHORT,
- **M5** — wykrywanie pullbacku,
- **M1** — trigger wybiciowy,
- **SL** — swing z M5 z buforem ATR,
- **TP** — domyślnie 2R,
- prowizja i poślizg uwzględnione w wynikach,
- opcjonalne wyjście po utracie reżimu M15.

Skaner używa publicznych danych rynkowych MEXC. Klucze API nie są wymagane.

## Pliki

```text
mtf_regime_diagnostic.py
requirements.txt

.github/
└── workflows/
    └── mtf_regime_diagnostic.yml
```

## Instalacja lokalna

Wymagany Python 3.11.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Test wybranych tickerów

```bash
python -u mtf_regime_diagnostic.py \
  --symbols ETH_USDT SUI_USDT BEAT_USDT \
  --days 30 \
  --rr 2.0
```

## Test wielu kontraktów

Pierwsze 50 aktywnych kontraktów USDT:

```bash
python -u mtf_regime_diagnostic.py \
  --all \
  --limit 50 \
  --days 30 \
  --rr 2.0
```

Wszystkie aktywne kontrakty USDT:

```bash
python -u mtf_regime_diagnostic.py \
  --all \
  --days 30 \
  --rr 2.0
```

## Wyjście po utracie reżimu M15

Domyślnie wyjście po utracie reżimu jest włączone.

Wariant bez wcześniejszego wyjścia, wyłącznie SL albo TP:

```bash
python -u mtf_regime_diagnostic.py \
  --symbols ETH_USDT SUI_USDT BEAT_USDT \
  --days 30 \
  --rr 2.0 \
  --no-regime-exit
```

## Generowane pliki

### `ranking_mtf.csv`

Podsumowanie wyników każdego tickera:

- liczba transakcji,
- TP, SL i wyjścia po utracie reżimu,
- win rate,
- profit factor,
- expectancy w R,
- łączny wynik w R,
- maksymalny drawdown,
- serie strat,
- średnie MFE i MAE.

### `trades_mtf.csv`

Szczegóły każdej zamkniętej transakcji:

- czas i cena wejścia,
- czas i cena wyjścia,
- kierunek,
- SL i TP,
- powód zamknięcia,
- wynik brutto i netto,
- wynik w R,
- MFE i MAE,
- parametry M15, M5 i M1 w momencie wejścia.

### `rejected_setups.csv`

Potencjalne setupy, które nie zostały otwarte, wraz z powodem:

- brak reżimu M15,
- brak pullbacku M5,
- aktywna pozycja,
- cooldown,
- SL zbyt szeroki lub zbyt wąski,
- wygaśnięcie pullbacku bez triggera M1.

### `equity_curve_mtf.csv`

Kolejne wyniki transakcji, skumulowany wynik w R i drawdown.

### `open_trades_mtf.csv`

Pozycje, które pozostały otwarte na końcu badanego okresu.

### `run_config.json`

Parametry użyte podczas konkretnego uruchomienia skanera.

## GitHub Actions

Workflow znajduje się tutaj:

```text
.github/workflows/mtf_regime_diagnostic.yml
```

Uruchomienie ręczne:

```text
GitHub → Actions → MTF Regime Diagnostic Scanner → Run workflow
```

Pole `symbols`:

```text
ETH_USDT SUI_USDT BEAT_USDT
```

Pozostawienie pola pustego uruchamia skanowanie wielu aktywnych kontraktów USDT.

## Zalecany pierwszy test

```text
Tickery: ETH_USDT SUI_USDT BEAT_USDT BTC_USDT
Okres: 30 dni
RR: 2.0
Same-candle policy: conservative
Limit: 0 dla wybranych tickerów
```

Najlepiej wykonać dwa osobne uruchomienia:

1. z włączonym wyjściem po utracie reżimu,
2. z opcją `--no-regime-exit`.

Porównanie pokaże, czy problem leży w sygnałach wejścia, czy we wcześniejszym zamykaniu pozycji.
