# MEXC Scanner

Pierwsza wersja skanera odtwarza logikę Pine Script:
- MA12/MA50/MA200 na 1m,
- Heikin Ashi z 1m,
- filtr M15,
- MA200 raw,
- cooldown,
- TP,
- SL1 / SL2 / SL3,
- ranking CSV.

## Lokalny test

```bash
pip install -r requirements.txt
python mexc_scanner.py --symbols SUI_USDT ONDO_USDT VELVET_USDT ESPORTS_USDT BRETT_USDT --days 14 --output ranking.csv
```

## Test ograniczony wszystkich futures

```bash
python mexc_scanner.py --all --limit 20 --days 14 --output ranking.csv
```

## Pełny scan

```bash
python mexc_scanner.py --all --days 14 --output ranking.csv
```

Uwaga: pełny scan pobiera dużo świec 1m, więc na Render może wymagać Cron Job i cierpliwości.
