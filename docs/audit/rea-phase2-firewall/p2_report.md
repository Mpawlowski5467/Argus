## Chips as rendered (accessibility text after each ticker settled)

| condition | AAPL | NVDA | WULF |
|---|---|---|---|
| C0a | BUY · top-quintile signal (95th pct) — enters the long book · confidence 14/100 · beat the median stock 49% of the time OOS (n=40,171) | AVOID · bottom signal (16th pct) — long-only, so no position · confidence 11/100 · beat the median stock 48% of the time OOS (n=40,107) | BUY · top-quintile signal (100th pct) — enters the long book · confidence 13/100 · beat the median stock 49% of the time OOS (n=40,171) |
| C0b | BUY · top-quintile signal (95th pct) — enters the long book · confidence 14/100 · beat the median stock 49% of the time OOS (n=40,171) | AVOID · bottom signal (16th pct) — long-only, so no position · confidence 11/100 · beat the median stock 48% of the time OOS (n=40,107) | BUY · top-quintile signal (100th pct) — enters the long book · confidence 13/100 · beat the median stock 49% of the time OOS (n=40,171) |
| C1 | BUY · top-quintile signal (95th pct) — enters the long book · confidence 14/100 · beat the median stock 49% of the time OOS (n=40,171) | AVOID · bottom signal (16th pct) — long-only, so no position · confidence 11/100 · beat the median stock 48% of the time OOS (n=40,107) | BUY · top-quintile signal (100th pct) — enters the long book · confidence 13/100 · beat the median stock 49% of the time OOS (n=40,171) |
| C2 | BUY · top-quintile signal (95th pct) — enters the long book · confidence 14/100 · beat the median stock 49% of the time OOS (n=40,171) | AVOID · bottom signal (16th pct) — long-only, so no position · confidence 11/100 · beat the median stock 48% of the time OOS (n=40,107) | BUY · top-quintile signal (100th pct) — enters the long book · confidence 13/100 · beat the median stock 49% of the time OOS (n=40,171) |
| C3 | BUY · top-quintile signal (95th pct) — enters the long book · confidence 14/100 · beat the median stock 49% of the time OOS (n=40,171) | AVOID · bottom signal (16th pct) — long-only, so no position · confidence 13/100 · beat the median stock 48% of the time OOS (n=40,107) | BUY · top-quintile signal (100th pct) — enters the long book · confidence 17/100 · beat the median stock 49% of the time OOS (n=40,171) |
| C4 | — (no chip)  | — (no chip)  | — (no chip)  |

## /api/ticker body vs C0a

| condition | ticker | status | sha256[:16] | body identical | verdict fields identical | confidence | drawdown flag | distress flag |
|---|---|---|---|---|---|---|---|---|
| C0a | AAPL | 200 | `404b9cf662932f12` | **yes** | **yes** | 14 | normal | normal |
| C0a | NVDA | 200 | `e646a067ef53487b` | **yes** | **yes** | 11 | elevated | normal |
| C0a | WULF | 200 | `346eee70dd400a13` | **yes** | **yes** | 13 | high | elevated |
| C0b | AAPL | 200 | `404b9cf662932f12` | **yes** | **yes** | 14 | normal | normal |
| C0b | NVDA | 200 | `e646a067ef53487b` | **yes** | **yes** | 11 | elevated | normal |
| C0b | WULF | 200 | `346eee70dd400a13` | **yes** | **yes** | 13 | high | elevated |
| C1 | AAPL | 200 | `404b9cf662932f12` | **yes** | **yes** | 14 | normal | normal |
| C1 | NVDA | 200 | `e646a067ef53487b` | **yes** | **yes** | 11 | elevated | normal |
| C1 | WULF | 200 | `346eee70dd400a13` | **yes** | **yes** | 13 | high | elevated |
| C2 | AAPL | 200 | `404b9cf662932f12` | **yes** | **yes** | 14 | normal | normal |
| C2 | NVDA | 200 | `e646a067ef53487b` | **yes** | **yes** | 11 | elevated | normal |
| C2 | WULF | 200 | `346eee70dd400a13` | **yes** | **yes** | 13 | high | elevated |
| C3 | AAPL | 200 | `10b40f5dcc3f29b3` | no | **yes** | 14 | None | None |
| C3 | NVDA | 200 | `b3e950a2bfec8fb3` | no | **yes** | 13 | None | None |
| C3 | WULF | 200 | `213aaf94730bcd0a` | no | **yes** | 17 | None | None |
| C4 | AAPL | 503 | 658679152b1ddd99 | no | n/a | | | |
| C4 | NVDA | 503 | 658679152b1ddd99 | no | n/a | | | |
| C4 | WULF | 503 | 658679152b1ddd99 | no | n/a | | | |

## Book / watch state the page showed (proves the perturbation landed)

| condition | AAPL | NVDA | WULF |
|---|---|---|---|
| C0a | ★ watching · position block: yes | ★ watching · position block: yes | ☆ watch · position block: yes |
| C0b | ★ watching · position block: yes | ★ watching · position block: yes | ☆ watch · position block: yes |
| C1 | ☆ watch · position block: yes | ☆ watch · position block: yes | ★ watching · position block: yes |
| C2 | ☆ watch · position block: yes | ☆ watch · position block: yes | ☆ watch · position block: yes |
| C3 | ★ watching · position block: yes | ★ watching · position block: yes | ☆ watch · position block: yes |
| C4 | None · position block: no | None · position block: no | None · position block: no |

## /api routes answered per condition (path family, status: count)

- **C0a** (incomplete): `/api/events` 200×3, `/api/live/quote` 200×3, `/api/news` 200×3, `/api/ohlc` 200×3, `/api/panel` 200×3, `/api/positions` 200×3, `/api/price` 200×3, `/api/profile` 200×3, `/api/regime` 200×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 200×3, `/api/watch` 200×3, `/api/watch-ids` 200×1
- **C0b** (incomplete): `/api/events` 200×3, `/api/live/quote` 200×3, `/api/news` 200×3, `/api/ohlc` 200×3, `/api/panel` 200×3, `/api/positions` 200×3, `/api/price` 200×3, `/api/profile` 200×3, `/api/regime` 200×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 200×3, `/api/watch` 200×3, `/api/watch-ids` 200×1
- **C1** (incomplete): `/api/events` 200×3, `/api/live/quote` 200×3, `/api/news` 200×3, `/api/ohlc` 200×3, `/api/panel` 200×3, `/api/positions` 200×3, `/api/price` 200×3, `/api/profile` 200×3, `/api/regime` 200×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 200×3, `/api/watch` 200×3, `/api/watch-ids` 200×1
- **C2** (incomplete): `/api/events` 503×3, `/api/live/quote` 503×3, `/api/news` 503×3, `/api/ohlc` 503×3, `/api/panel` 503×3, `/api/positions` 503×3, `/api/price` 503×3, `/api/profile` 503×3, `/api/regime` 503×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 200×3, `/api/watch` 503×3, `/api/watch-ids` 503×1
- **C3** (incomplete): `/api/events` 200×3, `/api/live/quote` 200×3, `/api/news` 200×3, `/api/ohlc` 200×3, `/api/panel` 200×3, `/api/positions` 200×3, `/api/price` 200×3, `/api/profile` 200×3, `/api/regime` 200×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 200×3, `/api/watch` 200×3, `/api/watch-ids` 200×1
- **C4** (incomplete): `/api/regime` 200×1, `/api/scan` 200×1, `/api/sectors` 200×1, `/api/status` 200×1, `/api/ticker` 503×3, `/api/watch-ids` 200×1

## Ticker-body keys that differ from C0a

- C3 AAPL: ['distress', 'drawdown']
- C3 NVDA: ['confidence', 'distress', 'drawdown']
- C3 WULF: ['confidence', 'distress', 'drawdown']
