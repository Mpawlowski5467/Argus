## Steps

| # | step | action | elapsed | status |
|---|---|---|---|---|
| 0 | `scenario_start` | goto_start | 1217 ms | completed |
| 1 | `s01-scan-ready` | wait_for | 1846 ms | completed |
| 2 | `s02-search-aapl` | fill | 14 ms | completed |
| 3 | `s03-search-settle` | wait_for_timeout | 1501 ms | completed |
| 4 | `s04-open-ticker` | click | 84 ms | completed |
| 5 | `s05-ticker-head` | wait_for | 82 ms | completed |
| 6 | `s06-ticker-settle` | wait_for_timeout | 5001 ms | completed |
| 7 | `s07-narrate` | click | 45 ms | completed |
| 8 | `s08-narrate-done` | wait_for | 180.4 s | completed |
| 9 | `s09-panel` | click | 32 ms | completed |
| 10 | `s10-panel-done` | wait_for | 30.2 s | completed |
| 11 | `s11-ask-fill` | fill | 3 ms | completed |
| 12 | `s12-ask-send` | click | 30 ms | completed |
| 13 | `s13-ask-done` | wait_for | 7836 ms | completed |
| 14 | `s14-book` | click | 34 ms | completed |
| 15 | `s15-book-ready` | wait_for | 26 ms | completed |
| 16 | `s16-markets` | click | 37 ms | completed |
| 17 | `s17-markets-ready` | wait_for | 178 ms | completed |
| 18 | `s18-markets-settle` | wait_for_timeout | 4002 ms | completed |
| 19 | `s19-paper` | click | 39 ms | completed |
| 20 | `s20-paper-ready` | wait_for | 789 ms | completed |
| 21 | `s21-watch` | click | 42 ms | completed |
| 22 | `s22-watch-ready` | wait_for | 181 ms | completed |
| 23 | `s23-watch-settle` | wait_for_timeout | 4041 ms | completed |

## Every request (browser order)

| # | step | method | route | status | duration | resp size | req size |
|---|---|---|---|---|---|---|---|
| 1 | `scenario_start` | GET | `/` | 200 | 4 ms | 6.4 KB | — |
| 2 | `scenario_start` | GET | `/styles.css` | 200 | 4 ms | 19.1 KB | — |
| 3 | `scenario_start` | GET | `/charts.js` | 200 | 4 ms | 9.5 KB | — |
| 4 | `scenario_start` | GET | `/app.js` | 200 | 4 ms | 83.6 KB | — |
| 5 | `scenario_start` | GET | `/argus-mandala.html` | 200 | 2 ms | 35.2 KB | — |
| 6 | `scenario_start` | GET | `/api/status` | 200 | 21 ms | 171 B | — |
| 7 | `s01-scan-ready` | GET | `/api/sectors` | 200 | 12 ms | 1.1 KB | — |
| 8 | `s01-scan-ready` | GET | `/api/scan?sector=all` | 200 | 126 ms | 536.8 KB | — |
| 9 | `s01-scan-ready` | GET | `/api/watch-ids` | 200 | 6 ms | 48 B | — |
| 10 | `s01-scan-ready` | GET | `/api/regime` | 200 | 268 ms | 406 B | — |
| 11 | `s01-scan-ready` | GET | `/spacemono.woff2` | 200 | 20 ms | 16.1 KB | — |
| 12 | `s03-search-settle` | GET | `/api/search?q=AAPL` | 200 | 5 ms | 157 B | — |
| 13 | `s04-open-ticker` | GET | `/api/ticker/320193` | 200 | 65 ms | 5.2 KB | — |
| 14 | `s05-ticker-head` | GET | `/api/panel/320193` | 200 | 48 ms | 124 B | — |
| 15 | `s05-ticker-head` | GET | `/api/price/320193` | 200 | 19 ms | 147.9 KB | — |
| 16 | `s05-ticker-head` | GET | `/api/ohlc/320193` | 200 | 28 ms | 14.5 KB | — |
| 17 | `s05-ticker-head` | GET | `/api/watch/320193` | 200 | 23 ms | 29 B | — |
| 18 | `s05-ticker-head` | GET | `/api/positions` | 200 | 28 ms | 262 B | — |
| 19 | `s05-ticker-head` | GET | `/api/profile/320193` | 200 | 23 ms | 2.2 KB | — |
| 20 | `s05-ticker-head` | GET | `/api/events/320193` | 200 | 189 ms | 758 B | — |
| 21 | `s05-ticker-head` | GET | `/api/news/320193` | 200 | 20 ms | 4.6 KB | — |
| 22 | `s05-ticker-head` | GET | `/api/live/quote/320193` | 200 | 13 ms | 4 B | — |
| 23 | `s07-narrate` | POST | `/api/narrate/320193` | 200 | 180.0 s | 5.5 KB | 2 B |
| 24 | `s09-panel` | POST | `/api/panel/320193` | 200 | 4326 ms | 1.3 KB | 15 B |
| 25 | `s10-panel-done` | POST | `/api/panel/320193` | 200 | 10.1 s | 1.4 KB | 15 B |
| 26 | `s10-panel-done` | POST | `/api/panel/320193` | 200 | 4535 ms | 1.1 KB | 15 B |
| 27 | `s10-panel-done` | POST | `/api/panel/320193` | 200 | 10.9 s | 1.9 KB | 20 B |
| 28 | `s12-ask-send` | POST | `/api/ask/320193` | 200 | 7401 ms | 1.0 KB | 50 B |
| 29 | `s14-book` | GET | `/api/scorecard` | 200 | 26 ms | 3.3 KB | — |
| 30 | `s16-markets` | GET | `/api/markets` | 200 | 97 ms | 40.1 KB | — |
| 31 | `s17-markets-ready` | POST | `/api/market-caps` | 200 | 58 ms | 8.2 KB | 3.4 KB |
| 32 | `s19-paper` | GET | `/api/paper` | 200 | 318 ms | 906 B | — |
| 33 | `s21-watch` | GET | `/api/digest` | 200 | 252 ms | 731 B | — |
| 34 | `s21-watch` | GET | `/api/health` | 200 | 67 ms | 1.8 KB | — |
| 35 | `s21-watch` | GET | `/api/watch` | 200 | 84 ms | 1015 B | — |

## Slowest calls

| route | step | duration | resp size |
|---|---|---|---|
| POST `/api/narrate/320193` | `s07-narrate` | 180.0 s | 5.5 KB |
| POST `/api/panel/320193` | `s10-panel-done` | 10.9 s | 1.9 KB |
| POST `/api/panel/320193` | `s10-panel-done` | 10.1 s | 1.4 KB |
| POST `/api/ask/320193` | `s12-ask-send` | 7401 ms | 1.0 KB |
| POST `/api/panel/320193` | `s10-panel-done` | 4535 ms | 1.1 KB |
| POST `/api/panel/320193` | `s09-panel` | 4326 ms | 1.3 KB |
| GET `/api/paper` | `s19-paper` | 318 ms | 906 B |
| GET `/api/regime` | `s01-scan-ready` | 268 ms | 406 B |
| GET `/api/digest` | `s21-watch` | 252 ms | 731 B |
| GET `/api/events/320193` | `s05-ticker-head` | 189 ms | 758 B |
| GET `/api/scan?sector=all` | `s01-scan-ready` | 126 ms | 536.8 KB |
| GET `/api/markets` | `s16-markets` | 97 ms | 40.1 KB |

## Repeated routes (same method+path more than once)

| route | count | steps |
|---|---|---|
| POST `/api/panel/320193` | 4 | s09-panel, s10-panel-done |

## Console / page errors

- console.error (step `s01-scan-ready`): Failed to load resource: the server responded with a status of 404 (Not Found) — http://127.0.0.1:8025/favicon.ico

_proxy rows not matched to a browser request: 1_
- GET `/favicon.ico` 404 2 ms error=None

_total: 35 requests, 952.3 KB response bytes; scenario duration 237.8 s_
