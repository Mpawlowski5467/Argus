#!/usr/bin/env bash
# One firewall condition, self-contained: fresh demo server -> optional setup hook ->
# fault proxy -> REA chips scenario -> teardown.   cond.sh <name> <block-regex|""> [setup-hook]
set -uo pipefail
NAME="$1"; BLOCK="${2:-}"; HOOK="${3:-}"
P="$(cd "$(dirname "$0")" && pwd)"; DEMO="${DEMO:-$(dirname "$P")}"   # the demo_setup.sh dir
REPO=/Users/mpawlowski/Documents/Personal/stock-analysis
REA=/Users/mpawlowski/.npm/_npx/ad9a3fdbcad992c6/node_modules/.bin/rea
for port in 8024 8025; do pid=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t); [ -n "$pid" ] && kill $pid; done; sleep 1
( source "$DEMO/demo.env.sh"; cd "$REPO" && exec uv run --extra web python scripts/argus_web.py ) > "$P/$NAME.server.log" 2>&1 &
for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:8024/api/status)" = 200 ] && break; sleep 1; done
[ -n "$HOOK" ] && bash -c "$HOOK"
rm -f "$P/$NAME.timing.jsonl"
python3 -I "$P/fault_proxy.py" 8025 8024 "$P/$NAME.timing.jsonl" "$BLOCK" &
PROXY=$!; sleep 0.7
"$REA" capture-browser-scenario "$P/chips.json" --format json --full-output > "$P/$NAME.result.json" 2> "$P/$NAME.stderr"
echo "$NAME rea exit=$?"
kill $PROXY; pid=$(lsof -nP -iTCP:8024 -sTCP:LISTEN -t); [ -n "$pid" ] && kill $pid; sleep 1
