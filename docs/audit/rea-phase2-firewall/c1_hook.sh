set -e
B=http://127.0.0.1:8024/api
curl -s -X POST $B/positions/320193 -H 'content-type: application/json' -d '{"shares": 5000, "cost": 1.00}' >/dev/null
curl -s -X DELETE $B/positions/1045810 >/dev/null
curl -s -X POST $B/positions/1083301 -H 'content-type: application/json' -d '{"shares": 777, "cost": 3.21}' >/dev/null
curl -s -X POST $B/watch/320193/toggle >/dev/null; curl -s -X POST $B/watch/1045810/toggle >/dev/null; curl -s -X POST $B/watch/1083301/toggle >/dev/null
source "/private/tmp/claude-501/-Users-mpawlowski-Documents-Personal-stock-analysis/654c9bb6-e3ac-49db-8ca4-82cad0fb9689/scratchpad/argus-demo/demo.env.sh"; cd "/Users/mpawlowski/Documents/Personal/stock-analysis" && uv run --extra web python -I -c "
import sys; sys.path.insert(0, 'src')
from stockscan.ops.state import OpsState
from stockscan.config import OPS_STATE_PATH
with OpsState(OPS_STATE_PATH) as st:
    st.add_alert('distress_risk', 'AAPL distress risk HIGH (audit perturbation)', cik=320193)
    st.add_alert('drawdown_risk', 'NVDA drawdown risk HIGH (audit perturbation)', cik=1045810)
print('book perturbed')"
