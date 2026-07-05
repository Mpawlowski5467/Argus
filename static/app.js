/* argus-web — router, fetch layer, view controllers.
   loader → tabs (scan · watch · book · paper · markets); click any name for the
   company view. "?" opens help; the .term glossary explains every number on hover. */
(function () {
  "use strict";

  // -- tiny helpers ----------------------------------------------------------
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var el = function (id) { return document.getElementById(id); };
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function api(p) { return fetch("/api" + p).then(function (r) { if (!r.ok) throw Object.assign(new Error(r.status), { status: r.status, res: r }); return r.json(); }); }
  function apiPost(p, body) { return fetch("/api" + p, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) }).then(function (r) { return r.json(); }); }
  function apiDelete(p) { return fetch("/api" + p, { method: "DELETE" }).then(function (r) { return r.json(); }); }
  function money(x) { if (x == null || !isFinite(x)) return "—"; return "$" + Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
  function plMoney(x) { if (x == null || !isFinite(x)) return "—"; return (x >= 0 ? "+" : "−") + money(Math.abs(x)); }
  function pctColor(x) { return x == null ? "muted" : x >= 0 ? "pos" : "neg"; }
  function sign(x, d) { if (x == null || !isFinite(x)) return "—"; return (x >= 0 ? "+" : "") + x.toFixed(d == null ? 1 : d) + "%"; }
  function fmtCap(x) {
    if (x == null) return "…";
    if (x >= 1e12) return "$" + (x / 1e12).toFixed(2) + "T";
    if (x >= 1e9) return "$" + (x / 1e9).toFixed(1) + "B";
    if (x >= 1e6) return "$" + (x / 1e6).toFixed(0) + "M";
    return "$" + Math.round(x).toLocaleString();
  }

  var app = el("app"), loader = el("loader");
  var state = { view: "scan", cik: null, chart: "candle", tk: null, auto: null, marketsLoaded: false, detailTarget: "tk-body", narrating: null };
  function heat(d) { d = d || 0; if (d >= 9) return "#3f6b4e"; if (d >= 7) return "#4a7359"; if (d >= 5) return "#484848"; if (d >= 3) return "#78504d"; return "#693f3f"; }

  // -- glossary: ONE source of truth for what every number means (deterministic; no
  // LLM, no latency, nothing to fabricate). Any element with class="term" and a
  // data-g key gets the matching plain-English tooltip on hover — including static
  // markup in index.html. Wording keeps the house honesty rules: peer rank ≠
  // forecast; risk heads are display-only; confidence never decouples from its
  // hit-rate.
  var GLOSS = {
    model: "The model signal is a RELATIVE rank from a frozen statistical model — where this name's score lands among all scored peers today. Not a predicted return, not a price target.",
    pct: "Percentile, 0–100 — the share of scored peers this name ranks above today. 90th = ahead of 90% of the market. A peer rank, not a probability of gains.",
    decile: "The percentile bucketed 1–10 (10 = top tenth of all scored names). The backtest's long book buys from the top quintile — deciles 9–10.",
    score: "The frozen model's raw output. Only its RANK against peers means anything — the magnitude has no unit and no dollar meaning.",
    shap: "SHAP splits this exact score into per-fundamental contributions that sum to it — which inputs pushed the rank up or down. An attribution of the rank, not a forecast.",
    signals: "Pre-computed fundamentals with sector percentiles. Each carries its own read (supports / detracts) decided by the pipeline — a HIGH percentile on a lower-is-better signal like leverage counts AGAINST the name.",
    confidence: "How much to trust this call: 0–100, derived from how often this decile actually beat the market out-of-sample (the hit-rate shown beside it). Capped — never certainty.",
    distress: "A learned probability of distress or delisting within ~12 months, from a separate firewalled model. A display-only risk flag — never a trade input.",
    drawdownRisk: "A learned probability of a deep (~30%+) peak-to-trough fall within ~6 months. A display-only risk flag — never a trade input.",
    ic: "Prediction accuracy = rank correlation between predicted and realized returns for a month. 0 is a coin-flip; small positive numbers are normal — the sign and the trend matter, not the size.",
    insample: "This date falls inside the model's training window, so the score may flatter itself. Only out-of-sample months are the honest test.",
    stale: "How old the filing behind this analysis is. Fundamentals move slowly, but a very stale 10-K means the model is reading old news.",
    adv: "Average daily dollar volume — how much of this name trades per day. The universe floor screens out names too illiquid to trade.",
    equalweight: "Every tracked name counts the same, regardless of position size — the spread-out view of your book.",
    valueweight: "Names weighted by the dollar value of your position — the view of where your money actually sits.",
  };
  function term(key, label) { return '<span class="term" data-g="' + key + '">' + label + "</span>"; }
  (function () {   // one shared tooltip, event-delegated so re-rendered views keep working
    var tip = document.createElement("div");
    tip.id = "gloss-tip"; tip.hidden = true;
    document.body.appendChild(tip);
    document.addEventListener("mouseover", function (e) {
      var t = e.target && e.target.closest ? e.target.closest(".term") : null;
      var g = t && GLOSS[t.dataset.g];
      if (!g) { tip.hidden = true; return; }
      tip.textContent = g; tip.hidden = false;
      var r = t.getBoundingClientRect();
      var y = r.bottom + 6;
      if (y + tip.offsetHeight > window.innerHeight - 8) y = r.top - tip.offsetHeight - 6;
      tip.style.left = Math.max(8, Math.min(r.left, window.innerWidth - tip.offsetWidth - 12)) + "px";
      tip.style.top = y + "px";
    });
  })();

  // -- loader → app handshake ------------------------------------------------
  var started = Date.now();
  function loaderMsg(msg, tone) {
    var d = el("loader-err");
    if (!d) {
      d = document.createElement("div");
      d.id = "loader-err";
      d.style.cssText = "position:absolute;left:0;right:0;bottom:14%;text-align:center;font:13px 'Space Mono',monospace;padding:0 20px";
      loader.appendChild(d);
    }
    d.style.color = tone === "err" ? "#c68a86" : "#8a8a8a";
    d.textContent = msg;
  }
  function loaderError(msg) { loaderMsg(msg, "err"); }
  function poll() {
    fetch("/api/status").then(function (r) { return r.json().then(function (b) { return { ok: r.ok, b: b }; }); })
      .then(function (s) {
        if (s.b && s.b.error) loaderError("load failed — " + s.b.error);
        if (s.ok && s.b && s.b.loading === false) { boot(s.b); return; }
        setTimeout(poll, 500);
      })
      .catch(function () { setTimeout(poll, 700); });
  }
  function boot(status) {
    renderStatus(status);
    Promise.all([api("/sectors"), api("/scan?sector=all")]).then(function (r) {
      Scan.initSectors(r[0]); Scan.render(r[1]);
      var wait = Math.max(0, 1400 - (Date.now() - started));
      setTimeout(reveal, wait);
    }).catch(function (e) { loaderError("boot error — " + e.message); });
  }
  function reveal() {
    app.hidden = false;
    loader.classList.add("hide");
    setTimeout(function () { loader.style.display = "none"; }, 750);
    switchView("scan");
  }

  function renderStatus(s) {
    el("s-asof").textContent = s.as_of || "—";
    el("s-fund").textContent = s.fund_quarter || "—";
    el("s-vintage").textContent = s.vintage || "—";
    el("s-vintage").className = s.artifact_registered ? "ok" : "bad";
    var nm = { ok: "ok", noop: "ok", degraded: "warn", failed: "bad" }[s.nightly_status] || "";
    el("s-nightly").textContent = s.nightly_status || "never"; el("s-nightly").className = nm;
    el("s-alerts").textContent = s.unseen_alerts || 0; el("s-alerts").className = s.unseen_alerts ? "warn" : "";
  }

  // -- view switching --------------------------------------------------------
  function switchView(v) {
    state.view = v;
    ["scan", "ticker", "watch", "scorecard", "paper", "markets"].forEach(function (name) {
      el("view-" + name).hidden = name !== v;
      var t = $('.tab[data-view="' + name + '"]'); if (t) t.classList.toggle("active", name === v);
    });
    if (v === "markets" && !state.marketsLoaded) Markets.load();
    if (v === "watch") Watch.load();
    if (v === "scorecard") Scorecard.load();
    if (v === "paper") Paper.load();
  }
  Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (t) {
    t.addEventListener("click", function () { switchView(t.dataset.view); });
  });

  // -- SCAN ------------------------------------------------------------------
  var Scan = {
    initSectors: function (secs) {
      el("sector").innerHTML = secs.map(function (s) { return '<option value="' + esc(s) + '">' + esc(s) + "</option>"; }).join("");
    },
    render: function (rows) {
      var b = $("#scan-table tbody");
      if (!rows.length) { b.innerHTML = '<tr><td colspan="7" class="empty">no matches</td></tr>'; return; }
      b.innerHTML = rows.map(function (r) {
        return '<tr data-cik="' + r.cik + '"><td class="num">' + r.rank + '</td><td class="tk">' + esc(r.ticker) +
          "</td><td>" + esc(r.name) + "</td><td class=\"muted\">" + esc(r.sector) + '</td><td class="num">' + r.pct +
          '%</td><td class="num">' + r.decile + '</td><td class="num muted">' + (r.fy || "—") + "</td></tr>";
      }).join("");
      Array.prototype.forEach.call(b.querySelectorAll("tr"), function (tr) {
        tr.addEventListener("click", function () { Ticker.open(+tr.dataset.cik); });
      });
    },
    markSelected: function (cik) {
      Array.prototype.forEach.call(document.querySelectorAll("#scan-table tbody tr"), function (tr) {
        tr.classList.toggle("sel", +tr.dataset.cik === cik);
      });
    },
    query: function () {
      var q = el("search").value.trim(), sec = el("sector").value;
      var p = q ? "/search?q=" + encodeURIComponent(q) : "/scan?sector=" + encodeURIComponent(sec);
      api(p).then(Scan.render);
    },
    submit: function () {
      var q = el("search").value.trim(); if (!q) return;
      api("/resolve?q=" + encodeURIComponent(q)).then(function (hit) { Ticker.open(hit.cik); })
        .catch(function () { Scan.render([]); });
    },
  };
  var debounce; el("search").addEventListener("input", function () { clearTimeout(debounce); debounce = setTimeout(Scan.query, 180); });
  el("search").addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); Scan.submit(); } if (e.key === "Escape") el("search").blur(); });
  el("sector").addEventListener("change", Scan.query);

  // -- TICKER ----------------------------------------------------------------
  var Ticker = {
    open: function (cik) {
      state.cik = cik;
      // remember where we came from so the detail view's ← back returns there
      if (state.view && state.view !== "ticker") state.cameFrom = state.view;
      state.detailTarget = "tk-body";
      switchView("ticker");
      el("tk-body").innerHTML = '<div class="muted">loading …</div>';
      api("/ticker/" + cik).then(function (res) {
        state.tk = { res: res, price: null, ohlc: null, quote: null, profile: null, events: null, news: null, watched: false, narr: null, position: null, posDraft: null, chat: [], askDraft: "", askNote: null };
        Ticker.render();
        Promise.all([
          api("/price/" + cik).then(function (d) { state.tk.price = d; }).catch(nop),
          api("/ohlc/" + cik).then(function (d) { state.tk.ohlc = d.ohlc; }).catch(nop),
          api("/watch/" + cik).then(function (d) { state.tk.watched = d.watched; }).catch(nop),
          api("/positions").then(function (list) { state.tk.position = (list || []).filter(function (p) { return +p.cik === +cik; })[0] || null; }).catch(nop),
        ]).then(Ticker.render);
        // live-view enrichment (fired after the packet; each fails open)
        api("/profile/" + cik).then(function (d) { state.tk.profile = d || {}; Ticker.render(); }).catch(nop);
        api("/events/" + cik).then(function (d) { state.tk.events = d; Ticker.render(); }).catch(nop);
        api("/news/" + cik).then(function (d) { state.tk.news = d; Ticker.render(); }).catch(nop);
        api("/live/quote/" + cik).then(function (d) { state.tk.quote = d; Ticker.render(); }).catch(nop);
      }).catch(function (e) {
        var msg = e.res ? "unavailable" : e.message;
        e.res && e.res.json && e.res.json().then(function (j) { el(state.detailTarget).innerHTML = '<div class="err">' + esc(j.detail || msg) + "</div>"; });
        if (!e.res) el(state.detailTarget).innerHTML = '<div class="err">' + esc(msg) + "</div>";
      });
    },
    render: function () {
      var t = state.tk; if (!t) return;
      var r = t.res, pk = r.packet, m = pk.meta, f = r.flags || {}, v = r.verdict || {};
      var model = pk.model || {};
      var pct = r.percentile != null ? r.percentile : model.percentile;
      var dec = r.decile != null ? r.decile : model.decile;
      var score = r.score != null ? r.score : model.score;
      var trained = r.trained_through || model.trained_through || m.trained_through;
      var h = [];
      // ← back to wherever we opened this from (scan / watch / book / markets)
      h.push('<div class="tk-nav"><button class="mini" id="tk-back">← back</button></div>');
      // header
      h.push('<div class="tk-head"><span class="name">' + esc(m.name) + '</span><span class="sep">·</span><span class="tk">' +
        esc(m.ticker || "") + '</span><span class="sep">·</span><span class="muted">' + esc(m.sector || "") + '</span>' +
        '<span class="watch-star ' + (t.watched ? "" : "off") + '" id="wstar">' + (t.watched ? "★ watching" : "☆ watch") + "</span></div>");
      // verdict + confidence (how much to trust this call, from the model's OOS record)
      h.push('<div class="verdict-row"><span class="badge ' + (v.color || "dim") + '">' + esc(v.call || "N/A") +
        '</span><span class="reason">' + esc(v.reason || "") + "</span>" + Ticker.confidenceChip(r) + "</div>");
      // profile
      h.push(Ticker.profileBlock(t.profile));
      // price + live + chart
      h.push(Ticker.priceBlock(t));
      // your position (personal live-view: value + P/L + three honest reads)
      h.push(Ticker.positionBlock(t));
      // filing + model line
      if (f.filed_date) h.push('<div class="muted" style="margin-top:12px">FY' + esc(m.fiscal_year) + " 10-K filed " + esc(f.filed_date) +
        " (usable " + esc(f.available_date) + ", " + term("stale", esc(f.staleness_days) + "d old") + ")</div>");
      var insample = f.in_sample ? ' <span class="warn">' + term("insample", "[in-sample]") + "</span>" : "";
      var liq = f.liquidity_pass === false ? ' <span class="neg">' + term("adv", "[below liquidity floor]") + "</span>" : "";
      h.push('<div style="margin-top:8px">' + term("model", "model signal") + " <b>" + (pct != null ? term("pct", pct + "th pct") : "—") + "</b> · " +
        term("decile", "decile " + (dec || "—") + "/10") + " · " +
        term("score", "score " + (score != null ? (score >= 0 ? "+" : "") + Number(score).toFixed(4) : "—")) +
        (trained ? " · trained through " + esc(trained) : "") + insample + liq + "</div>");
      // drivers (SHAP) if present
      var drivers = pk.drivers || model.drivers;
      if (drivers && drivers.length) {
        h.push('<div class="section-h">' + term("shap", "drivers (SHAP — exact decomposition)") + '</div><div class="drivers">');
        var mx = Math.max.apply(null, drivers.map(function (d) { return Math.abs(d.contribution || 0); })) || 1;
        drivers.forEach(function (d) {
          var pos = (d.direction || "").indexOf("support") === 0 || (d.contribution || 0) >= 0;
          var wpx = Math.max(2, Math.round(Math.abs(d.contribution || 0) / mx * 180));
          h.push('<div class="row"><span class="lab">' + esc(d.label) + '</span><span class="bar" style="width:' + wpx + "px;background:" +
            (pos ? "var(--up)" : "var(--down)") + '"></span><span class="' + (pos ? "pos" : "neg") + '">' +
            ((d.contribution || 0) >= 0 ? "+" : "") + Number(d.contribution || 0).toFixed(4) + "</span></div>");
        });
        h.push("</div>");
      }
      // signals
      var sigs = pk.signals || [];
      if (sigs.length) {
        h.push('<div class="section-h">' + term("signals", "signals") + '</div><div class="signals"><table class="grid"><tbody>');
        sigs.forEach(function (s) {
          var read = s.read || "";
          var cls = read === "supports" ? "pos" : read === "detracts" ? "neg" : "muted";
          var val = (s.value != null ? s.value : "—") + (s.unit || "");
          h.push("<tr><td>" + esc(s.label) + '</td><td class="num">' + esc(val) + '</td><td class="num muted">' +
            (s.pct_rank != null ? s.pct_rank + "th" : "—") + '</td><td class="' + cls + '">' + esc(read) + "</td></tr>");
        });
        h.push("</tbody></table></div>");
      }
      // AI read — the model's plain-English take, placed AFTER the quantitative
      // breakdown (drivers + signals). The grounded template shows immediately; the
      // button upgrades to the local-model read; a dead LLM degrades back and says so.
      h.push(Ticker.aiReadBlock(t, r));
      // grounded chat — the narration made interactive; refuses over fabricating
      h.push(Ticker.askBlock(t));
      // news + events
      h.push(Ticker.enrichBlock("news memory (Intrinio)", t.news, function (a) {
        return '<span class="date">' + esc(a.date) + "</span>  " + (a.event_type && a.event_type !== "other" ? '<span class="ev">' + esc(a.event_type) + "</span> " : "") +
          esc((a.title || "").slice(0, 80)) + (a.source ? ' <span class="muted">· ' + esc(a.source) + "</span>" : "");
      }));
      h.push(Ticker.enrichBlock("recent filings & events (SEC EDGAR)", t.events, function (e) {
        return '<span class="date">' + esc(e.filed_date) + "</span>  <b>" + esc(e.form) + '</b> <span class="muted">' + esc(e.label) + "</span>";
      }));
      el(state.detailTarget).innerHTML = h.join("");
      Ticker.wire();
    },
    confidenceChip: function (r) {
      // How much to trust the BUY/HOLD/AVOID call — a 0-100 conviction DERIVED from the
      // frozen model's own out-of-sample hit-rate for this decile (capped, never implies
      // certainty). Absent when no calibration is built. Colored by INTENSITY (neutral),
      // not buy/sell, so a high-confidence AVOID never reads as a green "good". The
      // hit-rate it's built from always rides along so the number can't decouple from it.
      var c = r && r.confidence; if (!c || c.score == null) return "";
      var tone = c.score >= 55 ? "high" : c.score >= 25 ? "mid" : "low";
      var hr = c.hit_rate != null ? Math.round(c.hit_rate * 100) : null;
      var n = c.n ? " (n=" + Number(c.n).toLocaleString() + ")" : "";
      var note = hr != null ? '<span class="conf-note">beat the market ' + hr + "% of the time OOS" + n + "</span>" : "";
      return '<span class="conf term ' + tone + '" data-g="confidence">confidence ' + c.score + "/100</span>" + note;
    },
    aiReadBlock: function (t, r) {
      // narr = the local-model result once generated (via /narrate), else null.
      // r.narrative = the grounded template, ALWAYS present in the signal packet — so
      // there is always a read to show; the button upgrades it to the local model.
      var narr = t.narr;
      var busy = state.narrating === state.cik;   // in-flight flag lives on state → survives re-render
      var text = (narr && narr.narrative) ? narr.narrative : r.narrative;
      var chip;
      if (busy) {
        chip = '<span class="ai-status muted">generating local-model read …</span>';
      } else if (narr) {
        var offline = narr.source !== "llm";   // /narrate degraded → LLM endpoint was down
        chip = offline
          ? '<span class="ai-status warn" title="the local model endpoint was unreachable">AI model offline · showing grounded template</span>'
          : '<span class="ai-status ok">local model · ' + esc(narr.tier || "full") + "</span>";
      } else {
        chip = '<span class="ai-status muted">grounded template — generate the local-model read →</span>';
      }
      var h = '<div class="ai-read"><div class="ai-read-head"><span class="ai-tag">AI read</span>' + chip + "</div>";
      h += text ? '<div class="narr">' + esc(text) + "</div>" : '<div class="muted">no narration available</div>';
      h += '<div class="ai-read-tools"><button class="mini ai-go" id="btn-narr"' + (busy ? " disabled" : "") + ">" +
        (busy ? "narrating …" : narr ? "re-run local model" : "Generate local-model read") + "</button>" +
        '<span class="ai-note">a local LLM’s opinion, grounded to the filing (plus any news it cites) — not a price forecast</span></div>';
      return h + "</div>";
    },
    // -- ask: grounded chat over THIS name's computed data --------------------
    // Same honesty contract as the AI read, made conversational: the local model
    // answers ONLY from the packet + the display reads + recalled news; a made-up
    // number is caught server-side and it refuses instead. Transcript and draft
    // live on state.tk so async re-renders can't wipe them (posDraft precedent).
    askBlock: function (t) {
      var chat = t.chat || [];
      var busy = state.asking === state.cik;
      var h = '<div class="ai-read ask"><div class="ai-read-head"><span class="ai-tag">ask</span>' +
        '<span class="ai-status muted">grounded chat — it answers from this page\'s data or refuses; not advice</span></div>';
      if (!chat.length && !busy) {
        h += '<div class="ask-sugs">' + ["why is it ranked here?", "what changed since last year?", "what are the risk flags?", "what does the news say?"].map(function (q) {
          return '<span class="ask-sug" data-q="' + esc(q) + '">' + esc(q) + "</span>";
        }).join("") + "</div>";
      }
      chat.forEach(function (turn) {
        h += '<div class="ask-q">' + esc(turn.q) + "</div>";
        if (turn.offline) {
          h += '<div class="ask-a"><span class="warn">the local model is unreachable — start Ollama (or set STOCKSCAN_LLM_URL), then ask again.</span></div>';
        } else {
          h += '<div class="ask-a' + (turn.refused ? " refused" : "") + '">' + esc(turn.a) +
            (turn.refused ? ' <span class="warn">[refused — not in this name\'s data]</span>' : "") + "</div>";
        }
      });
      if (busy) {
        if (t.askPending) h += '<div class="ask-q">' + esc(t.askPending) + "</div>";
        h += '<div class="ask-a muted">thinking … (local model)</div>';
      }
      h += '<div class="ask-form"><input id="ask-q" placeholder="ask about the numbers on this page …" autocomplete="off" value="' + esc(t.askDraft || "") + '"' + (busy ? " disabled" : "") + ">" +
        '<button class="mini" id="btn-ask"' + (busy ? " disabled" : "") + ">ask</button></div>";
      if (t.askNote) h += '<div class="ask-flash warn">' + esc(t.askNote) + "</div>";
      return h + "</div>";
    },
    ask: function (q) {
      var cik = state.cik, t = state.tk;
      if (!cik || !t) return;
      q = String(q == null ? "" : q).trim();
      if (!q || state.asking === cik) return;
      t.askDraft = ""; t.askNote = null; t.askPending = q;
      state.asking = cik;
      Ticker.render();
      // history = the browser's own transcript (server is stateless); refused and
      // offline turns are dropped, like ask.py's REPL, and capped to the last 4 Q&As
      var history = [];
      (t.chat || []).forEach(function (turn) {
        if (!turn.refused && !turn.offline) history.push({ role: "user", content: turn.q }, { role: "assistant", content: turn.a });
      });
      var done = function (turn, note) {
        if (state.asking === cik) state.asking = null;
        if (state.cik !== cik || !state.tk) return;   // user moved on — don't clobber
        if (turn) state.tk.chat = (state.tk.chat || []).concat([turn]);
        state.tk.askPending = null;
        state.tk.askNote = note || null;
        Ticker.render();
      };
      apiPost("/ask/" + cik, { question: q, history: history.slice(-8) }).then(function (d) {
        if (d && d.busy) { done(null, "the local model is busy with another request — try again in a moment"); return; }
        if (!d || d.answer == null) { done({ q: q, offline: true }); return; }
        var offline = (d.violations || []).some(function (v) { return String(v).indexOf("llm-error") === 0; });
        done({ q: q, a: d.answer, refused: !!d.refused && !offline, offline: offline });
      }).catch(function () { done({ q: q, offline: true }); });
    },
    profileBlock: function (p) {
      if (p == null) return '<div class="muted" style="margin-top:8px">company profile · loading …</div>';
      if (!p || !p.description) return "";
      var country = p.country === "United States of America" || p.country === "United States" ? "USA" : p.country;
      var loc = [p.city, p.state, country].filter(Boolean).join(", ");
      var bits = [];
      if (loc) bits.push("HQ " + loc);
      if (p.industry) bits.push(p.industry);
      if (p.employees) bits.push(p.employees.toLocaleString() + " employees");
      if (p.url) bits.push(p.url);
      return '<div class="profile">' + esc(p.description) + (bits.length ? '<div class="meta">' + esc(bits.join("  ·  ")) + "</div>" : "") + "</div>";
    },
    priceBlock: function (t) {
      var pr = t.price;
      var h = "";
      if (pr && pr.summary && pr.points && pr.points.length > 1) {
        var s = pr.summary;
        var adv = s.adv ? " · " + term("adv", "ADV $" + (s.adv / 1e6).toFixed(1) + "M") : "";
        h += '<div class="price-line"><span class="last"><b>' + Number(s.last).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + "</b></span>" +
          '<span>1m <span class="' + pctColor(s.chg_1m) + '">' + sign(s.chg_1m) + "</span></span>" +
          '<span>3m <span class="' + pctColor(s.chg_3m) + '">' + sign(s.chg_3m) + "</span></span>" +
          '<span>1y <span class="' + pctColor(s.chg_1y) + '">' + sign(s.chg_1y) + "</span></span>" +
          '<span class="muted">close · 52wk ' + Math.round(s.lo_52w) + "–" + Math.round(s.hi_52w) + adv + "</span></div>";
      } else if (pr) {
        h += '<div class="muted">— no price history —</div>';
      }
      var q = t.quote;
      if (q && q.last != null) {
        var tm = (q.time || "").slice(11, 16);
        var auto = state.auto ? '<span class="pos">● auto</span> · ' : "";
        h += '<div class="price-line live">live <b>' + Number(q.last).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) +
          '</b> <span class="' + pctColor(q.chg_pct) + '">' + sign(q.chg_pct) + "</span> <span class=\"muted\">" + auto + "bid " + (q.bid || "—") + " · ask " + (q.ask || "—") + " · " + tm + "Z</span></div>";
      }
      h += '<div class="chart-wrap"><canvas class="chart" id="tk-chart"></canvas><div class="chart-tip" id="tk-tip" hidden></div></div>';
      h += '<div class="chart-tools"><span>' + state.chart + ' chart</span>' +
        '<button class="mini" id="btn-chart">switch</button>' +
        '<button class="mini" id="btn-live">&#8635; quote</button>' +
        '<button class="mini' + (state.auto ? " on" : "") + '" id="btn-auto">auto ' + (state.auto ? "●" : "○") + "</button></div>";
      return h;
    },
    // -- your position: personal holdings (live-view display only; never the signal) --
    positionBlock: function (t) {
      var pos = t.position || {}, draft = t.posDraft, saved = t.position != null;
      var sharesV = draft ? draft.shares : (pos.shares != null ? pos.shares : "");
      var costV = draft ? draft.cost : (pos.cost_basis != null ? pos.cost_basis : "");
      var h = '<div class="section-h">your position</div><div class="position">';
      h += '<div class="pos-form">' +
        '<label class="pos-in">shares<input type="number" step="any" min="0" inputmode="decimal" id="pos-shares" value="' + esc(sharesV) + '" placeholder="0"></label>' +
        '<label class="pos-in">avg cost<input type="number" step="any" min="0" inputmode="decimal" id="pos-cost" value="' + esc(costV) + '" placeholder="0.00"></label>' +
        '<button class="mini" id="btn-pos-save">' + (saved ? "update" : "save") + "</button>" +
        (saved ? '<button class="mini" id="btn-pos-remove">remove</button>' : "") +
        "</div>";
      h += '<div class="pos-pl" id="pos-pl"></div>';   // value + P/L, filled live by updatePL()
      h += Ticker.honestPanels(t);
      return h + "</div>";
    },
    computePL: function (t, shares, cost) {
      // current price: prefer the live quote, fall back to the last close — label which
      var price = null, label = null;
      if (t.quote && t.quote.last != null) { price = +t.quote.last; label = "live quote"; }
      else if (t.price && t.price.summary && t.price.summary.last != null) { price = +t.price.summary.last; label = "last close"; }
      if (price == null || !isFinite(price)) return { price: null };
      if (!(shares > 0)) return { price: price, label: label };
      var value = shares * price, hasCost = cost > 0, basis = shares * cost;
      return {
        price: price, label: label, value: value,
        pl: hasCost ? value - basis : null,
        plPct: hasCost ? (value - basis) / basis * 100 : null,
      };
    },
    updatePL: function () {
      var t = state.tk; if (!t) return;
      var pe = el("pos-pl"); if (!pe) return;
      var sh = el("pos-shares"), co = el("pos-cost");
      var shares = parseFloat(sh ? sh.value : ""), cost = parseFloat(co ? co.value : "");
      var c = Ticker.computePL(t, shares, cost);
      if (!c.price) { pe.innerHTML = '<span class="muted">waiting for a current price to value this position …</span>'; return; }
      if (!(shares > 0)) {
        pe.innerHTML = '<span class="muted">enter shares to see value &amp; P/L · current price ' + money(c.price) + ' <span class="pos-src">(' + c.label + ")</span></span>";
        return;
      }
      var out = '<span class="pos-val">value <b>' + money(c.value) + "</b></span>";
      if (c.pl != null) out += '<span class="pos-plv ' + (c.pl >= 0 ? "pos" : "neg") + '">unrealized ' + plMoney(c.pl) +
        (c.plPct != null ? " (" + (c.plPct >= 0 ? "+" : "") + c.plPct.toFixed(1) + "%)" : "") + "</span>";
      else out += '<span class="muted">enter avg cost for unrealized P/L</span>';
      out += '<span class="pos-src">' + shares.toLocaleString(undefined, { maximumFractionDigits: 4 }) + " sh × " + money(c.price) +
        ' <span class="muted">(' + c.label + ")</span></span>";
      pe.innerHTML = out;
    },
    onPosInput: function () {
      // capture the user's in-progress edit so an async re-render (auto-live quote,
      // late enrichment) can't wipe it; then repaint the readout.
      var t = state.tk; if (!t) return;
      var sh = el("pos-shares"), co = el("pos-cost");
      t.posDraft = { shares: sh ? sh.value : "", cost: co ? co.value : "" };
      Ticker.updatePL();
    },
    honestPanels: function (t) {
      // three HONEST reads for a holder — each backed by something real; no invented
      // multi-horizon price forecasts (the model is one monthly cross-sectional rank).
      var r = t.res, pk = r.packet, v = r.verdict || {}, model = pk.model || {};
      var pct = r.percentile != null ? r.percentile : model.percentile;
      var h = '<div class="honest">';
      // 1 · NOW — reuse the deterministic BUY / HOLD / AVOID verdict
      h += '<div class="hp"><div class="hp-h">now · this month\'s call</div><div class="hp-body">' +
        '<span class="badge ' + (v.color || "dim") + '">' + esc(v.call || "N/A") + "</span> " +
        '<span class="muted">' + (pct != null ? pct + "th pct" : "—") + "</span> " +
        Ticker.confidenceChip(r) +
        '<div class="hp-reason">' + esc(v.reason || "") + "</div></div></div>";
      // 2 · DOWNSIDE RISK — large-drawdown head (~6mo) + distress flag (~12mo) + a
      //     fundamentals-trajectory hint from the sum of SHAP drivers. Each omits gracefully.
      var dist = r.distress || pk.distress;
      var draw = r.drawdown || pk.drawdown;
      var drivers = pk.drivers || model.drivers || [];
      if ((draw && draw.flag) || (dist && dist.flag) || drivers.length) {
        h += '<div class="hp"><div class="hp-h">downside risk</div><div class="hp-body">';
        if (draw && draw.flag) {
          var wcls = draw.flag === "high" ? "neg" : draw.flag === "elevated" ? "warn" : "muted";
          var thr = draw.threshold != null ? Math.round(Math.abs(draw.threshold) * 100) : 30;
          h += "<div>" + term("drawdownRisk", "drawdown risk") + ' <span class="' + wcls + '">' + esc(draw.flag) + "</span>" +
            (draw.prob != null ? ' <span class="muted">P≈' + (draw.prob * 100).toFixed(0) + "%" +
              (draw.percentile != null ? " · " + draw.percentile + "th pct of peers" : "") + "</span>" : "") +
            '<div class="hp-reason">learned P(a ' + thr + "%+ peak-to-trough fall within ~" + (draw.horizon_months || 6) +
            "mo) — a display-only risk flag, never a trade input</div></div>";
        }
        if (dist && dist.flag) {
          var dcls = dist.flag === "high" ? "neg" : dist.flag === "elevated" ? "warn" : "muted";
          h += "<div>" + term("distress", "distress risk") + ' <span class="' + dcls + '">' + esc(dist.flag) + "</span>" +
            (dist.prob != null ? ' <span class="muted">P≈' + (dist.prob * 100).toFixed(1) + "%" +
              (dist.percentile != null ? " · " + dist.percentile + "th pct of peers" : "") + "</span>" : "") +
            '<div class="hp-reason">learned P(distress / delist within ~' + (dist.horizon_months || 12) +
            "mo) — display-only risk flag, never a trade input</div></div>";
        }
        if (drivers.length) {
          var net = drivers.reduce(function (a, d) { return a + (d.contribution || 0); }, 0);
          h += '<div>fundamentals trajectory <span class="' + (net >= 0 ? "pos" : "neg") + '">' +
            (net >= 0 ? "net-supportive" : "net-detracting") + "</span>" +
            '<div class="hp-reason">sum of the SHAP drivers ' + (net >= 0 ? "+" : "") + net.toFixed(4) +
            " — a fundamentals hint, not a price forecast</div></div>";
        }
        h += "</div></div>";
      }
      // 3 · LONG TERM — no number. The model's plain-English take is the SHARED AI read at
      //     the top; point to it rather than duplicating the narrative or offering a second
      //     button (both used to fire the SAME one /narrate call and show the text twice).
      h += '<div class="hp"><div class="hp-h">long term</div><div class="hp-body">' +
        '<div class="muted">No price target — the model makes one monthly cross-sectional call, not multi-year forecasts.</div>';
      if (t.narr && t.narr.narrative) {
        h += '<div class="muted hp-hint">The local model\'s plain-English take is in the <b>AI read</b> below ↓ — an opinion, not a forecast.</div>';
      } else {
        h += '<div class="muted hp-hint">Want the model\'s plain-English take? Use <b>Generate local-model read</b> in the AI read below ↓.</div>';
      }
      return h + "</div></div></div>";
    },
    savePosition: function () {
      if (!state.cik) return;
      var sh = el("pos-shares"), co = el("pos-cost");
      var shares = parseFloat(sh ? sh.value : ""), cost = parseFloat(co ? co.value : "");
      if (!(shares > 0)) { Ticker.updatePL(); return; }   // nothing meaningful to save yet
      var c = cost > 0 ? cost : 0;
      apiPost("/positions/" + state.cik, { shares: shares, cost: c }).then(function (p) {
        state.tk.position = p || { cik: state.cik, shares: shares, cost_basis: c };
        state.tk.posDraft = null;
        Ticker.render();
      });
    },
    removePosition: function () {
      if (!state.cik) return;
      apiDelete("/positions/" + state.cik).then(function () {
        state.tk.position = null; state.tk.posDraft = null; Ticker.render();
      });
    },
    enrichBlock: function (title, items, rowFn) {
      var badge = (items && items.length) ? ' <span class="count-badge">' + items.length + "</span>" : "";
      var h = '<div class="section-h">' + esc(title) + badge + '</div><div class="enrich">';
      if (items == null) h += '<div class="row muted">loading …</div>';
      else if (!items.length) h += '<div class="row muted">none</div>';
      else items.forEach(function (it) { h += '<div class="row">' + rowFn(it) + "</div>"; });
      return h + "</div>";
    },
    drawChart: function () {
      var t = state.tk; if (!t) return;
      var cv = el("tk-chart"); if (!cv) return;
      if (state.chart === "candle" && t.ohlc) Charts.candles(cv, t.ohlc);
      else if (t.price && t.price.points && t.price.points.length > 1) {
        var col = (t.price.summary && t.price.summary.chg_1y >= 0) ? Charts.colors.up : Charts.colors.down;
        Charts.line(cv, t.price.points, col);
      } else if (el("tk-chart")) { var c = cv.getContext("2d"); }
      Charts.hover(cv, el("tk-tip"));   // crosshair + data tooltip on hover
    },
    wire: function () {
      Ticker.drawChart();
      var bk = el("tk-back"); if (bk) bk.onclick = function () { switchView(state.cameFrom || "scan"); };
      var bc = el("btn-chart"); if (bc) bc.onclick = Ticker.toggleChart;
      var ws = el("wstar"); if (ws) ws.onclick = Ticker.toggleWatch;
      var bn = el("btn-narr"); if (bn) bn.onclick = Ticker.narrate;
      var bl = el("btn-live"); if (bl) bl.onclick = Ticker.live;
      var ba = el("btn-auto"); if (ba) ba.onclick = Ticker.autolive;
      var ps = el("btn-pos-save"); if (ps) ps.onclick = Ticker.savePosition;
      var prm = el("btn-pos-remove"); if (prm) prm.onclick = Ticker.removePosition;
      var psh = el("pos-shares"); if (psh) psh.oninput = Ticker.onPosInput;
      var pco = el("pos-cost"); if (pco) pco.oninput = Ticker.onPosInput;
      // ask box: draft + focus survive the async re-renders (quote/profile/news arrivals)
      var aq = el("ask-q");
      if (aq) {
        aq.oninput = function () { if (state.tk) state.tk.askDraft = aq.value; };
        aq.onfocus = function () { state.askFocus = true; };
        aq.onblur = function () { state.askFocus = false; };
        aq.onkeydown = function (e) { if (e.key === "Enter") { e.preventDefault(); Ticker.ask(aq.value); } };
        if (state.askFocus && document.activeElement !== aq) {
          aq.focus();
          var L = aq.value.length; try { aq.setSelectionRange(L, L); } catch (x) {}
        }
      }
      var ab = el("btn-ask"); if (ab) ab.onclick = function () { var i = el("ask-q"); Ticker.ask(i ? i.value : ""); };
      Array.prototype.forEach.call(document.querySelectorAll(".ask-sug"), function (s) {
        s.addEventListener("click", function () { Ticker.ask(s.dataset.q); });
      });
      Ticker.updatePL();
    },
    toggleChart: function () { state.chart = state.chart === "candle" ? "line" : "candle"; Ticker.render(); },
    toggleWatch: function () { if (!state.cik) return; apiPost("/watch/" + state.cik + "/toggle").then(function (d) { state.tk.watched = d.watched; Ticker.render(); }); },
    live: function () { if (state.cik) api("/live/quote/" + state.cik + "?refresh=true").then(function (d) { state.tk.quote = d; Ticker.render(); }); },
    autolive: function () {
      if (state.auto) { clearInterval(state.auto); state.auto = null; }
      else { state.auto = setInterval(function () { if (state.view === "ticker" && state.cik) Ticker.live(); }, 12000); Ticker.live(); }
      Ticker.render();
    },
    narrate: function () {
      if (!state.cik) return;
      var cik = state.cik;                       // guard: a slow narrate must not clobber a different ticker
      if (state.narrating === cik) return;       // already in flight — don't fire a duplicate POST
      state.narrating = cik;
      Ticker.render();                           // reflect the in-flight state on every narrate trigger (survives re-render)
      var done = function (d) {
        if (state.narrating === cik) state.narrating = null;
        if (state.cik === cik && state.tk) { state.tk.narr = d; Ticker.render(); }
      };
      // On any failure (server error / timeout / dropped request) fall back to a template
      // marker so the button re-enables and the block shows the grounded template — never
      // a permanently-stuck "narrating …".
      apiPost("/narrate/" + cik)
        .then(function (d) { done(d || { narrative: "", source: "template", tier: "?" }); })
        .catch(function () { done({ narrative: "", source: "error", tier: "?" }); });
    },
  };
  function nop() {}

  // -- MARKETS ---------------------------------------------------------------
  var Markets = {
    data: null,
    load: function () {
      state.marketsLoaded = true;
      api("/markets").then(function (d) { Markets.data = d; Markets.overview(); });
    },
    picker: function (kind, name) {
      var d = Markets.data, o = ['<option value="">overview · all markets</option>'];
      function grp(label, list, k) {
        o.push('<optgroup label="' + label + '">');
        list.forEach(function (g) {
          o.push('<option value="' + k + "::" + esc(g.market) + '"' + (kind === k && name === g.market ? " selected" : "") + ">" + esc(g.market) + "</option>");
        });
        o.push("</optgroup>");
      }
      grp("THEMES", d.themes, "theme");
      grp("INDUSTRIES", d.industries, "ind");
      return '<select class="market-pick" id="market-pick">' + o.join("") + "</select>";
    },
    overview: function () {
      var d = Markets.data;
      var h = [Markets.picker(), '<div class="map-head muted">top names by the model, sized by live market cap · pick a market for its treemap</div>'];
      h.push('<div class="mkt-kind">THEMES · auto-tagged from company descriptions</div>');
      d.themes.forEach(function (g) { h.push(Markets.group(g, "theme")); });
      h.push('<div class="mkt-kind">INDUSTRIES</div>');
      d.industries.forEach(function (g) { h.push(Markets.group(g, "ind")); });
      el("markets-body").innerHTML = h.join("");
      Markets.wire();
      Array.prototype.forEach.call(document.querySelectorAll(".pick"), function (p) { p.addEventListener("click", function () { Ticker.open(+p.dataset.cik); }); });
      Array.prototype.forEach.call(document.querySelectorAll(".mkt-group h3"), function (hh) { hh.addEventListener("click", function () { Markets.map(hh.dataset.kind, hh.dataset.name); }); });
      var ciks = [];
      d.themes.concat(d.industries).forEach(function (g) { g.picks.forEach(function (p) { ciks.push(p.cik); }); });
      apiPost("/market-caps", { ciks: ciks }).then(function (c) { Markets.fillCaps(c.caps); });
    },
    group: function (g, kind) {
      var h = '<div class="mkt-group"><h3 data-kind="' + kind + '" data-name="' + esc(g.market) + '" title="view treemap">' + esc(g.market.toUpperCase()) +
        '<span class="cnt">' + g.count + " names · top " + g.picks.length + " · ▦ treemap</span></h3>";
      g.picks.forEach(function (p) {
        h += '<div class="pick" data-cik="' + p.cik + '"><span class="tk">' + esc(p.ticker) +
          "</span><span>" + esc(p.name) + '</span><span class="num muted">' + p.pct + 'th</span><span class="num cap" data-cik="' + p.cik + '">…</span><span class="capbar" data-cik="' + p.cik + '" style="width:2px"></span></div>';
      });
      return h + "</div>";
    },
    map: function (kind, name) {
      el("markets-body").innerHTML = Markets.picker(kind, name) + '<div class="map-head muted">' + esc(name) + " · loading treemap …</div>";
      Markets.wire();
      api("/market?kind=" + encodeURIComponent(kind) + "&name=" + encodeURIComponent(name)).then(Markets.renderMap);
    },
    renderMap: function (d) {
      var h = [Markets.picker(d.kind, d.name),
        '<div class="map-head"><b>' + esc(d.name.toUpperCase()) + '</b> <span class="muted">· ' + d.tiles.length + " names · sized by market cap, colored by model signal · click a tile to open</span></div>"];
      if (!d.tiles.length) {
        h.push('<div class="empty">no market-cap data for this market</div>');
      } else {
        var t = '<div class="treemap" style="aspect-ratio:' + d.aspect + '">';
        d.tiles.forEach(function (x) {
          t += '<div class="tile" data-cik="' + x.cik + '" title="' + esc(x.name || "") + " · " + x.pct + 'th" style="left:' + (x.x * 100).toFixed(3) + "%;top:" + (x.y * 100).toFixed(3) +
            "%;width:" + (x.w * 100).toFixed(3) + "%;height:" + (x.h * 100).toFixed(3) + "%;background:" + heat(x.decile) + '">' +
            '<span class="t-tk">' + esc(x.ticker) + '</span><span class="t-cap">' + fmtCap(x.cap) + "</span></div>";
        });
        h.push(t + "</div>");
      }
      el("markets-body").innerHTML = h.join("");
      Markets.wire();
      Array.prototype.forEach.call(document.querySelectorAll(".tile"), function (tl) { tl.addEventListener("click", function () { Ticker.open(+tl.dataset.cik); }); });
    },
    wire: function () {
      var p = el("market-pick"); if (!p) return;
      p.onchange = function () {
        var v = p.value;
        if (!v) { Markets.overview(); return; }
        var i = v.indexOf("::"); Markets.map(v.slice(0, i), v.slice(i + 2));
      };
    },
    fillCaps: function (caps) {
      // per-group max for bar scaling
      Array.prototype.forEach.call(document.querySelectorAll(".mkt-group"), function (grp) {
        var picks = grp.querySelectorAll(".pick");
        var vals = [];
        Array.prototype.forEach.call(picks, function (p) { var c = caps[p.dataset.cik]; if (c) vals.push(c); });
        var mx = vals.length ? Math.max.apply(null, vals) : 1;
        Array.prototype.forEach.call(picks, function (p) {
          var c = caps[p.dataset.cik];
          var capEl = grp.querySelector('.cap[data-cik="' + p.dataset.cik + '"]');
          var barEl = grp.querySelector('.capbar[data-cik="' + p.dataset.cik + '"]');
          if (capEl) capEl.textContent = fmtCap(c);
          if (barEl && c) barEl.style.width = Math.max(2, Math.round(c / mx * 160)) + "px";
        });
      });
    },
  };

  // -- WATCH -----------------------------------------------------------------
  var Watch = {
    load: function () {
      api("/watch").then(function (w) {
        var h = ['<div class="section-h">watchlist</div>'];
        if (!w.rows.length) h.push('<div class="empty">watchlist empty — use <code>ops.py watch add</code></div>');
        else {
          h.push('<table class="grid" id="watch-table"><thead><tr><th>ticker</th><th class="num">' + term("pct", "model") + '</th><th class="num">Δ 30d</th><th>last 10-K</th><th>flag</th></tr></thead><tbody>');
          w.rows.forEach(function (r) {
            h.push('<tr data-cik="' + r.cik + '"><td class="tk">' + esc(r.ticker) + '</td><td class="num">' + (r.pct != null ? r.pct + "%" : "—") +
              '</td><td class="num ' + pctColor(r.delta) + '">' + (r.delta != null ? (r.delta >= 0 ? "+" : "") + r.delta : "—") + "</td><td class=\"muted\">" +
              esc(r.last_filing || "—") + '</td><td class="warn">' + esc(r.flag || "—") + "</td></tr>");
          });
          h.push("</tbody></table>");
        }
        h.push('<div class="section-h">alerts</div>');
        if (!w.alerts.length) h.push('<div class="muted">no alerts</div>');
        else { h.push('<table class="grid"><tbody>'); w.alerts.forEach(function (a) { h.push('<tr><td class="alert-star">' + (a.seen ? " " : "*") + '</td><td class="muted">' + esc((a.created || "").slice(0, 16)) + "</td><td>" + esc(a.kind) + "</td><td>" + esc(a.message) + "</td></tr>"); }); h.push("</tbody></table>"); }
        el("watch-body").innerHTML = h.join("");
        Array.prototype.forEach.call(document.querySelectorAll("#watch-table tbody tr"), function (tr) { tr.addEventListener("click", function () { Ticker.open(+tr.dataset.cik); }); });
      });
    },
  };

  // -- SCORECARD (your book) -------------------------------------------------
  // Book-level view of holdings (DISPLAY-ONLY, firewalled). A same-day peer-rank
  // snapshot — equal- AND value-weighted percentile, distress exposure, and
  // concentration — with the full holdings list ALWAYS shown so no single number
  // stands in for the distribution. Never a portfolio forecast.
  var Scorecard = {
    load: function () {
      el("scorecard-body").innerHTML = '<div class="muted">loading …</div>';
      api("/scorecard").then(Scorecard.render).catch(function (e) {
        el("scorecard-body").innerHTML = '<div class="err">could not load the book — ' + esc(e.message || e) + "</div>";
      });
    },
    stat: function (big, label) {
      return '<div class="sc-stat"><div class="sc-big">' + big + '</div><div class="sc-lab">' + esc(label) + "</div></div>";
    },
    pctCell: function (p) { return p == null ? '<span class="muted">—</span>' : Math.round(p) + '<span class="muted">th</span>'; },
    concentration: function (title, buckets) {
      if (!buckets || !buckets.length) return "";
      var byValue = buckets.some(function (b) { return b.weight_value != null; });
      var h = ['<div class="section-h">' + esc(title) + " · by " + (byValue ? "position value" : "holding count") + "</div><div class=\"sc-conc\">"];
      buckets.slice(0, 8).forEach(function (b) {
        var w = byValue ? (b.weight_value || 0) : (b.weight_count || 0);
        h.push('<div class="sc-crow"><span class="sc-cname">' + esc(b.name) + "</span>" +
          '<span class="sc-cbar"><span class="sc-cfill" style="width:' + (w * 100).toFixed(1) + '%"></span></span>' +
          '<span class="sc-cpct num">' + Math.round(w * 100) + '%</span><span class="sc-ccnt muted">' + b.count + " name" + (b.count === 1 ? "" : "s") + "</span></div>");
      });
      return h.join("") + "</div>";
    },
    render: function (sc) {
      if (!sc || !sc.n_total) {
        el("scorecard-body").innerHTML = '<div class="empty">Nothing tracked yet — star a stock to watch it (☆ on a ticker page), or add shares under <b>your position</b>. Watched names show here with their model standing; add shares to any to get value &amp; P/L.</div>';
        return;
      }
      var h = [];
      // lead: what this is + the honesty caveat (mirrors the ticker "honest panels")
      h.push('<p class="paper-lead">Your book — the names you track (holdings + watchlist), as a <b>same-day peer-rank snapshot</b>' +
        (sc.as_of ? " as of <b>" + esc(sc.as_of) + "</b>" : "") +
        ". The model makes one monthly cross-sectional call, <b>not a portfolio forecast</b>. Every name is listed below, so no single number stands in for the spread.</p>");

      // top-line stat cards
      h.push('<div class="sc-stats">');
      h.push(Scorecard.stat(sc.n_total + (sc.n_total === 1 ? " name" : " names"),
        sc.n_owned + " held · " + sc.n_watch + " watching"));
      h.push(Scorecard.stat(money(sc.total_value),
        sc.n_owned ? "book value · " + sc.n_owned + " held" : "book value · add shares to value"));
      var plc = sc.unrealized_pl == null ? "muted" : sc.unrealized_pl >= 0 ? "pos" : "neg";
      h.push(Scorecard.stat('<span class="' + plc + '">' + plMoney(sc.unrealized_pl) + "</span>" +
        (sc.unrealized_pl_pct != null ? ' <span class="sc-sub ' + plc + '">' + sign(sc.unrealized_pl_pct) + "</span>" : ""), "unrealized P/L"));
      h.push("</div>");

      // model standing — BOTH weightings, side by side
      h.push('<div class="section-h">model standing · where your names rank vs. peers</div>');
      h.push('<div class="sc-stats"><div class="sc-stat"><div class="sc-big">' + Scorecard.pctCell(sc.percentile_equal) +
        '</div><div class="sc-lab">' + term("equalweight", "equal-weight") + " · all tracked names</div></div>" +
        '<div class="sc-stat"><div class="sc-big">' + Scorecard.pctCell(sc.percentile_value) +
        '</div><div class="sc-lab">' + (sc.percentile_value == null ? term("valueweight", "value-weight") + " · add shares to enable" : term("valueweight", "value-weighted") + " · your holdings by size") + "</div></div></div>");
      h.push('<div class="hp-reason">Both are shown on purpose — equal-weight treats every name the same; value-weight leans on where your money actually sits. A peer-rank percentile, not a return estimate.</div>');

      // distress exposure (only when the risk head is loaded)
      if (sc.distress && sc.distress.known) {
        var d = sc.distress;
        h.push('<div class="section-h">distress exposure · display-only risk flag, never a trade input</div>');
        if (d.at_risk) {
          h.push('<div class="sc-flags">' +
            (d.count.high ? '<span class="badge red">' + d.count.high + " high</span>" : "") +
            (d.count.elevated ? '<span class="badge yellow">' + d.count.elevated + " elevated</span>" : "") +
            '<span class="badge dim">' + d.count.normal + " normal</span></div>");
          if (d.value && (d.value.high || d.value.elevated)) {
            h.push('<div class="hp-reason">' + money(d.value.high + d.value.elevated) +
              " of book value sits in flagged names — learned P(distress / delist within ~12mo).</div>");
          }
        } else {
          h.push('<div class="muted">No holdings flagged — all ' + d.count.normal + " rank normal on the distress head.</div>");
        }
      }

      // concentration — value-weighted bars (falls back to count when unpriced)
      h.push(Scorecard.concentration("industry concentration", sc.industry_concentration));

      // the full tracked-names table (never hidden) — held names first, then watched
      h.push('<div class="section-h">names · ' + sc.n_owned + " held, " + sc.n_watch + " watched</div>");
      h.push('<table class="grid" id="sc-table"><thead><tr>' +
        "<th>ticker</th><th>name</th><th>industry</th><th class=\"num\">shares</th>" +
        '<th class="num">value</th><th class="num">' + term("pct", "model") + '</th><th class="num">' + term("decile", "dec") + "</th><th>" + term("distress", "risk") + '</th><th class="num">P/L</th>' +
        "</tr></thead><tbody>");
      sc.holdings.slice().sort(function (a, b) { return (b.owned ? 1 : 0) - (a.owned ? 1 : 0); }).forEach(function (p) {
        var flag = p.dflag && p.dflag !== "normal"
          ? '<span class="' + (p.dflag === "high" ? "neg" : "warn") + '">' + esc(p.dflag) + "</span>"
          : (p.in_universe ? '<span class="muted">—</span>' : '<span class="warn">' + esc(p.status) + "</span>");
        var plc2 = p.unrealized_pl == null ? "muted" : p.unrealized_pl >= 0 ? "pos" : "neg";
        var tag = p.owned ? "" : ' <span class="sc-tag">watch</span>';
        h.push('<tr data-cik="' + p.cik + '">' +
          '<td class="tk">' + esc(p.ticker) + tag + "</td>" +
          "<td>" + esc(p.name || "—") + "</td>" +
          '<td class="muted">' + esc(p.in_universe ? p.industry : "—") + "</td>" +
          '<td class="num">' + (p.owned && p.shares != null ? p.shares.toLocaleString(undefined, { maximumFractionDigits: 4 }) : "—") + "</td>" +
          '<td class="num">' + (p.value != null ? money(p.value) : "—") + "</td>" +
          '<td class="num">' + (p.pct != null ? p.pct + "%" : "—") + "</td>" +
          '<td class="num">' + (p.decile != null ? p.decile : "—") + "</td>" +
          "<td>" + flag + "</td>" +
          '<td class="num ' + plc2 + '">' + (p.unrealized_pl != null ? plMoney(p.unrealized_pl) : "—") + "</td>" +
          "</tr>");
      });
      h.push("</tbody></table>");

      el("scorecard-body").innerHTML = h.join("");
      Array.prototype.forEach.call(document.querySelectorAll("#sc-table tbody tr"), function (tr) {
        tr.addEventListener("click", function () { Ticker.open(+tr.dataset.cik); });
      });
    },
  };

  // -- PAPER -----------------------------------------------------------------
  // Plain-English live scorecard for the FROZEN model. Reads /api/paper only
  // (firewalled from the signal): leads with what the page means, then tucks the
  // raw quant numbers into a muted "details" footnote for power users.
  var Paper = {
    // small signed-decimal formatter shared by the prose + the details line
    ic: function (x) { return (x != null && isFinite(x)) ? (x >= 0 ? "+" : "") + Number(x).toFixed(3) : "—"; },
    load: function () {
      api("/paper").then(function (rep) {
        if (!rep) { el("paper-body").innerHTML = '<div class="empty">no baseline frozen — run <code>ops.py paper freeze</code></div>'; return; }
        var b = rep.baseline || {}, ic = Paper.ic;
        var oos = rep.months_scored_oos || 0, deg = rep.degraded;

        var verdict = deg === false
          ? '<span class="pos">Tracking as expected ✓</span>'
          : deg === true
            ? '<span class="neg">⚠ Degraded — live accuracy has fallen below tolerance</span>'
            : '<span class="muted">Still gathering data.</span>';

        var h = "";
        // pretty date (frozen_on is a full ISO timestamp; the day is all we need)
        var frozen = b.frozen_on ? esc(String(b.frozen_on).slice(0, 10)) : "—";
        // OOS months' accuracy — drives both the trend chart and its empty state
        var ics = (rep.months || []).filter(function (m) { return m.h63 && !m.in_sample; }).map(function (m) { return m.h63.rank_ic; });

        // what this page IS
        h += '<p class="paper-lead">Live scorecard — the model was frozen on <b>' + frozen +
          "</b> and hasn't been retrained since. Each month we grade the picks it actually made against what prices really did.</p>";
        // translate "IC" into plain terms
        h += '<p class="paper-lead">Prediction accuracy is the rank correlation between predicted and actual returns — <b>0 is a coin-flip, higher is better</b>. The frozen backtest expected about <b>' +
          ic(b.expected_ic) + "</b>.</p>";
        // progress + live number
        h += '<p class="paper-lead"><b>' + oos + "</b> month" + (oos === 1 ? "" : "s") +
          " graded so far · live accuracy <b>" + ic(rep.live_mean_ic) + "</b>.</p>";
        // verdict (+ the "need ~6 months" caveat while still gathering)
        h += '<p class="paper-lead">' + verdict;
        if (deg == null) h += ' <span class="muted">— need ~6 clean months before we can judge on-track vs. degraded.</span>';
        h += "</p>";
        // trend — sparkline once months are graded, else a plain placeholder
        h += '<div class="section-h">' + term("ic", "accuracy by graded month") + "</div>";
        if (ics.length) {
          h += '<canvas class="spark" id="paper-spark"></canvas>';
          h += '<div class="muted" style="font-size:12px;margin-top:4px">Each point is one graded month — higher is better.</div>';
        } else {
          h += '<div class="muted" style="font-size:12px">No graded months yet — the chart appears once the first month is scored.</div>';
        }
        // raw numbers, tucked away for power users
        var det = [
          "expected IC " + ic(b.expected_ic),
          "expected spread/63d " + ic(b.expected_spread_63d),
          "live mean IC " + ic(rep.live_mean_ic),
          "live mean spread " + ic(rep.live_mean_spread),
          oos + " OOS month(s) scored",
          (rep.months_scored_in_sample || 0) + " in-sample (excluded)",
        ];
        if (rep.degradation_floor != null) det.push("degradation floor " + ic(rep.degradation_floor));
        h += '<div class="muted" style="margin-top:16px;font-size:12px">details · ' + esc(det.join(" · ")) + "</div>";

        el("paper-body").innerHTML = h;
        if (ics.length) Charts.spark(el("paper-spark"), ics, Charts.colors.ink);
      });
    },
  };

  // -- refresh (re-shows the mandala while the cross-section rebuilds) --------
  el("btn-refresh").addEventListener("click", function () {
    loader.style.display = "block"; loader.classList.remove("hide");
    started = Date.now();
    apiPost("/refresh").then(poll).catch(poll);
  });

  // "update data" — run the nightly job now (pull fresh prices/filings/news), then a full
  // reload from disk. The mandala loader covers the whole thing; we poll for completion.
  function nightlyPoll() {
    fetch("/api/nightly").then(function (r) { return r.json(); }).then(function (s) {
      if (s.running) {
        var e = s.elapsed || 0;
        loaderMsg("updating data — pulling fresh prices, filings & news … (" +
          Math.floor(e / 60) + "m " + Math.floor(e % 60) + "s)");
        setTimeout(nightlyPoll, 3000);
        return;
      }
      if (s.returncode && s.returncode !== 0) {
        loaderMsg("update failed (exit " + s.returncode + ") — data unchanged. see data/logs/nightly.web.log", "err");
        return;
      }
      loaderMsg("data updated — reloading …");
      started = Date.now();
      apiPost("/reload").then(poll).catch(poll);   // full reload picks up the new data
    }).catch(function () { setTimeout(nightlyPoll, 3000); });
  }
  el("btn-update").addEventListener("click", function () {
    loader.style.display = "block"; loader.classList.remove("hide");
    loaderMsg("starting data update …");
    apiPost("/nightly").then(function (s) {
      if (s && s.already) loaderMsg("an update is already running — waiting for it …");
      nightlyPoll();
    }).catch(function (e) { loaderMsg("could not start update — " + (e.message || e), "err"); });
  });

  window.addEventListener("resize", function () { if (state.view === "ticker") Ticker.drawChart(); });

  // -- help overlay ("what am I looking at?") --------------------------------
  (function () {
    var ov = el("help-overlay"), openBtn = el("btn-help"), closeBtn = el("help-close");
    function show(v) { if (ov) ov.hidden = !v; }
    if (openBtn) openBtn.addEventListener("click", function () { show(true); });
    if (closeBtn) closeBtn.addEventListener("click", function () { show(false); });
    if (ov) ov.addEventListener("click", function (e) { if (e.target === ov) show(false); });   // click backdrop
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { if (ov && !ov.hidden) show(false); return; }
      // "?" opens help, but never while the user is typing in the search box
      var tag = document.activeElement && document.activeElement.tagName;
      if (e.key === "?" && tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") show(true);
    });
  })();

  poll();   // start the handshake
})();
