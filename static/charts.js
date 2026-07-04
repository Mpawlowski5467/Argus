/* Canvas chart renderers for argus-web — line, candlestick+volume, sparkline.
   All draw from the raw arrays the API returns (no server-side rendering). */
(function () {
  "use strict";

  var C = {
    up: "#86b394", down: "#c68a86", amber: "#eaa23c",
    ink: "#e8e8e8", dim: "#8a8a8a", grid: "rgba(232,232,232,0.15)", bg: "#060606",
  };

  function setup(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || 700, h = canvas.clientHeight || 260;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Space Mono', monospace";
    return { ctx: ctx, w: w, h: h };
  }

  function fmt(n) {
    if (n == null || !isFinite(n)) return "—";
    if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  var PADL = 62, PADR = 12, PADT = 12, PADB = 22;

  function frame(ctx, w, h, hi, lo) {
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.strokeRect(PADL, PADT, w - PADL - PADR, h - PADT - PADB);
    // faint dashed mid gridline
    ctx.beginPath();
    ctx.moveTo(PADL, (PADT + (h - PADB)) / 2); ctx.lineTo(w - PADR, (PADT + (h - PADB)) / 2);
    ctx.stroke();
    ctx.setLineDash([]);   // solid again for the price line / candles drawn next
    ctx.fillStyle = C.dim; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(fmt(hi), PADL - 8, PADT + 6);
    ctx.fillText(fmt((hi + lo) / 2), PADL - 8, (PADT + (h - PADB)) / 2);
    ctx.fillText(fmt(lo), PADL - 8, h - PADB - 6);
  }

  function drawLine(canvas, points, color) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var vals = points.map(function (p) { return p.close; }).filter(isFinite);
    if (vals.length < 2) { ctx.fillStyle = C.dim; ctx.fillText("— no price history —", PADL, h / 2); return; }
    var hi = Math.max.apply(null, vals), lo = Math.min.apply(null, vals);
    var span = (hi - lo) || 1;
    frame(ctx, w, h, hi, lo);
    var plotW = w - PADL - PADR, plotH = h - PADT - PADB;
    ctx.beginPath();
    points.forEach(function (p, i) {
      var x = PADL + (i / (points.length - 1)) * plotW;
      var y = PADT + (1 - (p.close - lo) / span) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color || C.amber; ctx.lineWidth = 1.4; ctx.lineJoin = "round";
    ctx.stroke();
  }

  function drawCandles(canvas, o) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    if (!o || !o.close || o.close.length < 2) { ctx.fillStyle = C.dim; ctx.fillText("— no price history —", PADL, h / 2); return; }
    var n = o.close.length;
    var highs = o.high.filter(isFinite), lows = o.low.filter(isFinite);
    var hi = Math.max.apply(null, highs), lo = Math.min.apply(null, lows);
    var span = (hi - lo) || 1;
    var volH = 34, plotH = h - PADT - PADB - volH;
    frame(ctx, w, h - volH, hi, lo);
    var plotW = w - PADL - PADR;
    var cw = plotW / n;
    var bw = Math.max(1, Math.min(9, cw * 0.62));
    var maxVol = Math.max.apply(null, o.volume.map(function (v) { return v || 0; })) || 1;
    function yOf(v) { return PADT + (1 - (v - lo) / span) * plotH; }
    for (var i = 0; i < n; i++) {
      var op = o.open[i], cl = o.close[i], hg = o.high[i], lw = o.low[i];
      if (![op, cl, hg, lw].every(isFinite)) continue;
      var xc = PADL + (i + 0.5) * cw;
      var col = cl >= op ? C.up : C.down;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xc, yOf(hg)); ctx.lineTo(xc, yOf(lw)); ctx.stroke();  // wick
      var yo = yOf(op), yct = yOf(cl);
      var top = Math.min(yo, yct), bh = Math.max(1, Math.abs(yo - yct));
      ctx.fillRect(xc - bw / 2, top, bw, bh);  // body
      // volume
      var vh = ((o.volume[i] || 0) / maxVol) * (volH - 6);
      ctx.globalAlpha = 0.5;
      ctx.fillRect(xc - bw / 2, h - PADB - vh, bw, vh);
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = C.dim; ctx.textAlign = "left"; ctx.fillText("vol", 8, h - PADB - volH / 2);
  }

  function drawSpark(canvas, values, color) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var vals = values.filter(isFinite);
    if (vals.length < 2) { ctx.fillStyle = C.dim; ctx.fillText("no data", 4, h / 2); return; }
    var hi = Math.max.apply(null, vals), lo = Math.min.apply(null, vals), span = (hi - lo) || 1;
    var zero = h - 4 - (0 - lo) / span * (h - 8);
    if (lo < 0 && hi > 0) { ctx.strokeStyle = "#1c1c1c"; ctx.beginPath(); ctx.moveTo(0, zero); ctx.lineTo(w, zero); ctx.stroke(); }
    ctx.beginPath();
    vals.forEach(function (v, i) {
      var x = (i / (vals.length - 1)) * w;
      var y = h - 4 - (v - lo) / span * (h - 8);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color || C.amber; ctx.lineWidth = 1.4; ctx.stroke();
  }

  window.Charts = { line: drawLine, candles: drawCandles, spark: drawSpark, colors: C };
})();
