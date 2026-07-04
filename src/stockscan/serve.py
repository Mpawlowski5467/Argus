"""The per-ticker ONLINE path: parse -> compute -> score (frozen model) -> narrate.

For a company and an as-of date this builds the point-in-time cross-section (latest
10-K per company with ``available_date <= as_of``, liquidity-filtered, sector-ranked),
scores every name with the FROZEN artifact (no retraining, ever), and narrates the
target from a grounded signal packet.

Train/serve parity is structural, not aspirational: the cross-section here is built
from the SAME functions (`prepare_features` / `pit_snapshot` / `liquidity_mask` /
`add_sector_ranks`) the training panel uses, so the feature vector served for
(cik, d) is bit-identical to the panel row the model trained on — enforced by
tests/test_serve.py. A delisted company flows through unchanged: its column simply
stops having prices after death, and its filings go stale ~18 months later.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from .concepts import WIDE_PATH
from .config import (
    LABEL_HORIZON_DAYS,
    MAX_STALE_DAYS,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    MIN_SECTOR_BUCKET,
)
from .confidence import load_calibration_optional, score_confidence
from .distress import (
    DistressArtifact,
    distress_flag,
    load_distress_artifact_optional,
)
from .drawdown import (
    DrawdownArtifact,
    drawdown_flag,
    load_drawdown_artifact_optional,
)
from .edgar.tickers import cik_for
from .features import compute_features
from .fundamental_panel import add_sector_ranks, liquidity_mask, pit_snapshot, prepare_features
from .intrinio_universe import universe_ticker_map
from .model import Artifact, load_artifact
from .narrate.ground import check_grounding
from .narrate.packet import LABELS, build_packet
from .narrate.narrator import narrate_packet
from .panel import load_matrices_cached


@dataclass
class ServeData:
    """The heavyweight inputs, loaded once and reused across analyze() calls."""

    feats: pd.DataFrame            # prepare_features() output for every filing
    close: pd.DataFrame            # wide adjusted closes (columns = universe price columns)
    dv_med: pd.DataFrame           # 20d median dollar volume, same shape
    ticker_map: dict[int, str]     # cik -> price column (dead names are TICKER~CIK)
    # OPTIONAL, FIREWALLED risk-flag head — display/alert ONLY, never a trade input; None
    # when no distress artifact is frozen (serve then behaves exactly as before).
    distress_artifact: DistressArtifact | None = None
    # OPTIONAL, FIREWALLED confidence calibration (per-decile OOS hit-rate). Display-only;
    # None when unbuilt (serve then behaves exactly as before).
    confidence_calibration: dict | None = None
    # OPTIONAL, FIREWALLED large-drawdown risk head — display/alert ONLY; None when unfrozen.
    drawdown_artifact: DrawdownArtifact | None = None


def load_serve_data() -> ServeData:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = prepare_features(compute_features(wide))
    # cached wide matrices when in sync with the per-column store (seconds instead
    # of ~2 minutes — what makes a daily monitor loop viable), slow path otherwise
    close, dv = load_matrices_cached()
    if close.empty:
        raise FileNotFoundError("no prices on disk; run scripts/fetch_intrinio_prices.py")
    tmap = universe_ticker_map()
    if not tmap:
        raise FileNotFoundError("no universe map; run scripts/build_intrinio_universe.py")
    return ServeData(
        feats=feats,
        close=close,
        dv_med=dv.rolling(20, min_periods=10).median(),
        ticker_map=tmap,
        distress_artifact=load_distress_artifact_optional(),
        confidence_calibration=load_calibration_optional(),
        drawdown_artifact=load_drawdown_artifact_optional(),
    )


def resolve_company(query, ticker_map: dict[int, str]) -> tuple[int, str | None]:
    """Resolve a ticker / TICKER~CIK column / CIK to (cik, price_column).

    A plain ticker resolves against the universe map first (survivorship-safe:
    active columns are plain tickers there), then falls back to EDGAR's current
    ticker list. Dead names are addressed by their column (``BBBY~886158``) or CIK.
    """
    if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
        cik = int(query)
        return cik, ticker_map.get(cik)
    q = str(query).upper()
    if "~" in q:
        return int(q.split("~")[1]), q
    inverse = {col: cik for cik, col in ticker_map.items()}
    cik = inverse.get(q)
    if cik is None:
        cik = cik_for(q)
    if cik is None:
        raise ValueError(f"cannot resolve company: {query!r}")
    return cik, ticker_map.get(cik)


def build_cross_section(
    data: ServeData,
    as_of,
    max_stale_days: int = MAX_STALE_DAYS,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    min_price: float = MIN_PRICE,
    min_sector_bucket: int = MIN_SECTOR_BUCKET,
    include_cik: int | None = None,
) -> pd.DataFrame:
    """The as-of cross-section: PIT snapshot -> liquidity filter -> sector ranks.

    Mirrors one date-iteration of the training panel build exactly (shared code).
    ``include_cik`` keeps the target company even if it fails the liquidity floor
    (flagged in ``liquidity_pass``) so an illiquid name can still be analyzed —
    everyone else must clear the tradable-universe floors, as in training.
    """
    as_of = pd.Timestamp(as_of)
    latest = pit_snapshot(data.feats, as_of, max_stale_days)
    if latest.empty:
        raise ValueError(f"no fundamentals available point-in-time at {as_of.date()}")
    latest["ticker"] = latest["cik"].map(data.ticker_map)

    price_date = data.close.index.asof(as_of)  # last trading day <= as_of
    if pd.isna(price_date):
        raise ValueError(f"as-of {as_of.date()} predates the price history")
    liquid = liquidity_mask(latest, price_date, data.close, data.dv_med,
                            min_dollar_volume, min_price)
    keep = liquid | (latest["cik"] == include_cik) if include_cik is not None else liquid
    cross = latest[keep].copy()
    cross["liquidity_pass"] = liquid[keep]
    cross.attrs["price_date"] = price_date
    return add_sector_ranks(cross, min_sector_bucket)


def analyze(
    company,
    as_of=None,
    data: ServeData | None = None,
    artifact: Artifact | None = None,
    llm=None,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    max_stale_days: int = MAX_STALE_DAYS,
    news=None,
    distress_artifact: DistressArtifact | None = None,
    calibration: dict | None = None,
    drawdown_artifact: DrawdownArtifact | None = None,
) -> dict:
    """End-to-end per-ticker analysis at ``as_of`` (default: latest price date).

    Everything is keyed off ``available_date <= as_of``; the frozen artifact only
    scores. Returns packet, model signal, grounded narrative, and honesty flags.

    ``news`` (optional): recalled article takeaways for narration context ONLY
    (LIVE-VIEW). It rides into the packet AFTER scoring — the score/percentile/drivers
    above are already fixed by the time news is attached, so it cannot touch the signal.

    ``distress_artifact`` (optional): the FIREWALLED distress-risk head. When present
    (passed, or loaded into ``data``) it adds a ``distress`` block — P(distress-delist
    within the horizon), its cross-sectional percentile, and a flag — computed AFTER the
    return score is fixed, from the SAME ranks. It is a risk-flag for the human ONLY: it
    never enters the score/percentile/decile/drivers, the packet, or any trade rule.

    ``calibration`` (optional): the FIREWALLED confidence calibration table. When present
    (passed, or loaded into ``data``) it adds a ``confidence`` block — a 0-100 conviction
    for the call, derived from the model's per-decile OOS hit-rate — computed AFTER the
    score is fixed. Display-only: never a feature, never a trade input.

    ``drawdown_artifact`` (optional): the FIREWALLED large-drawdown risk head. Same contract
    as distress — adds a ``drawdown`` block (P(deep peak-to-trough fall within the horizon),
    peer percentile, flag) computed after the score is fixed; display-only, never a trade input.
    """
    data = data or load_serve_data()
    artifact = artifact or load_artifact()
    distress_artifact = distress_artifact if distress_artifact is not None \
        else getattr(data, "distress_artifact", None)
    calibration = calibration if calibration is not None \
        else getattr(data, "confidence_calibration", None)
    drawdown_artifact = drawdown_artifact if drawdown_artifact is not None \
        else getattr(data, "drawdown_artifact", None)
    cik, column = resolve_company(company, data.ticker_map)
    as_of = pd.Timestamp(as_of) if as_of is not None else data.close.index[-1]

    cross = build_cross_section(
        data, as_of, max_stale_days=max_stale_days,
        min_dollar_volume=min_dollar_volume, include_cik=cik,
    )
    hit = cross[cross["cik"] == cik]
    if hit.empty:
        visible = data.feats[(data.feats["cik"] == cik) & (data.feats["available_date"] <= as_of)]
        if visible.empty:
            raise ValueError(f"cik {cik}: no 10-K available point-in-time at {as_of.date()}")
        last = visible["available_date"].max().date()
        raise ValueError(
            f"cik {cik}: latest 10-K (available {last}) is staler than {max_stale_days}d "
            f"at {as_of.date()} — likely long-dead or a lapsed filer"
        )
    row = hit.iloc[0]

    # Frozen model scores the whole cross-section; the signal is the target's
    # cross-sectional rank of that score (never a raw "predicted return").
    scores = artifact.score(cross)
    score_pct = pd.Series(scores, index=cross.index).rank(pct=True)
    target_pct = float(score_pct.loc[hit.index[0]])
    decile = int(np.clip(np.ceil(target_pct * 10), 1, 10))

    feats_pit = data.feats[data.feats["available_date"] <= as_of]
    packet = build_packet(cik, features_df=feats_pit, snapshot=cross, as_of=as_of, news=news)
    packet["meta"]["ticker"] = column
    # SHAP drivers: an exact decomposition of the target's score into per-feature
    # contributions — the ML -> narration bridge (DESIGN.md §7). Sign convention:
    # positive contribution pushes the model signal UP ("supports").
    contrib = artifact.explain(hit).iloc[0]
    ranked_drivers = sorted(
        ((c, float(contrib[c])) for c in artifact.feature_cols),
        key=lambda x: -abs(x[1]),
    )
    # driver ids are namespaced ("driver:roa") because the MODEL's learned direction
    # can legitimately disagree with the textbook signal direction (that is the
    # learned-signs edge) — the citation validator must never conflate the two
    drivers = [
        {
            "id": f"driver:{c.removesuffix('_rank')}",
            "label": LABELS.get(c.removesuffix("_rank"), c),
            "contribution": round(v, 4),
            "direction": "supports" if v > 0 else "detracts",
        }
        for c, v in ranked_drivers[:5]
        if abs(v) > 1e-6
    ]

    packet["model"] = {
        "label": "Frozen-model cross-sectional signal (relative rank, not a return forecast)",
        "score": round(float(scores[cross.index.get_loc(hit.index[0])]), 4),
        "percentile": int(round(target_pct * 100)),
        "decile": decile,
        "n_names": int(len(cross)),
        "as_of": str(as_of.date()),
        "trained_through": artifact.meta["trained_through"],
        "drivers": drivers,
    }

    # FIREWALLED distress read: score the SAME cross-section with the distress head and
    # attach the target's probability + peer percentile + flag. This happens AFTER the
    # model block above is fully fixed and is NOT written into the packet — it is a
    # display/alert risk-flag only, never a feature, never a trade input (verdict:
    # RESULTS.md distress-overlay gate). Any scoring hiccup degrades to None, never breaks
    # the return path.
    distress = None
    if distress_artifact is not None:
        try:
            dprob = pd.Series(distress_artifact.score(cross), index=cross.index)
            target_dp = float(dprob.loc[hit.index[0]])
            distress = {
                "prob": round(target_dp, 4),
                "percentile": int(round(float(dprob.rank(pct=True).loc[hit.index[0]]) * 100)),
                "flag": distress_flag(target_dp),
                "horizon_months": int(distress_artifact.meta.get("horizon_months", 12)),
                "trained_through": distress_artifact.meta.get("trained_through"),
                "n_names": int(len(cross)),
            }
        except (KeyError, ValueError):
            distress = None

    # FIREWALLED large-drawdown read: identical contract to distress — score the SAME
    # cross-section, attach P(deep peak-to-trough fall within the horizon) + peer percentile
    # + flag, AFTER the model block is fixed. Display/alert risk-flag only; degrades to None.
    drawdown = None
    if drawdown_artifact is not None:
        try:
            wprob = pd.Series(drawdown_artifact.score(cross), index=cross.index)
            target_wp = float(wprob.loc[hit.index[0]])
            drawdown = {
                "prob": round(target_wp, 4),
                "percentile": int(round(float(wprob.rank(pct=True).loc[hit.index[0]]) * 100)),
                "flag": drawdown_flag(target_wp),
                "horizon_months": int(drawdown_artifact.meta.get("horizon_months", 6)),
                "threshold": float(drawdown_artifact.meta.get("threshold", -0.30)),
                "trained_through": drawdown_artifact.meta.get("trained_through"),
                "n_names": int(len(cross)),
            }
        except (KeyError, ValueError):
            drawdown = None

    result = narrate_packet(packet, llm=llm)
    violations = check_grounding(result["narrative"], packet)  # invariant 3, re-checked here

    # The training information window extends PAST trained_through by the label
    # horizon: labels sampled on the last training date are realized over the next
    # `horizon` trading days. An as_of inside that window is still in-sample.
    idx = data.close.index
    horizon = int(artifact.meta.get("label_horizon_days", LABEL_HORIZON_DAYS))
    loc = idx.searchsorted(artifact.trained_through, side="right") - 1
    info_through = idx[min(loc + horizon, len(idx) - 1)] if loc >= 0 else artifact.trained_through

    flags = {
        "liquidity_pass": bool(row["liquidity_pass"]),
        "filed_date": str(pd.Timestamp(row["filed_date"]).date()),
        "available_date": str(pd.Timestamp(row["available_date"]).date()),
        "staleness_days": int((as_of - row["available_date"]).days),
        "price_date": str(pd.Timestamp(cross.attrs["price_date"]).date()),
        "in_sample": bool(as_of <= info_through),
    }

    # FIREWALLED confidence read: derive a 0-100 conviction for this call from the frozen
    # model's OOS track record (calibration) plus the decile/percentile/drivers/flags
    # already fixed above. Display-only; degrades to None. Never a feature/score/trade input.
    confidence = None
    if calibration is not None:
        try:
            confidence = score_confidence(
                decile, packet["model"]["percentile"], drivers, flags, calibration
            )
        except Exception:
            confidence = None

    return {
        "as_of": as_of,
        "cik": cik,
        "column": column,
        "packet": packet,
        "ranks": {c: float(row[c]) for c in artifact.feature_cols},
        "score": packet["model"]["score"],
        "percentile": packet["model"]["percentile"],
        "decile": decile,
        "distress": distress,   # FIREWALLED risk-flag (or None); never touches the signal
        "drawdown": drawdown,   # FIREWALLED downside-risk flag (or None); never touches the signal
        "confidence": confidence,  # FIREWALLED conviction read (or None); never touches the signal
        "narrative": result["narrative"],
        "reasoning": result.get("reasoning", ""),
        "citations": result.get("citations", []),
        "attempts": result.get("attempts", 0),
        "first_pass_ok": result.get("first_pass_ok", True),
        "narration_violations": result.get("violations", []),
        "violation_log": result.get("violation_log", []),
        "source": result["source"],
        "grounded": not violations,
        "grounding_violations": violations,
        "flags": flags,
    }
