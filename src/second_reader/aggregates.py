"""Layer 0: deterministic annotator-level aggregates.

The failure inventory's top two misses (POSITION_BIAS, and BOILERPLATE under
tight thresholds) are annotator-scoped: invisible per item, obvious across a
worker's history. This layer computes two signals per annotator from the
submissions table alone — no model calls, no ground-truth access:

  side_ratio  share of votes on the annotator's most-voted side
  dup_ratio   share of items carrying their most-reused justification
              (exact match after whitespace normalization)

An annotator over min_items with either signal above its flag threshold is
flagged, and all of their items route to auto-reject. Cost: $0.

Scope honesty: this catches *total* pattern defects by construction. An
annotator defective on 3 of 10 items sits below these thresholds — that is
failure mode #5 in the inventory, unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from .config import Thresholds


@dataclass(frozen=True)
class AnnotatorStats:
    annotator_id: str
    n_items: int
    side_ratio: float
    dup_ratio: float
    flagged: bool
    reason: str | None  # 'constant_side' | 'justification_reuse' | both joined


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def annotator_stats(con: duckdb.DuckDBPyConnection,
                    thresholds: Thresholds) -> list[AnnotatorStats]:
    rows = con.execute(
        "SELECT annotator_id, vote, justification FROM submissions ORDER BY annotator_id, item_id"
    ).fetchall()
    by_ann: dict[str, list[tuple[str, str]]] = {}
    for annotator_id, vote, justification in rows:
        by_ann.setdefault(annotator_id, []).append((vote, justification))

    out: list[AnnotatorStats] = []
    for annotator_id in sorted(by_ann):
        subs = by_ann[annotator_id]
        n = len(subs)
        votes = [v for v, _ in subs]
        side_ratio = max(votes.count("A"), votes.count("B")) / n
        counts: dict[str, int] = {}
        for _, j in subs:
            key = _norm(j)
            counts[key] = counts.get(key, 0) + 1
        dup_ratio = max(counts.values()) / n

        reasons = []
        if n >= thresholds.agg_min_items:
            if side_ratio >= thresholds.agg_side_ratio_flag:
                reasons.append("constant_side")
            if dup_ratio >= thresholds.agg_dup_ratio_flag:
                reasons.append("justification_reuse")
        out.append(AnnotatorStats(
            annotator_id=annotator_id, n_items=n,
            side_ratio=round(side_ratio, 4), dup_ratio=round(dup_ratio, 4),
            flagged=bool(reasons), reason="+".join(reasons) or None,
        ))
    return out


def flagged_annotators(con: duckdb.DuckDBPyConnection,
                       thresholds: Thresholds) -> dict[str, str]:
    """annotator_id -> reason for every aggregate-flagged annotator."""
    return {s.annotator_id: s.reason for s in annotator_stats(con, thresholds)
            if s.flagged}


def apply_to_records(records: list[dict], flagged: dict[str, str]) -> list[dict]:
    """Overlay the aggregate layer on per-item scored records.

    Items of a flagged annotator become auto-rejected (score 0, not
    escalated); everything else passes through unchanged. Records must carry
    an 'annotator_id' field.
    """
    out = []
    for r in records:
        if r["annotator_id"] in flagged:
            out.append({**r, "score": 0.0, "escalated": False,
                        "agg_flagged": True})
        else:
            out.append({**r, "agg_flagged": False})
    return out
