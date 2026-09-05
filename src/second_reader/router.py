"""Component 4: three-tier router on the acceptance score.

acceptance score in [0,1]:
    confidence          if verdict == accept
    1 - confidence      if verdict == reject
A confident reject lands near 0, a confident accept near 1, uncertainty near
0.5. An order-swap flip pulls the score toward 0.5 — into the review band.

score >= high -> auto_accept
score <= low  -> auto_reject (the FLAG decision)
else          -> human_review
"""

from __future__ import annotations

import duckdb

from .config import Thresholds


def acceptance_score(
    verdict: str, confidence: float, flipped: bool, flip_penalty: float
) -> float:
    score = confidence if verdict == "accept" else 1.0 - confidence
    if flipped:
        score = 0.5 + (score - 0.5) * (1.0 - flip_penalty)
    return score


def route_label(score: float, high: float, low: float) -> str:
    if score >= high:
        return "auto_accept"
    if score <= low:
        return "auto_reject"
    return "human_review"


def route_all(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
              rubric_version: str) -> int:
    """Join original+swapped grades, compute scores, write the routed table."""
    rows = con.execute(
        """
        SELECT o.item_id, o.verdict, o.confidence, o.reason, s.verdict
        FROM grades o
        JOIN grades s ON s.item_id = o.item_id
                     AND s.rubric_version = o.rubric_version
                     AND s.order_label = 'swapped'
        WHERE o.order_label = 'original' AND o.rubric_version = ?
        ORDER BY o.item_id
        """,
        [rubric_version],
    ).fetchall()

    con.execute("DELETE FROM routed")
    for item_id, verdict_o, conf_o, reason, verdict_s in rows:
        flipped = verdict_o != verdict_s
        score = acceptance_score(verdict_o, conf_o, flipped, thresholds.flip_penalty)
        con.execute(
            "INSERT INTO routed VALUES (?,?,?,?,?,?,?,?)",
            [
                item_id, rubric_version, verdict_o, verdict_s, flipped,
                score, route_label(score, thresholds.high, thresholds.low), reason,
            ],
        )
    return len(rows)
