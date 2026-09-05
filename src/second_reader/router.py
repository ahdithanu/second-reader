"""Component 3: three-tier router with two escalation paths.

acceptance score in [0,1]:
    confidence          if verdict == accept
    1 - confidence      if verdict == reject
A confident reject lands near 0, a confident accept near 1, uncertainty near
0.5. An order-swap flip pulls the score toward 0.5 by flip_penalty.

Routes:
    score >= high -> auto_accept                       (path AUTO)
    score <= low  -> auto_reject, the FLAG decision    (path AUTO)
    else          -> human_review                      (path THRESHOLD_ROUTED)
An agent flag_for_review in EITHER order overrides everything:
    human_review, path AGENT_ESCALATED (no score is computable — the
    escalating run emitted no verdict).
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
              rubric_version: str, mode: str) -> int:
    """Join original+swapped grades for a mode, compute routes, write `routed`."""
    rows = con.execute(
        """
        SELECT o.item_id, o.verdict, o.confidence, o.reason, o.escalated,
               s.verdict, s.escalated
        FROM grades o
        JOIN grades s ON s.item_id = o.item_id
                     AND s.rubric_version = o.rubric_version
                     AND s.mode = o.mode
                     AND s.order_label = 'swapped'
        WHERE o.order_label = 'original' AND o.rubric_version = ? AND o.mode = ?
        ORDER BY o.item_id
        """,
        [rubric_version, mode],
    ).fetchall()

    con.execute("DELETE FROM routed WHERE mode = ?", [mode])
    for item_id, v_o, conf_o, reason, esc_o, v_s, esc_s in rows:
        if esc_o or esc_s:
            con.execute(
                "INSERT INTO routed VALUES (?,?,?,?,?,?,?,?,?,?)",
                [item_id, mode, rubric_version, v_o, v_s, None, None,
                 "human_review", "AGENT_ESCALATED", reason],
            )
            continue
        flipped = v_o != v_s
        score = acceptance_score(v_o, conf_o, flipped, thresholds.flip_penalty)
        route = route_label(score, thresholds.high, thresholds.low)
        path = "THRESHOLD_ROUTED" if route == "human_review" else "AUTO"
        con.execute(
            "INSERT INTO routed VALUES (?,?,?,?,?,?,?,?,?,?)",
            [item_id, mode, rubric_version, v_o, v_s, flipped, score,
             route, path, reason],
        )
    return len(rows)
