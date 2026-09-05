"""Component 5: calibration sweep, ablation comparison, tool-use metrics.

The flag decision is `score <= low` (auto-reject) among non-escalated items.
Escalated items land in human review by definition, so they are never
flagged; `catch` metrics count flagged-OR-escalated. Both are reported —
never blended.

`flag_metrics` is a pure function over plain records so the math is unit-
testable against a fixture with known labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import polars as pl

from .config import RESULTS_DIR, Thresholds
from .defects import DEFECT_TYPES
from .router import acceptance_score

MODES = ["single", "agent"]


def flag_metrics(records: list[dict], high: float, low: float) -> dict:
    """records: [{score: float|None, escalated: bool, is_defective, defect_type}].

    Pure precision/recall math. Escalated records have score=None and are in
    the review queue; they are excluded from the flag decision but counted in
    review_load and in the catch metrics.
    """
    n = len(records)
    decided = [r for r in records if not r["escalated"]]
    flagged = [r for r in decided if r["score"] <= low]
    review = [r for r in records
              if r["escalated"] or (low < r["score"] < high)]
    true_defects = [r for r in records if r["is_defective"]]

    tp = sum(1 for r in flagged if r["is_defective"])
    caught = [r for r in true_defects
              if r["escalated"] or r["score"] <= low]

    def per_type(pred) -> dict:
        out: dict[str, float | None] = {}
        for dt in DEFECT_TYPES:
            of_type = [r for r in true_defects if r["defect_type"] == dt]
            hits = sum(1 for r in of_type if pred(r))
            out[dt] = hits / len(of_type) if of_type else None
        return out

    return {
        "high": high,
        "low": low,
        "n_items": n,
        "n_flagged": len(flagged),
        "n_escalated": sum(1 for r in records if r["escalated"]),
        "precision": tp / len(flagged) if flagged else None,
        "recall": tp / len(true_defects) if true_defects else None,
        "catch_rate": len(caught) / len(true_defects) if true_defects else None,
        "review_load": len(review) / n if n else None,
        "recall_by_type": per_type(lambda r: not r["escalated"] and r["score"] <= low),
        "catch_by_type": per_type(lambda r: r["escalated"] or r["score"] <= low),
    }


def scored_records(con: duckdb.DuckDBPyConnection, rubric_version: str,
                   flip_penalty: float, mode: str) -> list[dict]:
    """Per-item records with acceptance score, escalation, and defect truth."""
    rows = con.execute(
        """
        SELECT o.item_id, o.verdict, o.confidence,
               (o.verdict IS NOT NULL AND s.verdict IS NOT NULL AND o.verdict != s.verdict),
               (o.escalated OR s.escalated),
               sub.is_defective, sub.defect_type
        FROM grades o
        JOIN grades s ON s.item_id = o.item_id AND s.rubric_version = o.rubric_version
                     AND s.mode = o.mode AND s.order_label = 'swapped'
        JOIN submissions sub ON sub.item_id = o.item_id
        WHERE o.order_label = 'original' AND o.rubric_version = ? AND o.mode = ?
        ORDER BY o.item_id
        """,
        [rubric_version, mode],
    ).fetchall()
    return [
        {
            "item_id": item_id,
            "score": None if escalated else acceptance_score(verdict, conf, bool(flipped), flip_penalty),
            "flipped": bool(flipped),
            "escalated": bool(escalated),
            "is_defective": bool(is_def),
            "defect_type": defect_type,
        }
        for item_id, verdict, conf, flipped, escalated, is_def, defect_type in rows
    ]


def grading_cost(con: duckdb.DuckDBPyConnection, mode: str) -> float:
    total = con.execute(
        "SELECT sum(cost_usd) FROM call_log WHERE purpose='grade' AND mode=?", [mode]
    ).fetchone()[0]
    return total or 0.0


def sweep(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
          rubric_version: str) -> pl.DataFrame:
    rows = []
    for mode in MODES:
        records = scored_records(con, rubric_version, thresholds.flip_penalty, mode)
        if not records:
            continue
        cost_1k = grading_cost(con, mode) / len(records) * 1000
        for high in thresholds.sweep_high:
            for low in thresholds.sweep_low:
                if low >= high:
                    continue
                m = flag_metrics(records, high, low)
                row = {
                    "mode": mode, "high": m["high"], "low": m["low"],
                    "n_items": m["n_items"], "n_flagged": m["n_flagged"],
                    "n_escalated": m["n_escalated"],
                    "precision": m["precision"], "recall": m["recall"],
                    "catch_rate": m["catch_rate"],
                    "review_load": m["review_load"],
                    "cost_per_1000_usd": cost_1k,
                }
                for dt in DEFECT_TYPES:
                    row[f"recall_{dt}"] = m["recall_by_type"][dt]
                    row[f"catch_{dt}"] = m["catch_by_type"][dt]
                rows.append(row)
    return pl.DataFrame(rows)


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def ablation_table(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
                   rubric_version: str) -> str:
    """The headline: recall by defect type, tools on vs off, at the operating point."""
    metrics = {}
    for mode in MODES:
        records = scored_records(con, rubric_version, thresholds.flip_penalty, mode)
        if records:
            metrics[mode] = flag_metrics(records, thresholds.high, thresholds.low)
    if set(metrics) != set(MODES):
        return "(ablation incomplete: missing a mode)"

    lines = [
        "| Defect type | Recall, tools OFF | Recall, tools ON | Catch incl. agent escalation (ON) |",
        "|---|---|---|---|",
    ]
    for dt in DEFECT_TYPES:
        lines.append(
            f"| {dt} | {_fmt(metrics['single']['recall_by_type'][dt], pct=True)} "
            f"| {_fmt(metrics['agent']['recall_by_type'][dt], pct=True)} "
            f"| {_fmt(metrics['agent']['catch_by_type'][dt], pct=True)} |"
        )
    lines.append(
        f"| **Overall** | **{_fmt(metrics['single']['recall'], pct=True)}** "
        f"| **{_fmt(metrics['agent']['recall'], pct=True)}** "
        f"| **{_fmt(metrics['agent']['catch_rate'], pct=True)}** |"
    )
    lines.append("")
    lines.append(
        f"Flag precision: tools OFF {_fmt(metrics['single']['precision'], pct=True)}, "
        f"tools ON {_fmt(metrics['agent']['precision'], pct=True)}. "
        f"Review load: OFF {_fmt(metrics['single']['review_load'], pct=True)}, "
        f"ON {_fmt(metrics['agent']['review_load'], pct=True)} "
        f"(of which {metrics['agent']['n_escalated']} agent-escalated)."
    )
    return "\n".join(lines)


def tool_metrics(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
                 rubric_version: str) -> dict:
    """Mean tool calls, distribution by defect type, cap rate, marginal cost."""
    rows = con.execute(
        """
        SELECT t.item_id, t.n_tool_calls, t.hit_cap, t.trace, sub.defect_type
        FROM traces t JOIN submissions sub USING (item_id)
        WHERE t.mode = 'agent' AND t.rubric_version = ?
        """,
        [rubric_version],
    ).fetchall()
    if not rows:
        return {}

    n = len(rows)
    tool_counts: Counter = Counter()
    by_type: dict[str, list[int]] = {}
    for _, n_calls, _, trace_json, defect_type in rows:
        key = defect_type or "CLEAN"
        by_type.setdefault(key, []).append(n_calls)
        for ev in json.loads(trace_json):
            if ev.get("type") == "tool_call":
                tool_counts[ev["name"]] += 1

    # Marginal cost per point of recall gained by enabling tools.
    result: dict = {
        "mean_tool_calls_per_run": sum(r[1] for r in rows) / n,
        "pct_runs_hitting_cap": sum(1 for r in rows if r[2]) / n,
        "tool_call_counts": dict(tool_counts),
        "mean_tool_calls_by_defect_type": {
            k: sum(v) / len(v) for k, v in sorted(by_type.items())
        },
    }
    per_mode = {}
    for mode in MODES:
        records = scored_records(con, rubric_version, thresholds.flip_penalty, mode)
        if records:
            m = flag_metrics(records, thresholds.high, thresholds.low)
            per_mode[mode] = {
                "recall": m["recall"],
                "cost_per_1000": grading_cost(con, mode) / len(records) * 1000,
            }
    if set(per_mode) == set(MODES):
        d_recall_pts = (per_mode["agent"]["recall"] - per_mode["single"]["recall"]) * 100
        d_cost = per_mode["agent"]["cost_per_1000"] - per_mode["single"]["cost_per_1000"]
        result["cost_per_1000"] = {m: per_mode[m]["cost_per_1000"] for m in MODES}
        result["recall_at_operating_point"] = {m: per_mode[m]["recall"] for m in MODES}
        result["marginal_cost_per_recall_point_per_1000"] = (
            d_cost / d_recall_pts if d_recall_pts > 0 else None
        )
    return result


def escalation_report(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
                      rubric_version: str) -> dict:
    """Compare the two escalation paths for the agent mode.

    AGENT_ESCALATED items have no computable threshold score (the escalating
    run emitted no verdict), so the honest comparison is set sizes plus, for
    each escalated item, what the OTHER order's run concluded.
    """
    records = scored_records(con, rubric_version, thresholds.flip_penalty, "agent")
    escalated = [r for r in records if r["escalated"]]
    threshold_routed = [
        r for r in records
        if not r["escalated"] and thresholds.low < r["score"] < thresholds.high
    ]
    details = []
    for r in escalated:
        other = con.execute(
            """
            SELECT order_label, verdict, confidence, escalated FROM grades
            WHERE item_id=? AND mode='agent' AND rubric_version=?
            ORDER BY order_label
            """,
            [r["item_id"], rubric_version],
        ).fetchall()
        details.append({
            "item_id": r["item_id"],
            "defect_type": r["defect_type"],
            "orders": [
                {"order": o, "verdict": v, "confidence": c, "escalated": bool(e)}
                for o, v, c, e in other
            ],
        })
    return {
        "n_agent_escalated": len(escalated),
        "n_threshold_routed": len(threshold_routed),
        "escalated_truly_defective": sum(1 for r in escalated if r["is_defective"]),
        "threshold_routed_truly_defective": sum(1 for r in threshold_routed if r["is_defective"]),
        "escalated_items": details,
    }


def export_example_trace(con: duckdb.DuckDBPyConnection, rubric_version: str,
                         out_dir: Path) -> Path | None:
    """One verbatim trace of the agent catching a SELF_CONTRADICTION."""
    row = con.execute(
        """
        SELECT t.item_id, t.trace FROM traces t
        JOIN submissions s USING (item_id)
        JOIN grades g ON g.item_id = t.item_id AND g.order_label = t.order_label
                     AND g.mode = t.mode AND g.rubric_version = t.rubric_version
        WHERE t.mode='agent' AND t.order_label='original' AND t.rubric_version=?
          AND s.defect_type='SELF_CONTRADICTION'
          AND (g.verdict='reject' OR g.escalated)
        ORDER BY t.item_id LIMIT 1
        """,
        [rubric_version],
    ).fetchone()
    if row is None:
        return None
    item_id, trace_json = row
    path = out_dir / "example_trace.json"
    path.write_text(json.dumps(
        {"item_id": item_id, "defect_type": "SELF_CONTRADICTION",
         "trace": json.loads(trace_json)},
        indent=2,
    ))
    return path


def render_markdown(df: pl.DataFrame, con: duckdb.DuckDBPyConnection,
                    thresholds: Thresholds, rubric_version: str) -> str:
    parts = ["# second-reader metrics", "",
             "## Ablation: recall by defect type, tools on vs off (operating point)",
             "", ablation_table(con, thresholds, rubric_version), "",
             "## Tool-use metrics", "",
             "```json",
             json.dumps(tool_metrics(con, thresholds, rubric_version), indent=2),
             "```", "",
             "## Escalation paths", "",
             "```json",
             json.dumps(escalation_report(con, thresholds, rubric_version), indent=2),
             "```", "",
             "## Calibration sweep", ""]
    header = ("| Mode | HIGH | LOW | Flagged | Escalated | Precision | Recall | Catch | "
              "Review load | $/1k | " + " | ".join(f"R:{dt}" for dt in DEFECT_TYPES) + " |")
    parts.append(header)
    parts.append("|" + "---|" * (10 + len(DEFECT_TYPES)))
    for r in df.iter_rows(named=True):
        marker = " ←" if (r["high"] == thresholds.high and r["low"] == thresholds.low) else ""
        cells = [
            r["mode"], f"{r['high']:.2f}{marker}", f"{r['low']:.2f}",
            str(r["n_flagged"]), str(r["n_escalated"]),
            _fmt(r["precision"], pct=True), _fmt(r["recall"], pct=True),
            _fmt(r["catch_rate"], pct=True), _fmt(r["review_load"], pct=True),
            f"${r['cost_per_1000_usd']:.2f}",
        ] + [_fmt(r[f"recall_{dt}"], pct=True) for dt in DEFECT_TYPES]
        parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)


def write_results(con: duckdb.DuckDBPyConnection, df: pl.DataFrame,
                  thresholds: Thresholds, rubric_version: str,
                  out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "calibration.csv"
    md_path = out_dir / "metrics.md"
    df.write_csv(csv_path)
    md_path.write_text(render_markdown(df, con, thresholds, rubric_version) + "\n")
    export_example_trace(con, rubric_version, out_dir)
    return csv_path, md_path
