"""Component 5: calibration sweep and metrics.

The flag decision is `score <= low` (auto-reject). For each (high, low) pair:
  precision     of flagged items, how many were truly defective
  recall        of injected defects, how many were flagged
  review_load   fraction of items landing in the human review queue
  cost_per_1000 grading cost per 1,000 items, from logged token costs
plus recall broken out BY DEFECT TYPE — never averaged away.

`flag_metrics` is a pure function over plain records so the math is unit-
testable against a fixture with known labels.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from .config import RESULTS_DIR, Thresholds
from .defects import DEFECT_TYPES
from .router import acceptance_score, route_label


def flag_metrics(records: list[dict], high: float, low: float) -> dict:
    """records: [{score, is_defective, defect_type}]. Pure precision/recall math."""
    n = len(records)
    flagged = [r for r in records if r["score"] <= low]
    review = [r for r in records if low < r["score"] < high]
    true_defects = [r for r in records if r["is_defective"]]

    tp = sum(1 for r in flagged if r["is_defective"])
    precision = tp / len(flagged) if flagged else None
    recall = tp / len(true_defects) if true_defects else None

    by_type: dict[str, float | None] = {}
    for dt in DEFECT_TYPES:
        of_type = [r for r in true_defects if r["defect_type"] == dt]
        caught = sum(1 for r in of_type if r["score"] <= low)
        by_type[dt] = caught / len(of_type) if of_type else None

    return {
        "high": high,
        "low": low,
        "n_items": n,
        "n_flagged": len(flagged),
        "precision": precision,
        "recall": recall,
        "review_load": len(review) / n if n else None,
        "recall_by_type": by_type,
    }


def scored_records(con: duckdb.DuckDBPyConnection, rubric_version: str,
                   flip_penalty: float) -> list[dict]:
    """Per-item records with acceptance score and defect ground truth."""
    rows = con.execute(
        """
        SELECT o.item_id, o.verdict, o.confidence, (o.verdict != s.verdict) AS flipped,
               sub.is_defective, sub.defect_type
        FROM grades o
        JOIN grades s ON s.item_id = o.item_id
                     AND s.rubric_version = o.rubric_version
                     AND s.order_label = 'swapped'
        JOIN submissions sub ON sub.item_id = o.item_id
        WHERE o.order_label = 'original' AND o.rubric_version = ?
        ORDER BY o.item_id
        """,
        [rubric_version],
    ).fetchall()
    return [
        {
            "item_id": item_id,
            "score": acceptance_score(verdict, conf, bool(flipped), flip_penalty),
            "flipped": bool(flipped),
            "is_defective": bool(is_def),
            "defect_type": defect_type,
        }
        for item_id, verdict, conf, flipped, is_def, defect_type in rows
    ]


def grading_cost_per_1000(con: duckdb.DuckDBPyConnection, n_items: int) -> float | None:
    total = con.execute(
        "SELECT sum(cost_usd) FROM call_log WHERE purpose = 'grade'"
    ).fetchone()[0]
    if total is None or n_items == 0:
        return None
    return total / n_items * 1000


def sweep(con: duckdb.DuckDBPyConnection, thresholds: Thresholds,
          rubric_version: str) -> pl.DataFrame:
    records = scored_records(con, rubric_version, thresholds.flip_penalty)
    cost_1k = grading_cost_per_1000(con, len(records))

    rows = []
    for high in thresholds.sweep_high:
        for low in thresholds.sweep_low:
            if low >= high:
                continue
            m = flag_metrics(records, high, low)
            row = {
                "high": m["high"],
                "low": m["low"],
                "n_items": m["n_items"],
                "n_flagged": m["n_flagged"],
                "precision": m["precision"],
                "recall": m["recall"],
                "review_load": m["review_load"],
                "cost_per_1000_usd": cost_1k,
            }
            for dt in DEFECT_TYPES:
                row[f"recall_{dt}"] = m["recall_by_type"][dt]
            rows.append(row)
    return pl.DataFrame(rows)


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def render_markdown(df: pl.DataFrame, thresholds: Thresholds) -> str:
    """Markdown metrics table for the sweep, with the operating point marked."""
    lines = [
        "| HIGH | LOW | Flagged | Precision | Recall | Review load | $/1k items | "
        + " | ".join(f"R:{dt}" for dt in DEFECT_TYPES) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(DEFECT_TYPES),
    ]
    for r in df.iter_rows(named=True):
        marker = " ←" if (r["high"] == thresholds.high and r["low"] == thresholds.low) else ""
        cells = [
            f"{r['high']:.2f}{marker}", f"{r['low']:.2f}", str(r["n_flagged"]),
            _fmt(r["precision"], pct=True), _fmt(r["recall"], pct=True),
            _fmt(r["review_load"], pct=True),
            "—" if r["cost_per_1000_usd"] is None else f"${r['cost_per_1000_usd']:.2f}",
        ] + [_fmt(r[f"recall_{dt}"], pct=True) for dt in DEFECT_TYPES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(df: pl.DataFrame, thresholds: Thresholds,
                  out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "calibration.csv"
    md_path = out_dir / "metrics.md"
    df.write_csv(csv_path)
    md_path.write_text(render_markdown(df, thresholds) + "\n")
    return csv_path, md_path
