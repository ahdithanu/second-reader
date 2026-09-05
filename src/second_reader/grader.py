"""Shared grader plumbing: errors, call logging, grade persistence.

The grading logic itself lives in agent.py (tool-using loop + tools-off
ablation single call).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import duckdb

from .config import Thresholds
from .schemas import GraderOutput

MAX_ATTEMPTS = 3  # bounded validation retries before failing loud


class GraderError(RuntimeError):
    """Raised after bounded validation retries. Carries the item_id."""

    def __init__(self, item_id: str, order_label: str, last_error: str):
        self.item_id = item_id
        super().__init__(
            f"grader failed on item_id={item_id} order={order_label}: {last_error}"
        )


@dataclass
class CallStats:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    attempts: int = 0
    last_error: str = ""


def cost_usd(model: str, tokens_in: int, tokens_out: int, thresholds: Thresholds) -> float:
    price = thresholds.pricing.get(model)
    if price is None:
        return 0.0
    return tokens_in / 1e6 * price["input"] + tokens_out / 1e6 * price["output"]


def log_call(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    purpose: str,
    mode: str | None,
    order_label: str | None,
    rubric_version: str | None,
    model: str,
    mock: bool,
    stats: CallStats,
    thresholds: Thresholds,
) -> None:
    cost = 0.0 if mock else cost_usd(model, stats.tokens_in, stats.tokens_out, thresholds)
    con.execute(
        "INSERT INTO call_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
        [
            str(uuid.uuid4()), item_id, purpose, mode, order_label, rubric_version,
            model, mock, stats.tokens_in, stats.tokens_out, stats.latency_ms,
            cost, stats.attempts,
        ],
    )


def save_grade(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    order_label: str,
    mode: str,
    rubric_version: str,
    out: GraderOutput | None,
    escalated: bool,
    escalation_reason: str | None,
    budget_exhausted: bool,
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO grades VALUES (?,?,?,?,?,?,?,?,?,?, current_timestamp)",
        [
            item_id, order_label, mode, rubric_version,
            json.dumps(out.dimension_scores) if out else None,
            out.verdict if out else None,
            out.confidence if out else None,
            out.reason if out else escalation_reason,
            escalated, budget_exhausted,
        ],
    )


def save_trace(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    order_label: str,
    mode: str,
    rubric_version: str,
    trace: list,
    n_tool_calls: int,
    hit_cap: bool,
    escalated: bool,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
        [
            item_id, order_label, mode, rubric_version, json.dumps(trace),
            n_tool_calls, hit_cap, escalated, tokens_in, tokens_out, latency_ms,
        ],
    )


def graded_ids(
    con: duckdb.DuckDBPyConnection, order_label: str, mode: str, rubric_version: str
) -> set[str]:
    rows = con.execute(
        "SELECT item_id FROM grades WHERE order_label=? AND mode=? AND rubric_version=?",
        [order_label, mode, rubric_version],
    ).fetchall()
    return {r[0] for r in rows}
