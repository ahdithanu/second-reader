"""DuckDB schema and connection. One local file, no server."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    question_id INTEGER,
    judge TEXT,
    turn INTEGER,
    model_a TEXT,
    model_b TEXT,
    question TEXT,
    response_a TEXT,
    response_b TEXT,
    human_gold_vote TEXT CHECK (human_gold_vote IN ('A', 'B'))
);

CREATE TABLE IF NOT EXISTS submissions (
    item_id TEXT PRIMARY KEY,
    vote TEXT CHECK (vote IN ('A', 'B')),
    justification TEXT,
    is_defective BOOLEAN DEFAULT FALSE,
    defect_type TEXT
);

CREATE TABLE IF NOT EXISTS defects (
    item_id TEXT PRIMARY KEY,
    defect_type TEXT NOT NULL,
    injected_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS grades (
    item_id TEXT,
    order_label TEXT CHECK (order_label IN ('original', 'swapped')),
    rubric_version TEXT,
    dimension_scores TEXT,        -- JSON
    verdict TEXT CHECK (verdict IN ('accept', 'reject')),
    confidence DOUBLE,
    reason TEXT,
    graded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (item_id, order_label, rubric_version)
);

CREATE TABLE IF NOT EXISTS call_log (
    call_id TEXT PRIMARY KEY,
    item_id TEXT,
    purpose TEXT,                 -- 'grade' | 'annotate'
    order_label TEXT,
    rubric_version TEXT,
    model TEXT,
    mock BOOLEAN DEFAULT FALSE,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms DOUBLE,
    cost_usd DOUBLE,
    attempts INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS routed (
    item_id TEXT PRIMARY KEY,
    rubric_version TEXT,
    verdict_original TEXT,
    verdict_swapped TEXT,
    flipped BOOLEAN,
    confidence DOUBLE,            -- final acceptance score after flip penalty
    route TEXT CHECK (route IN ('auto_accept', 'auto_reject', 'human_review')),
    reason TEXT
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con
