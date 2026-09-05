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

-- Other human judges' votes on the same (question, model pair), from the
-- full dataset. Powers the get_peer_judgments tool. 'tie' is allowed here.
CREATE TABLE IF NOT EXISTS peer_votes (
    item_id TEXT,
    judge TEXT,
    vote TEXT CHECK (vote IN ('A', 'B', 'tie')),
    PRIMARY KEY (item_id, judge)
);

CREATE TABLE IF NOT EXISTS annotators (
    annotator_id TEXT PRIMARY KEY,
    is_defective BOOLEAN DEFAULT FALSE,
    defect_type TEXT CHECK (defect_type IN ('BOILERPLATE', 'POSITION_BIAS'))
);

-- Which annotator owns which item; fixed before any generation (seed 42).
CREATE TABLE IF NOT EXISTS assignments (
    item_id TEXT PRIMARY KEY,
    annotator_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    item_id TEXT PRIMARY KEY,
    annotator_id TEXT NOT NULL,
    vote TEXT CHECK (vote IN ('A', 'B')),
    justification TEXT,
    is_defective BOOLEAN DEFAULT FALSE,
    defect_type TEXT
);

-- Item-level ground truth for every defective item, including items owned
-- by defective annotators. Annotator-level truth lives in `annotators`.
CREATE TABLE IF NOT EXISTS defects (
    item_id TEXT PRIMARY KEY,
    defect_type TEXT NOT NULL,
    scope TEXT CHECK (scope IN ('item', 'annotator')),
    injected_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS grades (
    item_id TEXT,
    order_label TEXT CHECK (order_label IN ('original', 'swapped')),
    mode TEXT CHECK (mode IN ('agent', 'single')),
    rubric_version TEXT,
    dimension_scores TEXT,        -- JSON; NULL when escalated
    verdict TEXT CHECK (verdict IN ('accept', 'reject')),  -- NULL when escalated
    confidence DOUBLE,
    reason TEXT,
    escalated BOOLEAN DEFAULT FALSE,
    budget_exhausted BOOLEAN DEFAULT FALSE,
    graded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (item_id, order_label, mode, rubric_version)
);

-- Full agent trace per (item, order, mode). A deliverable, not debug output.
CREATE TABLE IF NOT EXISTS traces (
    item_id TEXT,
    order_label TEXT,
    mode TEXT,
    rubric_version TEXT,
    trace TEXT,                   -- JSON list of events, in order
    n_tool_calls INTEGER,
    hit_cap BOOLEAN,
    escalated BOOLEAN,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (item_id, order_label, mode, rubric_version)
);

CREATE TABLE IF NOT EXISTS call_log (
    call_id TEXT PRIMARY KEY,
    item_id TEXT,
    purpose TEXT,                 -- 'grade' | 'grade_failed' | 'annotate'
    mode TEXT,                    -- 'agent' | 'single' | NULL for annotate
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
    item_id TEXT,
    mode TEXT,
    rubric_version TEXT,
    verdict_original TEXT,
    verdict_swapped TEXT,
    flipped BOOLEAN,
    confidence DOUBLE,            -- final acceptance score; NULL if escalated
    route TEXT CHECK (route IN ('auto_accept', 'auto_reject', 'human_review')),
    path TEXT CHECK (path IN ('AUTO', 'THRESHOLD_ROUTED', 'AGENT_ESCALATED', 'BOTH')),
    reason TEXT,
    PRIMARY KEY (item_id, mode)
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con
