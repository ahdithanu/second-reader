"""Load lmsys/mt_bench_human_judgments, sample deterministically, store in DuckDB."""

from __future__ import annotations

import hashlib
import random

import duckdb

from .schemas import Item

DATASET = "lmsys/mt_bench_human_judgments"
SEED = 42


def _stable_id(question_id: int, model_a: str, model_b: str, judge: str, turn: int) -> str:
    key = f"{question_id}|{model_a}|{model_b}|{judge}|{turn}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _extract(row: dict) -> Item | None:
    """Turn one raw HF row into an Item, or None if unusable.

    Only turn-1, non-tie judgments are kept: turn 2 needs multi-turn context
    and ties give no gold vote to anchor defect injection against.
    """
    if row["turn"] != 1 or row["winner"] not in ("model_a", "model_b"):
        return None
    conv_a, conv_b = row["conversation_a"], row["conversation_b"]
    if len(conv_a) < 2 or len(conv_b) < 2:
        return None
    question = conv_a[0]["content"]
    return Item(
        item_id=_stable_id(row["question_id"], row["model_a"], row["model_b"], row["judge"], row["turn"]),
        question_id=row["question_id"],
        judge=row["judge"],
        turn=row["turn"],
        model_a=row["model_a"],
        model_b=row["model_b"],
        question=question,
        response_a=conv_a[1]["content"],
        response_b=conv_b[1]["content"],
        human_gold_vote="A" if row["winner"] == "model_a" else "B",
    )


def load_items(con: duckdb.DuckDBPyConnection, max_items: int = 300) -> int:
    """Idempotent: if items are already loaded, do nothing and return the count."""
    existing = con.execute("SELECT count(*) FROM items").fetchone()[0]
    if existing >= max_items:
        return existing

    from datasets import load_dataset  # deferred: heavy import

    ds = load_dataset(DATASET, split="human")
    items: dict[str, Item] = {}
    for row in ds:
        item = _extract(row)
        if item is not None:
            items[item.item_id] = item  # dedupe on stable id

    ordered = sorted(items.values(), key=lambda it: it.item_id)
    rng = random.Random(SEED)
    rng.shuffle(ordered)
    sample = ordered[:max_items]

    con.execute("DELETE FROM items")
    con.executemany(
        "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                it.item_id, it.question_id, it.judge, it.turn, it.model_a,
                it.model_b, it.question, it.response_a, it.response_b,
                it.human_gold_vote,
            )
            for it in sample
        ],
    )
    return len(sample)


def get_items(con: duckdb.DuckDBPyConnection, limit: int | None = None) -> list[Item]:
    q = "SELECT * FROM items ORDER BY item_id"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = con.execute(q)
    cols = [d[0] for d in cur.description]
    return [Item(**dict(zip(cols, r))) for r in cur.fetchall()]
