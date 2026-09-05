"""Load lmsys/mt_bench_human_judgments, sample deterministically, store in DuckDB.

Also extracts peer votes: other judges' votes on the same (question, model
pair) from the full dataset, which power the get_peer_judgments tool.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

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
    return Item(
        item_id=_stable_id(row["question_id"], row["model_a"], row["model_b"], row["judge"], row["turn"]),
        question_id=row["question_id"],
        judge=row["judge"],
        turn=row["turn"],
        model_a=row["model_a"],
        model_b=row["model_b"],
        question=conv_a[0]["content"],
        response_a=conv_a[1]["content"],
        response_b=conv_b[1]["content"],
        human_gold_vote="A" if row["winner"] == "model_a" else "B",
    )


def load_items(con: duckdb.DuckDBPyConnection, max_items: int = 150) -> int:
    """Idempotent: if items are already loaded, do nothing and return the count."""
    existing = con.execute("SELECT count(*) FROM items").fetchone()[0]
    if existing >= max_items:
        return existing

    from datasets import load_dataset  # deferred: heavy import

    ds = load_dataset(DATASET, split="human")

    # All turn-1 votes per (question, ordered model pair), for peer lookups.
    votes_by_pair: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    items: dict[str, Item] = {}
    for row in ds:
        if row["turn"] != 1:
            continue
        vote = {"model_a": "A", "model_b": "B", "tie": "tie"}.get(row["winner"])
        if vote:
            votes_by_pair[(row["question_id"], row["model_a"], row["model_b"])].append(
                (row["judge"], vote)
            )
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

    # Peers: same pair same order, plus the reversed pair with votes flipped.
    con.execute("DELETE FROM peer_votes")
    flip = {"A": "B", "B": "A", "tie": "tie"}
    for it in sample:
        peers: dict[str, str] = {}
        for judge, vote in votes_by_pair.get((it.question_id, it.model_a, it.model_b), []):
            peers[judge] = vote
        for judge, vote in votes_by_pair.get((it.question_id, it.model_b, it.model_a), []):
            peers.setdefault(judge, flip[vote])
        peers.pop(it.judge, None)  # the item's own judge is not a peer
        if peers:
            con.executemany(
                "INSERT OR IGNORE INTO peer_votes VALUES (?,?,?)",
                [(it.item_id, judge, vote) for judge, vote in peers.items()],
            )
    return len(sample)


def get_items(con: duckdb.DuckDBPyConnection, limit: int | None = None) -> list[Item]:
    q = "SELECT * FROM items ORDER BY item_id"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = con.execute(q)
    cols = [d[0] for d in cur.description]
    return [Item(**dict(zip(cols, r))) for r in cur.fetchall()]
