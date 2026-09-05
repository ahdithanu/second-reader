"""Component 1a: the annotator model.

15 synthetic annotators, each assigned 8-12 items (seed 42). Two annotators
are BOILERPLATE (reuse one justification template across ALL their items) and
two are POSITION_BIAS (always vote Response A). Those two defect classes are
only detectable via annotator history — that is the point.
"""

from __future__ import annotations

import random

import duckdb

from .config import N_ANNOTATORS
from .schemas import Annotator

SEED = 42
MIN_ITEMS, MAX_ITEMS_PER = 8, 12
N_BOILERPLATE_ANNOTATORS = 2
N_POSITION_BIAS_ANNOTATORS = 2


def plan_assignment(item_ids: list[str], seed: int = SEED,
                    n_annotators: int = N_ANNOTATORS) -> dict[str, str]:
    """Deterministic item_id -> annotator_id, each annotator getting 8-12 items."""
    n = len(item_ids)
    base, rem = divmod(n, n_annotators)
    counts = [base + (1 if i < rem else 0) for i in range(n_annotators)]
    rng = random.Random(seed)
    # Perturb counts within [MIN, MAX] while keeping the total fixed.
    for _ in range(n_annotators * 3):
        i, j = rng.randrange(n_annotators), rng.randrange(n_annotators)
        if i != j and counts[i] < MAX_ITEMS_PER and counts[j] > MIN_ITEMS:
            counts[i] += 1
            counts[j] -= 1
    assert sum(counts) == n and all(MIN_ITEMS <= c <= MAX_ITEMS_PER for c in counts)

    shuffled = sorted(item_ids)
    rng.shuffle(shuffled)
    plan: dict[str, str] = {}
    pos = 0
    for i, c in enumerate(counts):
        for item_id in shuffled[pos:pos + c]:
            plan[item_id] = f"ann_{i:02d}"
        pos += c
    return plan


def plan_defective_annotators(seed: int = SEED,
                              n_annotators: int = N_ANNOTATORS) -> dict[str, str]:
    """annotator_id -> BOILERPLATE | POSITION_BIAS for the 4 defective annotators."""
    rng = random.Random(seed * 7 + 1)
    ids = [f"ann_{i:02d}" for i in range(n_annotators)]
    chosen = rng.sample(ids, N_BOILERPLATE_ANNOTATORS + N_POSITION_BIAS_ANNOTATORS)
    return {
        **{a: "BOILERPLATE" for a in chosen[:N_BOILERPLATE_ANNOTATORS]},
        **{a: "POSITION_BIAS" for a in chosen[N_BOILERPLATE_ANNOTATORS:]},
    }


def setup(con: duckdb.DuckDBPyConnection, seed: int = SEED) -> dict[str, int]:
    """Write annotators + assignments. Idempotent. Returns items per annotator."""
    if con.execute("SELECT count(*) FROM annotators").fetchone()[0] == 0:
        item_ids = [r[0] for r in con.execute("SELECT item_id FROM items").fetchall()]
        plan = plan_assignment(item_ids, seed=seed)
        bad = plan_defective_annotators(seed=seed)
        for i in range(N_ANNOTATORS):
            aid = f"ann_{i:02d}"
            con.execute(
                "INSERT INTO annotators VALUES (?,?,?)",
                [aid, aid in bad, bad.get(aid)],
            )
        con.executemany(
            "INSERT INTO assignments VALUES (?,?)",
            [(item_id, aid) for item_id, aid in plan.items()],
        )
    return dict(con.execute(
        "SELECT annotator_id, count(*) FROM assignments GROUP BY 1 ORDER BY 1"
    ).fetchall())


def get_annotators(con: duckdb.DuckDBPyConnection) -> list[Annotator]:
    cur = con.execute("SELECT * FROM annotators ORDER BY annotator_id")
    cols = [d[0] for d in cur.description]
    return [Annotator(**dict(zip(cols, r))) for r in cur.fetchall()]


def assignment_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    return dict(con.execute("SELECT item_id, annotator_id FROM assignments").fetchall())
