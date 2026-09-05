"""Component 1b: defect injection, now at two scopes.

Annotator-scoped (applied during submission generation, ALL items of the
defective annotator):
  BOILERPLATE     one justification template reused verbatim across all items
  POSITION_BIAS   vote forced to Response A on every item

Item-scoped (injected into 25% of items belonging to otherwise-clean
annotators, seed 42):
  RUSHED              justification under 10 words, generic
  SELF_CONTRADICTION  justification argues for the other response

Ground truth is recorded at both levels: `annotators.is_defective` and, for
every defective item regardless of scope, a row in `defects`.
"""

from __future__ import annotations

import random

import duckdb

from .schemas import Item, Submission

DEFECT_TYPES = ["RUSHED", "BOILERPLATE", "POSITION_BIAS", "SELF_CONTRADICTION"]
ITEM_DEFECT_TYPES = ["RUSHED", "SELF_CONTRADICTION"]
SEED = 42
ITEM_DEFECT_FRACTION = 0.25

RUSHED_TEXTS = [
    "{v} is just better.",
    "{v} obviously.",
    "Went with {v}, reads better.",
    "{v}. No contest.",
]

# One template per BOILERPLATE annotator, reused verbatim on ALL their items.
BOILERPLATE_TEMPLATES = [
    "The response I picked is more helpful, better structured, and more "
    "accurate overall, so it was the clear choice.",
    "I chose this one because it answers the question well and is easy to "
    "follow from start to finish.",
]

def swap_letters(text: str) -> str:
    """Swap 'Response A' <-> 'Response B' references in a justification."""
    placeholder = "\x00RESP\x00"
    text = text.replace("Response A", placeholder)
    text = text.replace("Response B", "Response A")
    return text.replace(placeholder, "Response B")


def swapped_view(item: Item, sub: Submission) -> tuple[Item, Submission]:
    """The order-swap consistency view: relabel A<->B everywhere.

    The semantic content is identical, so a well-behaved grader should return
    the same verdict. A flip is a grader-instability signal.
    """
    item2 = item.model_copy(update={
        "response_a": item.response_b,
        "response_b": item.response_a,
        "model_a": item.model_b,
        "model_b": item.model_a,
        "human_gold_vote": "B" if item.human_gold_vote == "A" else "A",
    })
    sub2 = sub.model_copy(update={
        "vote": "B" if sub.vote == "A" else "A",
        "justification": swap_letters(sub.justification),
    })
    return item2, sub2


def make_item_defective(sub: Submission, defect_type: str, rng: random.Random) -> Submission:
    """Apply an item-scoped defect (RUSHED / SELF_CONTRADICTION) to a clean submission."""
    if defect_type == "RUSHED":
        text = rng.choice(RUSHED_TEXTS).format(v=sub.vote)
        return sub.model_copy(update={
            "justification": text, "is_defective": True, "defect_type": "RUSHED",
        })
    if defect_type == "SELF_CONTRADICTION":
        # Vote unchanged; justification argues for the OTHER response.
        return sub.model_copy(update={
            "justification": swap_letters(sub.justification),
            "is_defective": True, "defect_type": "SELF_CONTRADICTION",
        })
    raise ValueError(f"not an item-scoped defect type: {defect_type}")


def plan_item_injection(clean_item_ids: list[str], seed: int = SEED,
                        fraction: float = ITEM_DEFECT_FRACTION) -> dict[str, str]:
    """Deterministic item_id -> RUSHED|SELF_CONTRADICTION over clean-annotator items."""
    rng = random.Random(seed + 3)
    ordered = sorted(clean_item_ids)
    n = int(len(ordered) * fraction)
    chosen = rng.sample(ordered, n)
    rng.shuffle(chosen)
    return {item_id: ITEM_DEFECT_TYPES[i % 2] for i, item_id in enumerate(chosen)}


def inject_item_defects(con: duckdb.DuckDBPyConnection, seed: int = SEED) -> dict[str, int]:
    """Inject RUSHED / SELF_CONTRADICTION into clean-annotator items.

    Runs after submission generation. Idempotent: skips if item-scoped
    defects already exist. Returns counts by type.
    """
    existing = con.execute(
        "SELECT count(*) FROM defects WHERE scope='item'"
    ).fetchone()[0]
    if existing > 0:
        return dict(con.execute(
            "SELECT defect_type, count(*) FROM defects WHERE scope='item' GROUP BY 1"
        ).fetchall())

    clean_ids = [r[0] for r in con.execute("""
        SELECT s.item_id FROM submissions s
        JOIN annotators a USING (annotator_id)
        WHERE NOT a.is_defective
    """).fetchall()]
    plan = plan_item_injection(clean_ids, seed=seed)
    rng = random.Random(seed + 4)

    counts: dict[str, int] = {}
    for item_id in sorted(plan):
        defect_type = plan[item_id]
        cur = con.execute("SELECT * FROM submissions WHERE item_id=?", [item_id])
        cols = [d[0] for d in cur.description]
        sub = Submission(**dict(zip(cols, cur.fetchone())))
        bad = make_item_defective(sub, defect_type, rng)
        con.execute(
            "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?,?)",
            [bad.item_id, bad.annotator_id, bad.vote, bad.justification,
             bad.is_defective, bad.defect_type],
        )
        con.execute(
            "INSERT INTO defects (item_id, defect_type, scope) VALUES (?,?, 'item')",
            [item_id, defect_type],
        )
        counts[defect_type] = counts.get(defect_type, 0) + 1
    return counts


def record_annotator_defect(con: duckdb.DuckDBPyConnection, item_id: str,
                            defect_type: str) -> None:
    """Item-level ground-truth row for an item owned by a defective annotator."""
    con.execute(
        "INSERT OR REPLACE INTO defects (item_id, defect_type, scope) VALUES (?,?, 'annotator')",
        [item_id, defect_type],
    )
