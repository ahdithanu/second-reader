"""Component 2b: defect injection.

Four defect types injected into a labeled 25% subset (seed 42). Ground truth
is recorded in the `defects` table so recall is unambiguous. Also home of the
letter-swap helpers used by both SELF_CONTRADICTION and the order-swap check.
"""

from __future__ import annotations

import random

import duckdb

from .schemas import Item, Submission

DEFECT_TYPES = ["RUSHED", "BOILERPLATE", "POSITION_BIAS", "SELF_CONTRADICTION"]
SEED = 42
DEFECT_FRACTION = 0.25

RUSHED_TEXTS = [
    "{v} is just better.",
    "{v} obviously.",
    "Went with {v}, reads better.",
    "{v}. No contest.",
]

# Verbatim-identical across every BOILERPLATE item — that IS the defect.
BOILERPLATE_TEXT = (
    "The response I picked is more helpful, better structured, and more "
    "accurate overall, so it was the clear choice."
)

POSITION_BIAS_TEXTS = [
    "Response A gets to the point faster and covers what was asked, so I went with it.",
    "I found Response A clearer and more complete than Response B.",
    "Response A does a better job answering the question directly, which decided it for me.",
    "Response A is stronger overall — better flow and it addresses the question head on.",
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


def make_defective(sub: Submission, defect_type: str, rng: random.Random) -> Submission:
    """Apply one defect to a clean submission."""
    if defect_type == "RUSHED":
        text = rng.choice(RUSHED_TEXTS).format(v=sub.vote)
        return sub.model_copy(update={
            "justification": text, "is_defective": True, "defect_type": "RUSHED",
        })
    if defect_type == "BOILERPLATE":
        return sub.model_copy(update={
            "justification": BOILERPLATE_TEXT,
            "is_defective": True, "defect_type": "BOILERPLATE",
        })
    if defect_type == "POSITION_BIAS":
        # Vote forced to A regardless of content. When the gold vote was
        # already A this defect is nearly invisible by construction — that
        # shows up honestly in recall-by-type.
        return sub.model_copy(update={
            "vote": "A",
            "justification": rng.choice(POSITION_BIAS_TEXTS),
            "is_defective": True, "defect_type": "POSITION_BIAS",
        })
    if defect_type == "SELF_CONTRADICTION":
        # Vote unchanged; justification argues for the OTHER response.
        return sub.model_copy(update={
            "justification": swap_letters(sub.justification),
            "is_defective": True, "defect_type": "SELF_CONTRADICTION",
        })
    raise ValueError(f"unknown defect type: {defect_type}")


def plan_injection(item_ids: list[str], seed: int = SEED,
                   fraction: float = DEFECT_FRACTION) -> dict[str, str]:
    """Deterministic plan: item_id -> defect_type for the injected subset."""
    rng = random.Random(seed)
    ordered = sorted(item_ids)
    n_defects = int(len(ordered) * fraction)
    chosen = rng.sample(ordered, n_defects)
    rng.shuffle(chosen)
    return {item_id: DEFECT_TYPES[i % len(DEFECT_TYPES)] for i, item_id in enumerate(chosen)}


def inject(con: duckdb.DuckDBPyConnection, seed: int = SEED,
           fraction: float = DEFECT_FRACTION) -> dict[str, int]:
    """Inject defects into the submissions table; record ground truth in `defects`.

    Idempotent: if the defects table is already populated, do nothing.
    Returns counts by defect type.
    """
    existing = con.execute("SELECT count(*) FROM defects").fetchone()[0]
    if existing > 0:
        rows = con.execute(
            "SELECT defect_type, count(*) FROM defects GROUP BY defect_type"
        ).fetchall()
        return dict(rows)

    item_ids = [r[0] for r in con.execute("SELECT item_id FROM submissions").fetchall()]
    plan = plan_injection(item_ids, seed=seed, fraction=fraction)
    rng = random.Random(seed + 1)

    counts: dict[str, int] = {}
    for item_id in sorted(plan):
        defect_type = plan[item_id]
        cur = con.execute("SELECT * FROM submissions WHERE item_id=?", [item_id])
        cols = [d[0] for d in cur.description]
        sub = Submission(**dict(zip(cols, cur.fetchone())))
        bad = make_defective(sub, defect_type, rng)
        con.execute(
            "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?)",
            [bad.item_id, bad.vote, bad.justification, bad.is_defective, bad.defect_type],
        )
        con.execute(
            "INSERT INTO defects (item_id, defect_type) VALUES (?,?)",
            [item_id, defect_type],
        )
        counts[defect_type] = counts.get(defect_type, 0) + 1
    return counts
