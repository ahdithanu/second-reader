"""Component 1c: submission generation, annotator-aware.

Clean annotators get a Claude-written justification for the human gold vote.
BOILERPLATE annotators reuse their fixed template (no API call — that's the
defect). POSITION_BIAS annotators vote A on everything and get a generated
justification arguing for A. Item-scoped defects are injected afterwards.
"""

from __future__ import annotations

import time

import duckdb

from . import annotators as ann_mod
from . import defects
from .config import Thresholds
from .grader import CallStats
from .schemas import Item, Submission

ANNOTATOR_PROMPT = """You are simulating a human annotator on a data labeling platform. You were shown a question and two AI responses, and you decided that Response {vote} is better.

## Question
{question}

## Response A
{response_a}

## Response B
{response_b}

Write the justification you would submit with your vote for Response {vote}. Requirements:
- 2 to 4 sentences, first person, plain conversational tone.
- Always refer to the responses as "Response A" and "Response B" (exactly that phrasing).
- Cite at least one concrete property of the responses (accuracy, completeness, structure, an error, etc.).
- Argue for Response {vote} even if the case is thin; do not hedge or mention voting the other way.
- Do not mention that you are an AI or that this is simulated.

Return only the justification text, nothing else."""


def annotate_call(client, item: Item, vote: str,
                  thresholds: Thresholds) -> tuple[str, CallStats]:
    prompt = ANNOTATOR_PROMPT.format(
        vote=vote, question=item.question,
        response_a=item.response_a, response_b=item.response_b,
    )
    t0 = time.monotonic()
    resp = client.messages.create(
        model=thresholds.model, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    stats = CallStats(
        tokens_in=resp.usage.input_tokens, tokens_out=resp.usage.output_tokens,
        latency_ms=(time.monotonic() - t0) * 1000, attempts=1,
    )
    return resp.content[0].text.strip(), stats


def save_submission(con, sub: Submission) -> None:
    con.execute(
        "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?,?)",
        [sub.item_id, sub.annotator_id, sub.vote, sub.justification,
         sub.is_defective, sub.defect_type],
    )


def submitted_ids(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT item_id FROM submissions").fetchall()}


def get_submissions(con) -> list[Submission]:
    cur = con.execute("SELECT * FROM submissions ORDER BY item_id")
    cols = [d[0] for d in cur.description]
    return [Submission(**dict(zip(cols, r))) for r in cur.fetchall()]


def generate_missing(con: duckdb.DuckDBPyConnection, client, items: list[Item],
                     thresholds: Thresholds, log_fn) -> int:
    """Generate submissions for items that lack one. Annotator-defect-aware."""
    assignment = ann_mod.assignment_map(con)
    by_id = {a.annotator_id: a for a in ann_mod.get_annotators(con)}
    boiler_ids = sorted(a.annotator_id for a in by_id.values()
                        if a.defect_type == "BOILERPLATE")
    done = submitted_ids(con)

    n_new = 0
    for item in items:
        if item.item_id in done:
            continue
        annotator = by_id[assignment[item.item_id]]

        if annotator.defect_type == "BOILERPLATE":
            template = defects.BOILERPLATE_TEMPLATES[
                boiler_ids.index(annotator.annotator_id) % len(defects.BOILERPLATE_TEMPLATES)
            ]
            sub = Submission(
                item_id=item.item_id, annotator_id=annotator.annotator_id,
                vote=item.human_gold_vote, justification=template,
                is_defective=True, defect_type="BOILERPLATE",
            )
            defects.record_annotator_defect(con, item.item_id, "BOILERPLATE")
        elif annotator.defect_type == "POSITION_BIAS":
            justification, stats = annotate_call(client, item, "A", thresholds)
            log_fn(item.item_id, "annotate", stats)
            sub = Submission(
                item_id=item.item_id, annotator_id=annotator.annotator_id,
                vote="A", justification=justification,
                is_defective=True, defect_type="POSITION_BIAS",
            )
            defects.record_annotator_defect(con, item.item_id, "POSITION_BIAS")
        else:
            justification, stats = annotate_call(
                client, item, item.human_gold_vote, thresholds
            )
            log_fn(item.item_id, "annotate", stats)
            sub = Submission(
                item_id=item.item_id, annotator_id=annotator.annotator_id,
                vote=item.human_gold_vote, justification=justification,
            )
        save_submission(con, sub)
        n_new += 1
    return n_new
