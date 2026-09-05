"""Component 2a: synthetic annotator.

The dataset has human votes but no written justifications, so we generate one
per item: the human's actual vote plus a Claude-written justification. These
are the CLEAN submissions that defects are later injected into.
"""

from __future__ import annotations

import time

from .config import Thresholds
from .grader import CallStats
from .schemas import Item, Submission

ANNOTATOR_PROMPT = """You are simulating a careful human annotator on a data labeling platform. You were shown a question and two AI responses, and you decided that Response {vote} is better.

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
- Do not mention that you are an AI or that this is simulated.

Return only the justification text, nothing else."""


def annotate_call(
    client, item: Item, thresholds: Thresholds
) -> tuple[Submission, CallStats]:
    """Generate a clean submission: the human gold vote + a justification."""
    prompt = ANNOTATOR_PROMPT.format(
        vote=item.human_gold_vote,
        question=item.question,
        response_a=item.response_a,
        response_b=item.response_b,
    )
    t0 = time.monotonic()
    resp = client.messages.create(
        model=thresholds.model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    stats = CallStats(
        tokens_in=resp.usage.input_tokens,
        tokens_out=resp.usage.output_tokens,
        latency_ms=(time.monotonic() - t0) * 1000,
        attempts=1,
    )
    justification = resp.content[0].text.strip()
    sub = Submission(
        item_id=item.item_id, vote=item.human_gold_vote, justification=justification
    )
    return sub, stats


def save_submission(con, sub: Submission) -> None:
    con.execute(
        "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?)",
        [sub.item_id, sub.vote, sub.justification, sub.is_defective, sub.defect_type],
    )


def submitted_ids(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT item_id FROM submissions").fetchall()}


def get_submissions(con) -> list[Submission]:
    cur = con.execute("SELECT * FROM submissions ORDER BY item_id")
    cols = [d[0] for d in cur.description]
    return [Submission(**dict(zip(cols, r))) for r in cur.fetchall()]
