"""Component 1: the grader. One Claude call per (item, order) with bounded retry.

`grade_call` is pure API + validation (no DB) so it can be unit-tested with a
fake client and run concurrently. The pipeline owns logging and persistence.
Validation failures are fed back into the prompt; after MAX_ATTEMPTS the
grader fails loud with the item_id.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass

import duckdb
from pydantic import ValidationError

from .config import Rubric, Thresholds
from .schemas import GraderOutput, Item, Submission

MAX_ATTEMPTS = 3


class GraderError(RuntimeError):
    """Raised after MAX_ATTEMPTS failed validations. Carries the item_id."""

    def __init__(self, item_id: str, order_label: str, last_error: str):
        self.item_id = item_id
        super().__init__(
            f"grader failed on item_id={item_id} order={order_label} "
            f"after {MAX_ATTEMPTS} attempts: {last_error}"
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


def _rubric_block(rubric: Rubric) -> str:
    lines = []
    for name, spec in rubric.dimensions.items():
        lines.append(f"- {name} (weight {spec['weight']}): {spec['description'].strip()}")
    return "\n".join(lines)


def build_grader_prompt(item: Item, submission: Submission, rubric: Rubric) -> str:
    schema_hint = {
        "dimension_scores": {name: "<int 0-5>" for name in rubric.dimension_names},
        "verdict": "accept | reject",
        "confidence": "<float 0-1>",
        "reason": "<one or two sentences>",
    }
    return f"""You are a QA grader for a human data labeling operation. An annotator was shown a question and two AI responses, voted for the better one, and wrote a justification. Grade the QUALITY OF THE ANNOTATOR'S WORK — not the AI responses themselves.

## Question
{item.question}

## Response A
{item.response_a}

## Response B
{item.response_b}

## Annotator submission
Vote: Response {submission.vote} is better.
Justification: {submission.justification}

## Rubric ({rubric.version})
{_rubric_block(rubric)}

Verdict rule: {rubric.verdict_rule.strip()}

Respond with ONLY a JSON object, no prose before or after, exactly this shape:
{json.dumps(schema_hint, indent=2)}"""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def _parse_and_validate(raw_text: str, rubric: Rubric) -> GraderOutput:
    data = extract_json(raw_text)
    out = GraderOutput.model_validate(data)
    out.check_dimensions(rubric.dimension_names)
    return out


def grade_call(
    client,
    item: Item,
    submission: Submission,
    rubric: Rubric,
    thresholds: Thresholds,
) -> tuple[GraderOutput | None, CallStats]:
    """One grading call with bounded retry. Returns (None, stats) on failure."""
    prompt = build_grader_prompt(item, submission, rubric)
    messages = [{"role": "user", "content": prompt}]
    stats = CallStats()
    t0 = time.monotonic()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        stats.attempts = attempt
        resp = client.messages.create(
            model=thresholds.model, max_tokens=1024, messages=messages
        )
        raw = resp.content[0].text
        stats.tokens_in += resp.usage.input_tokens
        stats.tokens_out += resp.usage.output_tokens
        try:
            out = _parse_and_validate(raw, rubric)
        except (ValueError, ValidationError) as e:
            stats.last_error = str(e)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation:\n{stats.last_error}\n"
                        "Return ONLY the corrected JSON object, nothing else."
                    ),
                },
            ]
            continue
        stats.latency_ms = (time.monotonic() - t0) * 1000
        return out, stats

    stats.latency_ms = (time.monotonic() - t0) * 1000
    return None, stats


# --- persistence helpers (called from the pipeline, single-threaded) ---------

def log_call(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    purpose: str,
    order_label: str | None,
    rubric_version: str | None,
    model: str,
    mock: bool,
    stats: CallStats,
    thresholds: Thresholds,
) -> None:
    cost = 0.0 if mock else cost_usd(model, stats.tokens_in, stats.tokens_out, thresholds)
    con.execute(
        "INSERT INTO call_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
        [
            str(uuid.uuid4()), item_id, purpose, order_label, rubric_version,
            model, mock, stats.tokens_in, stats.tokens_out, stats.latency_ms,
            cost, stats.attempts,
        ],
    )


def save_grade(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    order_label: str,
    rubric_version: str,
    out: GraderOutput,
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO grades VALUES (?,?,?,?,?,?,?, current_timestamp)",
        [
            item_id, order_label, rubric_version,
            json.dumps(out.dimension_scores), out.verdict, out.confidence,
            out.reason,
        ],
    )


def graded_ids(
    con: duckdb.DuckDBPyConnection, order_label: str, rubric_version: str
) -> set[str]:
    rows = con.execute(
        "SELECT item_id FROM grades WHERE order_label=? AND rubric_version=?",
        [order_label, rubric_version],
    ).fetchall()
    return {r[0] for r in rows}
