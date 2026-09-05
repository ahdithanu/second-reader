"""Component 2: the grader as a tool-using agent loop.

Starting context is deliberately partial: question, TRUNCATED responses,
vote, justification, annotator id. The agent decides what evidence it needs
via DuckDB-backed tools (no network), under a hard budget of
`max_tool_calls` evidence calls. It terminates by calling emit_verdict or
flag_for_review; on hitting the cap it is FORCED to emit a verdict on the
available evidence, with budget_exhausted recorded.

The full trace — every tool call in order with args plus per-round token and
latency counts — is a deliverable, stored in the traces table.

Ablation mode (`tools_on=False`): a single call on the same truncated
context, no evidence gathering, forced emit_verdict.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import duckdb
from pydantic import ValidationError

from .config import Rubric, Thresholds
from .defects import swap_letters
from .grader import GraderError
from .schemas import GraderOutput, Item, Submission

MAX_VALIDATION_FAILURES = 3

EVIDENCE_TOOLS = [
    {
        "name": "get_annotator_history",
        "description": "Return this annotator's submissions on OTHER items: their vote and justification for each. Use it to spot cross-item patterns (template reuse, always voting the same side).",
        "input_schema": {
            "type": "object",
            "properties": {
                "annotator_id": {"type": "string"},
                "limit": {"type": "integer", "description": "max rows, default 10"},
            },
            "required": ["annotator_id"],
        },
    },
    {
        "name": "get_peer_judgments",
        "description": "Return other human judges' votes (A, B, or tie) on this same question and response pair. Use it to sanity-check whether the annotator's vote is defensible.",
        "input_schema": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        },
    },
    {
        "name": "get_full_response",
        "description": "Return the FULL text of one response (the starting context only shows the first part).",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "which": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["item_id", "which"],
        },
    },
]

FLAG_TOOL = {
    "name": "flag_for_review",
    "description": "TERMINATING. Escalate this submission to a human reviewer instead of emitting a verdict. Use only when you genuinely cannot decide from the available evidence.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    },
}


def emit_verdict_tool(rubric: Rubric) -> dict:
    return {
        "name": "emit_verdict",
        "description": "TERMINATING. Emit your final grade for this submission.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dimension_scores": {
                    "type": "object",
                    "properties": {name: {"type": "integer", "minimum": 0, "maximum": 5}
                                   for name in rubric.dimension_names},
                    "required": rubric.dimension_names,
                    "additionalProperties": False,
                },
                "verdict": {"type": "string", "enum": ["accept", "reject"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["dimension_scores", "verdict", "confidence", "reason"],
        },
    }


class ToolExecutor:
    """Executes evidence tools against DuckDB, order-aware.

    In the swapped run, the item/submission passed in are already the
    relabeled copies; history and peer votes are relabeled here so the agent
    sees one consistent world in either order.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, item: Item,
                 sub: Submission, swapped: bool, thresholds: Thresholds):
        self.con = con
        self.item = item
        self.sub = sub
        self.swapped = swapped
        self.thresholds = thresholds

    def execute(self, name: str, args: dict) -> str:
        if name == "get_annotator_history":
            return self._history(args)
        if name == "get_peer_judgments":
            return self._peers(args)
        if name == "get_full_response":
            return self._full_response(args)
        return json.dumps({"error": f"unknown tool {name}"})

    def _flip(self, vote: str) -> str:
        if not self.swapped or vote == "tie":
            return vote
        return "B" if vote == "A" else "A"

    def _history(self, args: dict) -> str:
        if args.get("annotator_id") != self.sub.annotator_id:
            return json.dumps({"error": f"only annotator {self.sub.annotator_id} is in scope for this item"})
        limit = min(int(args.get("limit", self.thresholds.history_limit)), 20)
        rows = self.con.execute(
            "SELECT item_id, vote, justification FROM submissions "
            "WHERE annotator_id=? AND item_id != ? ORDER BY item_id LIMIT ?",
            [self.sub.annotator_id, self.sub.item_id, limit],
        ).fetchall()
        out = [
            {
                "item_id": item_id,
                "vote": self._flip(vote),
                "justification": swap_letters(justification) if self.swapped else justification,
            }
            for item_id, vote, justification in rows
        ]
        return json.dumps({"annotator_id": self.sub.annotator_id, "submissions": out})

    def _peers(self, args: dict) -> str:
        if args.get("item_id") != self.sub.item_id:
            return json.dumps({"error": f"only item {self.sub.item_id} is in scope"})
        rows = self.con.execute(
            "SELECT judge, vote FROM peer_votes WHERE item_id=? ORDER BY judge",
            [self.sub.item_id],
        ).fetchall()
        return json.dumps({
            "item_id": self.sub.item_id,
            "peer_votes": [{"judge": j, "vote": self._flip(v)} for j, v in rows],
            "note": "votes from other human judges on the same response pair",
        })

    def _full_response(self, args: dict) -> str:
        which = args.get("which")
        if which not in ("a", "b"):
            return json.dumps({"error": "which must be 'a' or 'b'"})
        # self.item is already the swapped copy in the swapped run.
        text = self.item.response_a if which == "a" else self.item.response_b
        return json.dumps({"which": which.upper(), "full_text": text})


def _rubric_block(rubric: Rubric) -> str:
    return "\n".join(
        f"- {name} (weight {spec['weight']}): {spec['description'].strip()}"
        for name, spec in rubric.dimensions.items()
    )


def build_agent_prompt(item: Item, sub: Submission, rubric: Rubric,
                       thresholds: Thresholds, tools_on: bool) -> str:
    t = thresholds.truncate_chars

    def trunc(text: str) -> str:
        if len(text) <= t:
            return text
        return f"{text[:t]}… [TRUNCATED — {len(text)} chars total]"

    header = f"""You are a QA grader for a human data labeling operation. An annotator was shown a question and two AI responses, voted for the better one, and wrote a justification. Grade the QUALITY OF THE ANNOTATOR'S WORK — not the AI responses themselves.

## Question
{item.question}

## Response A (truncated)
{trunc(item.response_a)}

## Response B (truncated)
{trunc(item.response_b)}

## Annotator submission
Annotator: {sub.annotator_id}
Item: {sub.item_id}
Vote: Response {sub.vote} is better.
Justification: {sub.justification}

## Rubric ({rubric.version})
{_rubric_block(rubric)}

Verdict rule: {rubric.verdict_rule.strip()}"""

    if tools_on:
        return header + f"""

Your starting context is deliberately partial. Decide what else you need: you may call get_annotator_history, get_peer_judgments, and get_full_response, with a hard budget of {thresholds.max_tool_calls} evidence calls total. When you have enough evidence, call emit_verdict. If you genuinely cannot decide, call flag_for_review with a reason — that escalates to a human."""
    return header + """

Grade on this context alone, then call emit_verdict."""


@dataclass
class AgentResult:
    output: GraderOutput | None          # None iff escalated
    escalated: bool = False
    escalation_reason: str | None = None
    budget_exhausted: bool = False
    n_tool_calls: int = 0
    trace: list = field(default_factory=list)
    rounds: list = field(default_factory=list)   # per-API-round token/latency
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


def run_grader(client, con: duckdb.DuckDBPyConnection, item: Item,
               sub: Submission, rubric: Rubric, thresholds: Thresholds,
               swapped: bool, tools_on: bool) -> AgentResult:
    """Run the grader agent loop (or the tools-off single call) for one item."""
    executor = ToolExecutor(con, item, sub, swapped, thresholds)
    emit_tool = emit_verdict_tool(rubric)
    if tools_on:
        tools = EVIDENCE_TOOLS + [FLAG_TOOL, emit_tool]
        tool_choice = {"type": "any"}
    else:
        tools = [emit_tool]
        tool_choice = {"type": "tool", "name": "emit_verdict"}

    messages: list = [{
        "role": "user",
        "content": build_agent_prompt(item, sub, rubric, thresholds, tools_on),
    }]
    res = AgentResult(output=None)
    validation_failures = 0
    forced = False
    t0 = time.monotonic()

    while True:
        r0 = time.monotonic()
        resp = client.messages.create(
            model=thresholds.model, max_tokens=1500,
            tools=tools, tool_choice=tool_choice, messages=messages,
        )
        round_ms = (time.monotonic() - r0) * 1000
        res.tokens_in += resp.usage.input_tokens
        res.tokens_out += resp.usage.output_tokens
        res.rounds.append({
            "tokens_in": resp.usage.input_tokens,
            "tokens_out": resp.usage.output_tokens,
            "latency_ms": round(round_ms, 1),
        })
        res.trace.append({
            "type": "api_round",
            "tokens_in": resp.usage.input_tokens,
            "tokens_out": resp.usage.output_tokens,
            "latency_ms": round(round_ms, 1),
        })

        tool_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_blocks:
            validation_failures += 1
            if validation_failures >= MAX_VALIDATION_FAILURES:
                raise GraderError(item.item_id, "swapped" if swapped else "original",
                                  "model produced no tool call")
            messages = messages + [{
                "role": "user",
                "content": "You must respond by calling a tool (emit_verdict to finish).",
            }]
            continue

        messages = messages + [{"role": "assistant", "content": resp.content}]
        results = []
        for block in tool_blocks:
            args = dict(block.input) if block.input else {}

            if block.name == "emit_verdict":
                try:
                    out = GraderOutput.model_validate(args)
                    out.check_dimensions(rubric.dimension_names)
                except (ValidationError, ValueError) as e:
                    validation_failures += 1
                    res.trace.append({"type": "invalid_verdict", "error": str(e)})
                    if validation_failures >= MAX_VALIDATION_FAILURES:
                        raise GraderError(
                            item.item_id, "swapped" if swapped else "original", str(e)
                        ) from e
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Validation failed: {e}. Call emit_verdict again with corrected input.",
                        "is_error": True,
                    })
                    continue
                res.output = out
                res.budget_exhausted = forced
                res.trace.append({
                    "type": "verdict", "verdict": out.verdict,
                    "confidence": out.confidence, "reason": out.reason,
                    "budget_exhausted": forced,
                })
                res.latency_ms = (time.monotonic() - t0) * 1000
                return res

            if block.name == "flag_for_review":
                res.escalated = True
                res.escalation_reason = str(args.get("reason", "")).strip() or "(no reason given)"
                res.trace.append({"type": "escalation", "reason": res.escalation_reason})
                res.latency_ms = (time.monotonic() - t0) * 1000
                return res

            # Evidence tool.
            if res.n_tool_calls >= thresholds.max_tool_calls:
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": "Tool budget exhausted.", "is_error": True,
                })
                continue
            res.n_tool_calls += 1
            result_text = executor.execute(block.name, args)
            res.trace.append({"type": "tool_call", "name": block.name, "args": args})
            res.trace.append({
                "type": "tool_result", "name": block.name,
                "chars": len(result_text), "preview": result_text[:200],
            })
            results.append({
                "type": "tool_result", "tool_use_id": block.id, "content": result_text,
            })

        content: list = list(results)
        if res.n_tool_calls >= thresholds.max_tool_calls and not forced:
            forced = True
            tools = [emit_tool]
            tool_choice = {"type": "tool", "name": "emit_verdict"}
            content.append({
                "type": "text",
                "text": "Tool budget exhausted. You MUST now emit a verdict on the available evidence.",
            })
            res.trace.append({"type": "budget_exhausted"})
        messages = messages + [{"role": "user", "content": content}]
