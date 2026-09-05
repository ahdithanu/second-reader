"""Third ablation arm: grader + adversarial verifier.

A second agent receives the first (agent-mode) grader's verdict on the same
truncated context and tries to OVERTURN it, with the same evidence tools
under a smaller budget. Combination rule (config/thresholds.yaml):

    uphold   -> verdict stands, confidence = mean(grader, verifier)
    overturn -> verdict flips,  confidence = verifier * (1 - overturn_penalty)

so an overturn lands near the review band unless the verifier is very sure —
disagreement between the two agents is itself a signal. If the first grader
escalated, the verifier arm inherits the escalation: there is no verdict to
verify, and inventing one would defeat the point.
"""

from __future__ import annotations

import time

import duckdb
from pydantic import ValidationError

from .agent import EVIDENCE_TOOLS, FLAG_TOOL, AgentResult, ToolExecutor, build_agent_prompt
from .config import Rubric, Thresholds
from .grader import GraderError
from .schemas import GraderOutput, Item, Submission, VerifierReview

MAX_VALIDATION_FAILURES = 3

EMIT_REVIEW_TOOL = {
    "name": "emit_review",
    "description": "TERMINATING. Emit your final review of the first grader's verdict: uphold it or overturn it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["uphold", "overturn"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["decision", "confidence", "reason"],
    },
}


def combine(grader_verdict: str, grader_conf: float, review: VerifierReview,
            overturn_penalty: float) -> tuple[str, float]:
    """Fold the verifier's review into the grader's verdict."""
    if review.decision == "uphold":
        return grader_verdict, (grader_conf + review.confidence) / 2
    flipped = "reject" if grader_verdict == "accept" else "accept"
    return flipped, review.confidence * (1.0 - overturn_penalty)


def build_verifier_prompt(item: Item, sub: Submission, rubric: Rubric,
                          thresholds: Thresholds, grader: GraderOutput) -> str:
    base = build_agent_prompt(item, sub, rubric, thresholds, tools_on=False)
    base = base.rsplit("\n\nGrade on this context alone", 1)[0]
    return base + f"""

## First grader's verdict (what you are reviewing)
Verdict: {grader.verdict}
Confidence: {grader.confidence}
Dimension scores: {grader.dimension_scores}
Reason: {grader.reason}

You are an adversarial verifier. Your job is to try to OVERTURN this verdict: hunt for the strongest evidence that it is wrong. You may call get_annotator_history, get_peer_judgments, and get_full_response, with a hard budget of {thresholds.verifier_max_tool_calls} evidence calls. Then call emit_review with uphold (the verdict survives your attack) or overturn (it does not). If you genuinely cannot decide, call flag_for_review with a reason — that escalates to a human. Do not rubber-stamp: uphold only what you actively failed to break."""


def run_verifier(client, con: duckdb.DuckDBPyConnection, item: Item,
                 sub: Submission, rubric: Rubric, thresholds: Thresholds,
                 swapped: bool, grader: GraderOutput) -> tuple[VerifierReview | None, AgentResult]:
    """Adversarial review loop. Returns (review, loop-result); review is None
    iff the verifier escalated. Mirrors the grader agent's loop contract."""
    executor = ToolExecutor(con, item, sub, swapped, thresholds)
    emit_tool = EMIT_REVIEW_TOOL
    tools = EVIDENCE_TOOLS + [FLAG_TOOL, emit_tool]
    tool_choice: dict = {"type": "any"}

    messages: list = [{
        "role": "user",
        "content": build_verifier_prompt(item, sub, rubric, thresholds, grader),
    }]
    res = AgentResult(output=None)
    review: VerifierReview | None = None
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
                                  "verifier produced no tool call")
            messages = messages + [{
                "role": "user",
                "content": "You must respond by calling a tool (emit_review to finish).",
            }]
            continue

        messages = messages + [{"role": "assistant", "content": resp.content}]
        results = []
        for block in tool_blocks:
            args = dict(block.input) if block.input else {}

            if block.name == "emit_review":
                try:
                    review = VerifierReview.model_validate(args)
                except ValidationError as e:
                    validation_failures += 1
                    res.trace.append({"type": "invalid_review", "error": str(e)})
                    if validation_failures >= MAX_VALIDATION_FAILURES:
                        raise GraderError(
                            item.item_id, "swapped" if swapped else "original", str(e)
                        ) from e
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Validation failed: {e}. Call emit_review again with corrected input.",
                        "is_error": True,
                    })
                    continue
                res.budget_exhausted = forced
                res.trace.append({
                    "type": "review", "decision": review.decision,
                    "confidence": review.confidence, "reason": review.reason,
                    "budget_exhausted": forced,
                })
                res.latency_ms = (time.monotonic() - t0) * 1000
                return review, res

            if block.name == "flag_for_review":
                res.escalated = True
                res.escalation_reason = str(args.get("reason", "")).strip() or "(no reason given)"
                res.trace.append({"type": "escalation", "reason": res.escalation_reason})
                res.latency_ms = (time.monotonic() - t0) * 1000
                return None, res

            if res.n_tool_calls >= thresholds.verifier_max_tool_calls:
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
        if res.n_tool_calls >= thresholds.verifier_max_tool_calls and not forced:
            forced = True
            tools = [emit_tool]
            tool_choice = {"type": "tool", "name": "emit_review"}
            content.append({
                "type": "text",
                "text": "Tool budget exhausted. You MUST now emit your review on the available evidence.",
            })
            res.trace.append({"type": "budget_exhausted"})
        messages = messages + [{"role": "user", "content": content}]
