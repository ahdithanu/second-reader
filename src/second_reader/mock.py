"""Deterministic mock Anthropic client for plumbing verification ONLY.

Speaks the tool-use protocol: emits tool_use blocks, reads tool_result blocks
back, respects tool_choice forcing. Lets the whole agent pipeline run
end-to-end with zero API calls. Its heuristics are intentionally crude;
numbers produced under --mock are NEVER publishable metrics. Call-log rows
are marked mock=true.

Mock agent policy (tools on): pull annotator history first; reject on
cross-item patterns (verbatim reuse, constant-side voting). Otherwise check
the item itself: rushed -> reject, contradiction -> reject, peer-majority
disagreement -> escalate or low-confidence reject, else read both full
responses and accept. A hash-picked ~5% of runs over-read until they hit the
tool budget, to exercise the forced-verdict path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from itertools import count

_ids = count(1)


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"
    id: str = field(default_factory=lambda: f"toolu_mock_{next(_ids)}")


@dataclass
class _Response:
    content: list
    usage: _Usage
    stop_reason: str = "tool_use"


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


_ANNOT_TEMPLATES = [
    "I went with Response {v} because it answers the question more directly and its structure is easier to follow. Response {o} wanders a bit and misses part of what was asked.",
    "Response {v} felt more complete to me — it covers the key points with concrete detail, while Response {o} stays vague in places. That completeness is what decided my vote.",
    "I picked Response {v} since it is more accurate on the specifics and better organized. Response {o} has a couple of claims that don't hold up on a closer read.",
    "Response {v} does a better job of actually addressing the prompt, and its examples are more useful. Response {o} is readable but shallower, so I voted for Response {v}.",
]

_DIMS_ACCEPT = {"decision_plausibility": 4, "justification_alignment": 5, "specificity": 4, "effort": 4}
_DIMS_REJECT = {"decision_plausibility": 2, "justification_alignment": 1, "specificity": 1, "effort": 1}


class MockMessages:
    def create(self, model: str, max_tokens: int, messages: list,
               tools: list | None = None, tool_choice: dict | None = None) -> _Response:
        prompt = messages[0]["content"]
        if isinstance(prompt, str) and "simulating a human annotator" in prompt:
            text = self._annotate(prompt)
            return _Response(
                content=[_TextBlock(text=text)],
                usage=_Usage(len(str(prompt)) // 4, len(text) // 4),
                stop_reason="end_turn",
            )
        if isinstance(prompt, str) and "adversarial verifier" in prompt:
            block = self._verify(messages, tool_choice or {})
        else:
            block = self._grade(messages, tools or [], tool_choice or {})
        est_in = sum(len(str(m)) for m in messages) // 4
        return _Response(content=[block], usage=_Usage(est_in, 80))

    # --- annotator simulation -------------------------------------------
    def _annotate(self, prompt: str) -> str:
        m = re.search(r"decided that Response (A|B) is better", prompt)
        v = m.group(1) if m else "A"
        o = "B" if v == "A" else "A"
        tpl = _ANNOT_TEMPLATES[_h(prompt) % len(_ANNOT_TEMPLATES)]
        return tpl.format(v=v, o=o)

    # --- grader agent simulation ----------------------------------------
    def _grade(self, messages: list, tools: list, tool_choice: dict) -> _ToolUseBlock:
        prompt = messages[0]["content"]
        vote = self._m(r"Vote: Response (A|B) is better", prompt)
        annotator_id = self._m(r"Annotator: (ann_\d+)", prompt) or "ann_00"
        item_id = self._m(r"Item: (\w+)", prompt) or ""
        just = self._between(prompt, "Justification: ", "\n\n## Rubric")
        seed = _h(prompt)

        called = self._called_tools(messages)
        results = self._tool_results(messages)
        forced = tool_choice.get("type") == "tool" or all(
            t["name"] == "emit_verdict" for t in tools
        )
        tools_on = any(t["name"] == "get_annotator_history" for t in tools)

        # Tools-off (or forced): verdict from the visible context alone.
        if forced or not tools_on:
            return self._verdict_from_context(vote, just, seed, results)

        # ~5% of runs over-read until the budget forces a verdict.
        if seed % 19 == 0:
            which = "a" if called.count("get_full_response") % 2 == 0 else "b"
            return _ToolUseBlock("get_full_response", {"item_id": item_id, "which": which})

        if "get_annotator_history" not in called:
            return _ToolUseBlock("get_annotator_history",
                                 {"annotator_id": annotator_id, "limit": 10})

        history = self._latest_json(results, "submissions")
        if history is not None:
            subs = history.get("submissions", [])
            justs = [s["justification"] for s in subs]
            votes = [s["vote"] for s in subs]
            if len(justs) >= 5 and len(set(justs)) == 1:
                return self._emit("reject", 0.92,
                                  "Annotator history shows the same justification reused verbatim across items.",
                                  _DIMS_REJECT)
            if len(votes) >= 6 and len(set(votes)) == 1:
                return self._emit("reject", 0.85,
                                  "Annotator history shows a constant-side vote on every item regardless of content.",
                                  _DIMS_REJECT)

        if len(just.split()) < 10:
            return self._emit("reject", 0.88,
                              "Justification is a one-line dismissal with no evidence of reading the responses.",
                              {"decision_plausibility": 2, "justification_alignment": 2, "specificity": 0, "effort": 0})

        other = "B" if vote == "A" else "A"
        if just.count(f"Response {other}") > just.count(f"Response {vote}"):
            return self._emit("reject", 0.84,
                              "The justification argues for the response the annotator voted against.",
                              {"decision_plausibility": 2, "justification_alignment": 0, "specificity": 3, "effort": 3})

        if "get_peer_judgments" not in called:
            return _ToolUseBlock("get_peer_judgments", {"item_id": item_id})

        peers = self._latest_json(results, "peer_votes")
        if peers is not None:
            pv = [p["vote"] for p in peers.get("peer_votes", []) if p["vote"] != "tie"]
            if len(pv) >= 2 and pv.count(vote) == 0:
                if seed % 2 == 0:
                    return _ToolUseBlock("flag_for_review", {
                        "reason": "Every peer judge voted the other way; I cannot tell whether the annotator misread the responses or has a defensible minority view.",
                    })
                return self._emit("reject", 0.6,
                                  "Vote contradicts every peer judgment on this pair.",
                                  {"decision_plausibility": 1, "justification_alignment": 4, "specificity": 3, "effort": 3})

        if called.count("get_full_response") < 2:
            which = "a" if called.count("get_full_response") == 0 else "b"
            return _ToolUseBlock("get_full_response", {"item_id": item_id, "which": which})

        noise = (seed % 100) / 100
        return self._emit("accept", round(0.6 + 0.35 * noise, 2),
                          "Vote is defensible against the full responses and the justification is specific and aligned.",
                          _DIMS_ACCEPT)

    def _verify(self, messages: list, tool_choice: dict) -> _ToolUseBlock:
        """Adversarial verifier simulation: pull history once, then attack."""
        prompt = messages[0]["content"]
        vote = self._m(r"Vote: Response (A|B) is better", prompt)
        annotator_id = self._m(r"Annotator: (ann_\d+)", prompt) or "ann_00"
        just = self._between(prompt, "Justification: ", "\n\n## Rubric")
        grader_verdict = self._m(r"Verdict: (accept|reject)", prompt) or "accept"
        seed = _h(prompt)
        forced = tool_choice.get("type") == "tool"

        called = self._called_tools(messages)
        if not forced and "get_annotator_history" not in called:
            return _ToolUseBlock("get_annotator_history",
                                 {"annotator_id": annotator_id, "limit": 10})

        def review(decision, confidence, reason):
            return _ToolUseBlock("emit_review", {
                "decision": decision, "confidence": confidence, "reason": reason,
            })

        if grader_verdict == "reject":
            return review("uphold", 0.9, "The reject verdict survives scrutiny.")

        history = self._latest_json(self._tool_results(messages), "submissions")
        if history is not None:
            subs = history.get("submissions", [])
            justs = [s["justification"] for s in subs]
            votes = [s["vote"] for s in subs]
            if len(justs) >= 5 and len(set(justs)) == 1:
                return review("overturn", 0.9,
                              "History shows verbatim justification reuse the grader missed.")
            if len(votes) >= 6 and len(set(votes)) == 1:
                return review("overturn", 0.85,
                              "History shows constant-side voting the grader missed.")
        other = "B" if vote == "A" else "A"
        if just.count(f"Response {other}") > just.count(f"Response {vote}"):
            return review("overturn", 0.85,
                          "The justification argues against the vote; the accept cannot stand.")
        if seed % 29 == 0:
            return _ToolUseBlock("flag_for_review",
                                 {"reason": "Cannot confirm or break this verdict on the available evidence."})
        noise = (seed % 100) / 100
        return review("uphold", round(0.55 + 0.4 * noise, 2),
                      "Tried and failed to break the accept verdict.")

    def _verdict_from_context(self, vote, just, seed, results) -> _ToolUseBlock:
        if len(just.split()) < 10:
            return self._emit("reject", 0.9,
                              "Justification is a one-line dismissal.",
                              {"decision_plausibility": 2, "justification_alignment": 2, "specificity": 0, "effort": 0})
        other = "B" if vote == "A" else "A"
        if just.count(f"Response {other}") > just.count(f"Response {vote}"):
            return self._emit("reject", 0.8,
                              "The justification argues for the other response.",
                              {"decision_plausibility": 2, "justification_alignment": 0, "specificity": 3, "effort": 3})
        noise = (seed % 100) / 100
        return self._emit("accept", round(0.55 + 0.4 * noise, 2),
                          "Nothing suspicious visible in the available context.",
                          _DIMS_ACCEPT)

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _emit(verdict, confidence, reason, dims) -> _ToolUseBlock:
        return _ToolUseBlock("emit_verdict", {
            "dimension_scores": dims, "verdict": verdict,
            "confidence": confidence, "reason": reason,
        })

    @staticmethod
    def _m(pattern: str, text: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1) if m else None

    @staticmethod
    def _between(text: str, start: str, end: str) -> str:
        i = text.find(start)
        j = text.find(end, i)
        return text[i + len(start):j].strip() if i != -1 and j != -1 else ""

    @staticmethod
    def _called_tools(messages: list) -> list[str]:
        names = []
        for m in messages:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if getattr(b, "type", None) == "tool_use":
                        names.append(b.name)
        return names

    @staticmethod
    def _tool_results(messages: list) -> list[str]:
        out = []
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        out.append(str(b.get("content", "")))
        return out

    @staticmethod
    def _latest_json(results: list[str], must_have_key: str) -> dict | None:
        for text in reversed(results):
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and must_have_key in data:
                return data
        return None


class MockClient:
    def __init__(self) -> None:
        self.messages = MockMessages()
