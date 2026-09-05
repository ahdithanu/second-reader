"""Deterministic mock Anthropic client for plumbing verification ONLY.

Lets the whole pipeline (data -> annotate -> inject -> grade x2 -> route ->
sweep) run end-to-end with zero API calls. Its heuristics are intentionally
crude; numbers produced under --mock are NEVER publishable metrics. The call
log marks these rows mock=true.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list
    usage: _Usage


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


_ANNOT_TEMPLATES = [
    "I went with Response {v} because it answers the question more directly and its structure is easier to follow. Response {o} wanders a bit and misses part of what was asked.",
    "Response {v} felt more complete to me — it covers the key points with concrete detail, while Response {o} stays vague in places. That completeness is what decided my vote.",
    "I picked Response {v} since it is more accurate on the specifics and better organized. Response {o} has a couple of claims that don't hold up on a closer read.",
    "Response {v} does a better job of actually addressing the prompt, and its examples are more useful. Response {o} is readable but shallower, so I voted for Response {v}.",
]


class MockMessages:
    def create(self, model: str, max_tokens: int, messages: list) -> _Response:
        prompt = messages[0]["content"]
        if "simulating a careful human annotator" in prompt:
            text = self._annotate(prompt)
        else:
            text = self._grade(prompt)
        return _Response(
            content=[_Block(text=text)],
            usage=_Usage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4),
        )

    def _annotate(self, prompt: str) -> str:
        m = re.search(r"decided that Response (A|B) is better", prompt)
        v = m.group(1) if m else "A"
        o = "B" if v == "A" else "A"
        tpl = _ANNOT_TEMPLATES[_h(prompt) % len(_ANNOT_TEMPLATES)]
        return tpl.format(v=v, o=o)

    def _grade(self, prompt: str) -> str:
        vote_m = re.search(r"Vote: Response (A|B) is better", prompt)
        just_m = re.search(r"Justification: (.*?)\n\n## Rubric", prompt, re.DOTALL)
        vote = vote_m.group(1) if vote_m else "A"
        just = just_m.group(1).strip() if just_m else ""
        other = "B" if vote == "A" else "A"

        n_words = len(just.split())
        mentions_vote = just.count(f"Response {vote}")
        mentions_other = just.count(f"Response {other}")
        generic = "Response A" not in just and "Response B" not in just

        if n_words < 10:
            verdict, conf, reason = "reject", 0.9, "Justification is a one-line dismissal with no evidence of reading the responses."
            dims = {"decision_plausibility": 2, "justification_alignment": 2, "specificity": 0, "effort": 0}
        elif generic and n_words < 30:
            verdict, conf, reason = "reject", 0.8, "Generic template praise that never references either response specifically."
            dims = {"decision_plausibility": 2, "justification_alignment": 2, "specificity": 1, "effort": 1}
        elif mentions_other > mentions_vote:
            verdict, conf, reason = "reject", 0.75, "The justification argues for the response the annotator voted against."
            dims = {"decision_plausibility": 2, "justification_alignment": 0, "specificity": 3, "effort": 3}
        else:
            noise = (_h(just or prompt) % 100) / 100
            verdict, conf = "accept", round(0.6 + 0.35 * noise, 2)
            reason = "Vote is defensible and the justification is specific and aligned with it."
            dims = {"decision_plausibility": 4, "justification_alignment": 5, "specificity": 4, "effort": 4}

        payload = json.dumps({
            "dimension_scores": dims, "verdict": verdict,
            "confidence": conf, "reason": reason,
        })
        # Occasionally wrap in a code fence to exercise the extractor.
        if _h(prompt) % 7 == 0:
            return f"```json\n{payload}\n```"
        return payload


class MockClient:
    def __init__(self) -> None:
        self.messages = MockMessages()
