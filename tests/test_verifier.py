"""Verifier arm: combination rule, loop contract, escalation."""

import pytest

from conftest import make_item, make_submission
from second_reader.config import load_rubric, load_thresholds
from second_reader.grader import GraderError
from second_reader.schemas import GraderOutput, VerifierReview
from second_reader.verifier import combine, run_verifier
from test_agent import ScriptedClient, ToolUse, seed_db

RUBRIC = load_rubric()
THRESHOLDS = load_thresholds()

GRADER_OUT = GraderOutput(
    dimension_scores={name: 4 for name in RUBRIC.dimension_names},
    verdict="accept", confidence=0.8, reason="Looks fine.",
)


class TestCombineRule:
    def test_uphold_averages_confidence(self):
        review = VerifierReview(decision="uphold", confidence=0.6, reason="survives")
        verdict, conf = combine("accept", 0.8, review, 0.4)
        assert verdict == "accept"
        assert conf == pytest.approx(0.7)

    def test_overturn_flips_and_penalizes(self):
        review = VerifierReview(decision="overturn", confidence=0.9, reason="broken")
        verdict, conf = combine("accept", 0.8, review, 0.4)
        assert verdict == "reject"
        assert conf == pytest.approx(0.9 * 0.6)

    def test_overturn_of_reject_becomes_accept(self):
        review = VerifierReview(decision="overturn", confidence=0.5, reason="too harsh")
        verdict, conf = combine("reject", 0.9, review, 0.4)
        assert verdict == "accept"
        assert conf == pytest.approx(0.3)


class VerifierScriptedClient(ScriptedClient):
    def create(self, model, max_tokens, messages, tools=None, tool_choice=None):
        if tool_choice and tool_choice.get("type") == "tool":
            from test_agent import Resp
            return Resp(content=[ToolUse("emit_review", {
                "decision": "uphold", "confidence": 0.5, "reason": "forced",
            })])
        from test_agent import Resp
        return Resp(content=[self.script.pop(0)])


class TestVerifierLoop:
    def test_evidence_then_review(self, con):
        item, sub = seed_db(con)
        client = VerifierScriptedClient([
            ToolUse("get_annotator_history", {"annotator_id": sub.annotator_id}),
            ToolUse("emit_review", {"decision": "overturn", "confidence": 0.9,
                                    "reason": "history breaks it"}),
        ])
        review, res = run_verifier(client, con, item, sub, RUBRIC, THRESHOLDS,
                                   False, GRADER_OUT)
        assert review is not None and review.decision == "overturn"
        assert res.n_tool_calls == 1
        assert res.trace[-1]["type"] == "review"

    def test_verifier_budget_forces_review(self, con):
        item, sub = seed_db(con)
        overreader = [
            ToolUse("get_full_response", {"item_id": item.item_id, "which": "a"})
            for _ in range(THRESHOLDS.verifier_max_tool_calls)
        ]
        client = VerifierScriptedClient(overreader)
        review, res = run_verifier(client, con, item, sub, RUBRIC, THRESHOLDS,
                                   False, GRADER_OUT)
        assert review is not None
        assert res.budget_exhausted is True
        assert res.n_tool_calls == THRESHOLDS.verifier_max_tool_calls

    def test_verifier_can_escalate(self, con):
        item, sub = seed_db(con)
        client = VerifierScriptedClient([
            ToolUse("flag_for_review", {"reason": "cannot break or confirm"}),
        ])
        review, res = run_verifier(client, con, item, sub, RUBRIC, THRESHOLDS,
                                   False, GRADER_OUT)
        assert review is None and res.escalated
        assert res.escalation_reason == "cannot break or confirm"

    def test_invalid_review_fails_loud_with_item_id(self, con):
        item, sub = seed_db(con)
        bad = ToolUse("emit_review", {"decision": "maybe", "confidence": 0.5, "reason": "x"})
        client = VerifierScriptedClient([
            ToolUse("emit_review", dict(bad.input)) for _ in range(3)
        ])
        with pytest.raises(GraderError, match=item.item_id):
            run_verifier(client, con, item, sub, RUBRIC, THRESHOLDS, False, GRADER_OUT)

    def test_prompt_contains_grader_verdict(self, con):
        item, sub = seed_db(con)
        seen = {}

        class SpyClient(VerifierScriptedClient):
            def create(self, model, max_tokens, messages, tools=None, tool_choice=None):
                seen.setdefault("prompt", messages[0]["content"])
                return super().create(model, max_tokens, messages, tools, tool_choice)

        client = SpyClient([
            ToolUse("emit_review", {"decision": "uphold", "confidence": 0.7, "reason": "ok"}),
        ])
        run_verifier(client, con, item, sub, RUBRIC, THRESHOLDS, False, GRADER_OUT)
        assert "adversarial verifier" in seen["prompt"]
        assert "Verdict: accept" in seen["prompt"]
        assert "Looks fine." in seen["prompt"]
