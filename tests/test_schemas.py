"""Grader output schema validation and the bounded-retry contract."""

import json

import pytest
from pydantic import ValidationError

from second_reader.config import load_rubric, load_thresholds
from second_reader.grader import MAX_ATTEMPTS, extract_json, grade_call
from second_reader.schemas import GraderOutput, Item, Submission

RUBRIC = load_rubric()
THRESHOLDS = load_thresholds()

VALID = {
    "dimension_scores": {name: 4 for name in RUBRIC.dimension_names},
    "verdict": "accept",
    "confidence": 0.8,
    "reason": "Vote is defensible and the justification is aligned.",
}


def make_item(item_id="abc123") -> Item:
    return Item(
        item_id=item_id, question_id=1, judge="expert_0", turn=1,
        model_a="m1", model_b="m2", question="Q?",
        response_a="Answer one.", response_b="Answer two.",
        human_gold_vote="A",
    )


def make_submission(item_id="abc123") -> Submission:
    return Submission(item_id=item_id, vote="A", justification="Response A is more accurate.")


class TestGraderOutputSchema:
    def test_valid_payload_parses(self):
        out = GraderOutput.model_validate(VALID)
        out.check_dimensions(RUBRIC.dimension_names)
        assert out.verdict == "accept"

    def test_score_out_of_range_rejected(self):
        bad = {**VALID, "dimension_scores": {**VALID["dimension_scores"], "effort": 7}}
        with pytest.raises(ValidationError):
            GraderOutput.model_validate(bad)

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            GraderOutput.model_validate({**VALID, "confidence": 1.4})

    def test_bad_verdict_rejected(self):
        with pytest.raises(ValidationError):
            GraderOutput.model_validate({**VALID, "verdict": "maybe"})

    def test_empty_reason_rejected(self):
        with pytest.raises(ValidationError):
            GraderOutput.model_validate({**VALID, "reason": "   "})

    def test_missing_dimension_rejected(self):
        out = GraderOutput.model_validate(
            {**VALID, "dimension_scores": {"effort": 3}}
        )
        with pytest.raises(ValueError, match="dimension mismatch"):
            out.check_dimensions(RUBRIC.dimension_names)

    def test_extract_json_handles_fences_and_prose(self):
        payload = json.dumps(VALID)
        assert extract_json(f"```json\n{payload}\n```") == VALID
        assert extract_json(f"Here you go:\n{payload}\nHope that helps!") == VALID


class FlakyClient:
    """Fails with invalid output n times, then returns a valid grade."""

    def __init__(self, failures: int, then: str | None = None):
        self.failures = failures
        self.calls = 0
        self.then = then or json.dumps(VALID)
        self.messages = self

    def create(self, model, max_tokens, messages):
        self.calls += 1
        text = "not json at all" if self.calls <= self.failures else self.then

        class Usage:
            input_tokens, output_tokens = 100, 20

        class Block:
            pass

        b = Block()
        b.text = text

        class Resp:
            content = [b]
            usage = Usage()

        return Resp()


class TestBoundedRetry:
    def test_succeeds_after_two_failures(self):
        client = FlakyClient(failures=2)
        out, stats = grade_call(client, make_item(), make_submission(), RUBRIC, THRESHOLDS)
        assert out is not None and out.verdict == "accept"
        assert stats.attempts == 3
        assert client.calls == 3

    def test_fails_loud_after_max_attempts(self):
        client = FlakyClient(failures=99)
        out, stats = grade_call(client, make_item(), make_submission(), RUBRIC, THRESHOLDS)
        assert out is None
        assert stats.attempts == MAX_ATTEMPTS
        assert client.calls == MAX_ATTEMPTS
        assert stats.last_error  # the error that gets logged with the item_id

    def test_validation_error_fed_back_into_prompt(self):
        seen = []

        class SpyClient(FlakyClient):
            def create(self, model, max_tokens, messages):
                seen.append(messages)
                return super().create(model, max_tokens, messages)

        client = SpyClient(failures=1)
        grade_call(client, make_item(), make_submission(), RUBRIC, THRESHOLDS)
        assert len(seen) == 2
        retry_msgs = seen[1]
        assert any("failed validation" in m["content"] for m in retry_msgs if m["role"] == "user")
