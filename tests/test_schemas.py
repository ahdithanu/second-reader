"""Grader output schema validation."""

import pytest
from pydantic import ValidationError

from second_reader.config import load_rubric
from second_reader.schemas import GraderOutput

RUBRIC = load_rubric()

VALID = {
    "dimension_scores": {name: 4 for name in RUBRIC.dimension_names},
    "verdict": "accept",
    "confidence": 0.8,
    "reason": "Vote is defensible and the justification is aligned.",
}


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
        out = GraderOutput.model_validate({**VALID, "dimension_scores": {"effort": 3}})
        with pytest.raises(ValueError, match="dimension mismatch"):
            out.check_dimensions(RUBRIC.dimension_names)
