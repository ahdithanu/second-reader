"""Pydantic models for grader output and DB rows. Validation is the contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DefectType = Literal["RUSHED", "BOILERPLATE", "POSITION_BIAS", "SELF_CONTRADICTION"]


class GraderOutput(BaseModel):
    """Structured verdict from one grader run. Never valid without a reason."""

    dimension_scores: dict[str, int]
    verdict: Literal["accept", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("dimension_scores")
    @classmethod
    def scores_in_range(cls, v: dict[str, int]) -> dict[str, int]:
        for name, score in v.items():
            if not (0 <= score <= 5):
                raise ValueError(f"dimension '{name}' score {score} outside 0-5")
        return v

    @field_validator("reason")
    @classmethod
    def reason_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must be a non-empty string")
        return v.strip()

    def check_dimensions(self, expected: list[str]) -> None:
        """Raise if scored dimensions don't exactly match the rubric."""
        got, want = set(self.dimension_scores), set(expected)
        if got != want:
            missing, extra = sorted(want - got), sorted(got - want)
            raise ValueError(
                f"dimension mismatch: missing={missing} extra={extra}; "
                f"rubric requires exactly {sorted(want)}"
            )


class VerifierReview(BaseModel):
    """The adversarial verifier's terminal output: uphold or overturn."""

    decision: Literal["uphold", "overturn"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must be a non-empty string")
        return v.strip()


class Annotator(BaseModel):
    """A synthetic annotator. Defects at this level span ALL their items."""

    annotator_id: str
    is_defective: bool = False
    defect_type: Literal["BOILERPLATE", "POSITION_BIAS"] | None = None


class Submission(BaseModel):
    """An annotator submission: a vote plus a written justification."""

    item_id: str
    annotator_id: str
    vote: Literal["A", "B"]
    justification: str
    is_defective: bool = False
    defect_type: DefectType | None = None


class Item(BaseModel):
    """One sampled MT-Bench pairwise comparison with its human gold vote."""

    item_id: str
    question_id: int
    judge: str
    turn: int
    model_a: str
    model_b: str
    question: str
    response_a: str
    response_b: str
    human_gold_vote: Literal["A", "B"]
