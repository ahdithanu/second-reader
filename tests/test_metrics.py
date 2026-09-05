"""Precision/recall math on a fixture with known labels."""

import pytest

from second_reader.evals import flag_metrics
from second_reader.router import acceptance_score, route_label

# 10 items, scores hand-picked so every count below is checkable by eye.
# Flag decision: score <= LOW. Review band: LOW < score < HIGH.
FIXTURE = [
    {"score": 0.05, "is_defective": True,  "defect_type": "RUSHED"},            # flagged TP
    {"score": 0.10, "is_defective": True,  "defect_type": "RUSHED"},            # flagged TP
    {"score": 0.15, "is_defective": True,  "defect_type": "BOILERPLATE"},       # flagged TP
    {"score": 0.20, "is_defective": False, "defect_type": None},                # flagged FP
    {"score": 0.50, "is_defective": True,  "defect_type": "POSITION_BIAS"},     # review (missed)
    {"score": 0.55, "is_defective": True,  "defect_type": "SELF_CONTRADICTION"},# review (missed)
    {"score": 0.60, "is_defective": False, "defect_type": None},                # review
    {"score": 0.85, "is_defective": False, "defect_type": None},                # accepted
    {"score": 0.90, "is_defective": True,  "defect_type": "POSITION_BIAS"},     # accepted (missed)
    {"score": 0.95, "is_defective": False, "defect_type": None},                # accepted
]


class TestFlagMetrics:
    def test_known_precision_recall(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.25)
        # flagged = 4 items (scores <= 0.25), 3 truly defective
        assert m["n_flagged"] == 4
        assert m["precision"] == pytest.approx(3 / 4)
        # 6 true defects, 3 caught
        assert m["recall"] == pytest.approx(3 / 6)
        # review band (0.25, 0.75): scores 0.50, 0.55, 0.60
        assert m["review_load"] == pytest.approx(3 / 10)

    def test_recall_by_type_not_averaged_away(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.25)
        by = m["recall_by_type"]
        assert by["RUSHED"] == pytest.approx(1.0)          # 2/2
        assert by["BOILERPLATE"] == pytest.approx(1.0)     # 1/1
        assert by["POSITION_BIAS"] == pytest.approx(0.0)   # 0/2
        assert by["SELF_CONTRADICTION"] == pytest.approx(0.0)  # 0/1

    def test_no_flags_gives_none_precision(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.01)
        assert m["n_flagged"] == 0
        assert m["precision"] is None
        assert m["recall"] == pytest.approx(0.0)

    def test_flag_everything(self):
        m = flag_metrics(FIXTURE, high=0.99, low=0.98)
        assert m["n_flagged"] == 10
        assert m["precision"] == pytest.approx(6 / 10)
        assert m["recall"] == pytest.approx(1.0)


class TestAcceptanceScore:
    def test_confident_accept_scores_high(self):
        assert acceptance_score("accept", 0.9, False, 0.6) == pytest.approx(0.9)

    def test_confident_reject_scores_low(self):
        assert acceptance_score("reject", 0.9, False, 0.6) == pytest.approx(0.1)

    def test_flip_pulls_toward_review_band(self):
        base = acceptance_score("accept", 0.9, False, 0.6)
        flipped = acceptance_score("accept", 0.9, True, 0.6)
        assert flipped == pytest.approx(0.5 + (base - 0.5) * 0.4)
        assert 0.5 < flipped < base

    def test_flip_is_symmetric(self):
        hi = acceptance_score("accept", 0.9, True, 0.6)
        lo = acceptance_score("reject", 0.9, True, 0.6)
        assert hi - 0.5 == pytest.approx(0.5 - lo)


class TestRouteLabel:
    def test_tiers(self):
        assert route_label(0.9, 0.75, 0.35) == "auto_accept"
        assert route_label(0.75, 0.75, 0.35) == "auto_accept"   # inclusive
        assert route_label(0.2, 0.75, 0.35) == "auto_reject"
        assert route_label(0.35, 0.75, 0.35) == "auto_reject"   # inclusive
        assert route_label(0.5, 0.75, 0.35) == "human_review"
