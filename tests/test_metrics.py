"""Precision/recall math on a fixture with known labels, escalation included."""

import pytest

from second_reader.evals import flag_metrics
from second_reader.router import acceptance_score, route_label

# 12 items, scores hand-picked so every count below is checkable by eye.
# Flag decision: not escalated and score <= LOW. Review: escalated or in band.
FIXTURE = [
    {"score": 0.05, "escalated": False, "is_defective": True,  "defect_type": "RUSHED"},             # flagged TP
    {"score": 0.10, "escalated": False, "is_defective": True,  "defect_type": "RUSHED"},             # flagged TP
    {"score": 0.15, "escalated": False, "is_defective": True,  "defect_type": "BOILERPLATE"},        # flagged TP
    {"score": 0.20, "escalated": False, "is_defective": False, "defect_type": None},                 # flagged FP
    {"score": None, "escalated": True,  "is_defective": True,  "defect_type": "SELF_CONTRADICTION"}, # escalated (caught, not flagged)
    {"score": None, "escalated": True,  "is_defective": False, "defect_type": None},                 # escalated clean
    {"score": 0.50, "escalated": False, "is_defective": True,  "defect_type": "POSITION_BIAS"},      # review (missed)
    {"score": 0.55, "escalated": False, "is_defective": True,  "defect_type": "SELF_CONTRADICTION"}, # review (missed)
    {"score": 0.60, "escalated": False, "is_defective": False, "defect_type": None},                 # review
    {"score": 0.85, "escalated": False, "is_defective": False, "defect_type": None},                 # accepted
    {"score": 0.90, "escalated": False, "is_defective": True,  "defect_type": "POSITION_BIAS"},      # accepted (missed)
    {"score": 0.95, "escalated": False, "is_defective": False, "defect_type": None},                 # accepted
]


class TestFlagMetrics:
    def test_known_precision_recall(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.25)
        assert m["n_flagged"] == 4              # scores <= 0.25, not escalated
        assert m["n_escalated"] == 2
        assert m["precision"] == pytest.approx(3 / 4)
        assert m["recall"] == pytest.approx(3 / 7)       # 7 true defects, 3 flagged
        assert m["catch_rate"] == pytest.approx(4 / 7)   # + 1 escalated defect
        # review band: 2 escalated + scores 0.50, 0.55, 0.60
        assert m["review_load"] == pytest.approx(5 / 12)

    def test_recall_by_type_not_averaged_away(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.25)
        by = m["recall_by_type"]
        assert by["RUSHED"] == pytest.approx(1.0)              # 2/2
        assert by["BOILERPLATE"] == pytest.approx(1.0)         # 1/1
        assert by["POSITION_BIAS"] == pytest.approx(0.0)       # 0/2
        assert by["SELF_CONTRADICTION"] == pytest.approx(0.0)  # 0/2 flagged
        catch = m["catch_by_type"]
        assert catch["SELF_CONTRADICTION"] == pytest.approx(1 / 2)  # escalation caught one

    def test_no_flags_gives_none_precision(self):
        m = flag_metrics(FIXTURE, high=0.75, low=0.01)
        assert m["n_flagged"] == 0
        assert m["precision"] is None
        assert m["recall"] == pytest.approx(0.0)
        assert m["catch_rate"] == pytest.approx(1 / 7)  # escalated defect still caught

    def test_flag_everything(self):
        m = flag_metrics(FIXTURE, high=0.99, low=0.98)
        assert m["n_flagged"] == 10                       # all except 2 escalated
        assert m["precision"] == pytest.approx(6 / 10)
        assert m["recall"] == pytest.approx(6 / 7)
        assert m["catch_rate"] == pytest.approx(1.0)


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
