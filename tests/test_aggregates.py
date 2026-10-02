"""Layer-0 aggregate signals: thresholds, flag logic, record overlay."""

import pytest

from second_reader.aggregates import annotator_stats, apply_to_records, flagged_annotators
from second_reader.config import load_thresholds
from second_reader.evals import flag_metrics

THRESHOLDS = load_thresholds()


def seed(con, annotator_id: str, subs: list[tuple[str, str]]):
    for i, (vote, justification) in enumerate(subs):
        con.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?)", [
            f"{annotator_id}_it{i:02d}", annotator_id, vote, justification, False, None,
        ])


class TestSignals:
    def test_constant_side_flagged(self, con):
        seed(con, "ann_pb", [("A", f"unique justification number {i}") for i in range(9)])
        stats = {s.annotator_id: s for s in annotator_stats(con, THRESHOLDS)}
        s = stats["ann_pb"]
        assert s.side_ratio == 1.0 and s.flagged and s.reason == "constant_side"

    def test_justification_reuse_flagged(self, con):
        votes = ["A", "B"] * 5
        seed(con, "ann_bp", [(votes[i], "Same text every time.") for i in range(10)])
        flags = flagged_annotators(con, THRESHOLDS)
        assert flags["ann_bp"] == "justification_reuse"

    def test_reuse_matches_after_whitespace_normalization(self, con):
        seed(con, "ann_ws", [("A", "Same  text."), ("B", "same text. "),
                             ("A", "SAME TEXT."), ("B", "x"), ("A", "y"),
                             ("B", "z"), ("A", "w"), ("B", "v")])
        stats = {s.annotator_id: s for s in annotator_stats(con, THRESHOLDS)}
        assert stats["ann_ws"].dup_ratio == pytest.approx(3 / 8)

    def test_clean_annotator_not_flagged(self, con):
        votes = ["A", "B", "A", "B", "A", "A", "B", "A", "B", "B"]
        seed(con, "ann_ok", [(votes[i], f"text {i}") for i in range(10)])
        assert flagged_annotators(con, THRESHOLDS) == {}

    def test_below_min_items_never_flagged(self, con):
        seed(con, "ann_small", [("A", "dup") for _ in range(THRESHOLDS.agg_min_items - 1)])
        assert flagged_annotators(con, THRESHOLDS) == {}


class TestOverlay:
    RECORDS = [
        {"item_id": "i1", "score": 0.9, "escalated": False, "is_defective": True,
         "defect_type": "POSITION_BIAS", "annotator_id": "bad"},
        {"item_id": "i2", "score": None, "escalated": True, "is_defective": True,
         "defect_type": "POSITION_BIAS", "annotator_id": "bad"},
        {"item_id": "i3", "score": 0.9, "escalated": False, "is_defective": False,
         "defect_type": None, "annotator_id": "ok"},
    ]

    def test_flagged_items_become_flags(self):
        out = apply_to_records(self.RECORDS, {"bad": "constant_side"})
        m = flag_metrics(out, high=0.75, low=0.35)
        # Both of 'bad''s items now flagged (incl. the previously escalated one);
        # the clean annotator's accept is untouched.
        assert m["n_flagged"] == 2
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["n_escalated"] == 0

    def test_no_flags_is_identity_on_metrics(self):
        out = apply_to_records(self.RECORDS, {})
        base = flag_metrics(self.RECORDS, 0.75, 0.35)
        overlaid = flag_metrics(out, 0.75, 0.35)
        assert base == overlaid
