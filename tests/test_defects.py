"""Annotator model + defect injection correctness at both scopes."""

import random

from conftest import make_item, make_submission
from second_reader.annotators import (
    plan_assignment,
    plan_defective_annotators,
)
from second_reader.defects import (
    BOILERPLATE_TEMPLATES,
    ITEM_DEFECT_TYPES,
    make_item_defective,
    plan_item_injection,
    swap_letters,
    swapped_view,
)

ITEM_IDS = [f"item{i:03d}" for i in range(150)]


class TestAnnotatorAssignment:
    def test_deterministic(self):
        assert plan_assignment(ITEM_IDS) == plan_assignment(ITEM_IDS)
        assert plan_assignment(list(reversed(ITEM_IDS))) == plan_assignment(ITEM_IDS)

    def test_every_item_assigned_once(self):
        plan = plan_assignment(ITEM_IDS)
        assert sorted(plan) == sorted(ITEM_IDS)

    def test_each_annotator_gets_8_to_12(self):
        plan = plan_assignment(ITEM_IDS)
        counts: dict[str, int] = {}
        for aid in plan.values():
            counts[aid] = counts.get(aid, 0) + 1
        assert len(counts) == 15
        assert all(8 <= c <= 12 for c in counts.values()), counts

    def test_defective_annotators_two_of_each(self):
        bad = plan_defective_annotators()
        assert len(bad) == 4
        types = sorted(bad.values())
        assert types == ["BOILERPLATE", "BOILERPLATE", "POSITION_BIAS", "POSITION_BIAS"]
        assert plan_defective_annotators() == bad  # deterministic


class TestItemLevelInjection:
    def test_plan_deterministic_and_25pct(self):
        clean = ITEM_IDS[:110]
        plan = plan_item_injection(clean)
        assert plan == plan_item_injection(clean)
        assert len(plan) == int(110 * 0.25)
        assert set(plan.values()) <= set(ITEM_DEFECT_TYPES)

    def test_rushed_under_10_words_vote_unchanged(self):
        rng = random.Random(0)
        for _ in range(20):
            bad = make_item_defective(make_submission(vote="B"), "RUSHED", rng)
            assert len(bad.justification.split()) < 10
            assert bad.is_defective and bad.defect_type == "RUSHED"
            assert bad.vote == "B"

    def test_self_contradiction_argues_for_other_response(self):
        rng = random.Random(0)
        bad = make_item_defective(make_submission(vote="B"), "SELF_CONTRADICTION", rng)
        assert bad.vote == "B"  # vote unchanged...
        # ...but the justification now argues for Response A.
        assert "I preferred Response A" in bad.justification
        assert "Response B misses" in bad.justification

    def test_boilerplate_templates_are_letter_free(self):
        # Templates must survive the order swap unchanged, or the swapped run
        # would see a different string and the verbatim-reuse signal would break.
        for t in BOILERPLATE_TEMPLATES:
            assert "Response A" not in t and "Response B" not in t


class TestSwap:
    def test_swap_letters_roundtrip(self):
        text = make_submission().justification
        assert swap_letters(swap_letters(text)) == text

    def test_swapped_view_is_semantically_identical(self):
        item = make_item(gold="B")
        sub = make_submission(vote="B")
        item2, sub2 = swapped_view(item, sub)
        assert item2.response_a == item.response_b
        assert item2.response_b == item.response_a
        assert item2.human_gold_vote == "A"
        assert sub2.vote == "A"
        chosen_before = item.response_b if sub.vote == "B" else item.response_a
        chosen_after = item2.response_a if sub2.vote == "A" else item2.response_b
        assert chosen_before == chosen_after
