"""Defect injection correctness: determinism, rates, and per-type properties."""

import random

from second_reader.defects import (
    BOILERPLATE_TEXT,
    DEFECT_TYPES,
    make_defective,
    plan_injection,
    swap_letters,
    swapped_view,
)
from second_reader.schemas import Item, Submission

ITEM_IDS = [f"item{i:03d}" for i in range(100)]


def clean_sub(item_id="item000", vote="B") -> Submission:
    return Submission(
        item_id=item_id, vote=vote,
        justification=(
            "I preferred Response B because it answers the question directly "
            "and is more accurate. Response A misses the second half of the prompt."
        ),
    )


class TestPlan:
    def test_deterministic(self):
        assert plan_injection(ITEM_IDS) == plan_injection(ITEM_IDS)
        assert plan_injection(list(reversed(ITEM_IDS))) == plan_injection(ITEM_IDS)

    def test_fraction(self):
        plan = plan_injection(ITEM_IDS, fraction=0.25)
        assert len(plan) == 25

    def test_all_types_represented(self):
        plan = plan_injection(ITEM_IDS)
        counts = {t: 0 for t in DEFECT_TYPES}
        for t in plan.values():
            counts[t] += 1
        # 25 items across 4 types -> 6 or 7 each
        assert all(6 <= c <= 7 for c in counts.values()), counts


class TestDefectProperties:
    def test_rushed_under_10_words(self):
        rng = random.Random(0)
        for _ in range(20):
            bad = make_defective(clean_sub(), "RUSHED", rng)
            assert len(bad.justification.split()) < 10
            assert bad.is_defective and bad.defect_type == "RUSHED"
            assert bad.vote == "B"  # vote unchanged

    def test_boilerplate_verbatim_identical(self):
        rng = random.Random(0)
        texts = {make_defective(clean_sub(item_id=i), "BOILERPLATE", rng).justification
                 for i in ITEM_IDS[:10]}
        assert texts == {BOILERPLATE_TEXT}

    def test_position_bias_always_votes_a(self):
        rng = random.Random(0)
        for vote in ("A", "B"):
            bad = make_defective(clean_sub(vote=vote), "POSITION_BIAS", rng)
            assert bad.vote == "A"
            assert bad.defect_type == "POSITION_BIAS"

    def test_self_contradiction_argues_for_other_response(self):
        rng = random.Random(0)
        bad = make_defective(clean_sub(vote="B"), "SELF_CONTRADICTION", rng)
        assert bad.vote == "B"  # vote unchanged...
        # ...but the justification now argues for Response A.
        assert "I preferred Response A" in bad.justification
        assert "Response B misses" in bad.justification


class TestSwap:
    def test_swap_letters_roundtrip(self):
        text = clean_sub().justification
        assert swap_letters(swap_letters(text)) == text
        swapped = swap_letters(text)
        assert "I preferred Response A" in swapped

    def test_swapped_view_is_semantically_identical(self):
        item = Item(
            item_id="x", question_id=1, judge="j", turn=1,
            model_a="m1", model_b="m2", question="Q?",
            response_a="first answer", response_b="second answer",
            human_gold_vote="B",
        )
        sub = clean_sub(item_id="x", vote="B")
        item2, sub2 = swapped_view(item, sub)
        assert item2.response_a == "second answer"
        assert item2.response_b == "first answer"
        assert item2.human_gold_vote == "A"
        assert sub2.vote == "A"
        # The vote still points at the same underlying response text.
        chosen_before = item.response_b if sub.vote == "B" else item.response_a
        chosen_after = item2.response_a if sub2.vote == "A" else item2.response_b
        assert chosen_before == chosen_after
