import duckdb
import pytest

from second_reader.db import SCHEMA
from second_reader.schemas import Item, Submission


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA)
    return c


def make_item(item_id="abc123", gold="A") -> Item:
    return Item(
        item_id=item_id, question_id=1, judge="expert_0", turn=1,
        model_a="m1", model_b="m2", question="Q?",
        response_a="first answer " * 30, response_b="second answer " * 30,
        human_gold_vote=gold,
    )


def make_submission(item_id="abc123", annotator_id="ann_00", vote="A") -> Submission:
    other = "B" if vote == "A" else "A"
    return Submission(
        item_id=item_id, annotator_id=annotator_id, vote=vote,
        justification=(
            f"I preferred Response {vote} because it answers the question directly "
            f"and is more accurate. Response {other} misses part of the prompt."
        ),
    )
