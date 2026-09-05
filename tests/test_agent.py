"""Agent loop contract: termination, budget cap, validation retry, traces."""

from dataclasses import dataclass, field
from itertools import count

import pytest

from conftest import make_item, make_submission
from second_reader.config import load_rubric, load_thresholds
from second_reader.agent import run_grader
from second_reader.grader import GraderError

RUBRIC = load_rubric()
THRESHOLDS = load_thresholds()

VALID_VERDICT = {
    "dimension_scores": {name: 4 for name in RUBRIC.dimension_names},
    "verdict": "accept",
    "confidence": 0.8,
    "reason": "Looks fine.",
}

_ids = count(1)


@dataclass
class ToolUse:
    name: str
    input: dict
    type: str = "tool_use"
    id: str = field(default_factory=lambda: f"toolu_{next(_ids)}")


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 20


@dataclass
class Resp:
    content: list
    usage: Usage = field(default_factory=Usage)


class ScriptedClient:
    """Plays back a fixed sequence of tool_use blocks, one per API round."""

    def __init__(self, script: list[ToolUse]):
        self.script = list(script)
        self.rounds = 0
        self.messages = self

    def create(self, model, max_tokens, messages, tools=None, tool_choice=None):
        self.rounds += 1
        if tool_choice and tool_choice.get("type") == "tool":
            # Forced emit: comply, like the API guarantees.
            return Resp(content=[ToolUse("emit_verdict", dict(VALID_VERDICT))])
        return Resp(content=[self.script.pop(0)])


def seed_db(con):
    item, sub = make_item(), make_submission()
    con.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?)", [
        item.item_id, item.question_id, item.judge, item.turn, item.model_a,
        item.model_b, item.question, item.response_a, item.response_b,
        item.human_gold_vote,
    ])
    con.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?)", [
        sub.item_id, sub.annotator_id, sub.vote, sub.justification, False, None,
    ])
    con.execute("INSERT INTO peer_votes VALUES (?,?,?)", [item.item_id, "expert_9", "A"])
    return item, sub


class TestTermination:
    def test_immediate_verdict(self, con):
        item, sub = seed_db(con)
        client = ScriptedClient([ToolUse("emit_verdict", dict(VALID_VERDICT))])
        res = run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)
        assert res.output is not None and res.output.verdict == "accept"
        assert res.n_tool_calls == 0 and not res.escalated and not res.budget_exhausted

    def test_flag_for_review_terminates(self, con):
        item, sub = seed_db(con)
        client = ScriptedClient([
            ToolUse("get_peer_judgments", {"item_id": item.item_id}),
            ToolUse("flag_for_review", {"reason": "cannot decide"}),
        ])
        res = run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)
        assert res.escalated and res.output is None
        assert res.escalation_reason == "cannot decide"
        assert res.n_tool_calls == 1

    def test_evidence_then_verdict_traced_in_order(self, con):
        item, sub = seed_db(con)
        client = ScriptedClient([
            ToolUse("get_annotator_history", {"annotator_id": sub.annotator_id}),
            ToolUse("get_full_response", {"item_id": item.item_id, "which": "a"}),
            ToolUse("emit_verdict", dict(VALID_VERDICT)),
        ])
        res = run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)
        assert res.output is not None
        names = [e["name"] for e in res.trace if e["type"] == "tool_call"]
        assert names == ["get_annotator_history", "get_full_response"]
        assert res.trace[-1]["type"] == "verdict"
        # Every API round carries a token/latency record.
        assert len(res.rounds) == 3
        assert all("tokens_in" in r and "latency_ms" in r for r in res.rounds)


class TestBudgetCap:
    def test_cap_forces_verdict_with_flag(self, con):
        item, sub = seed_db(con)
        overreader = [
            ToolUse("get_full_response", {"item_id": item.item_id, "which": "a"})
            for _ in range(THRESHOLDS.max_tool_calls)
        ]
        client = ScriptedClient(overreader)  # forced emit handles the rest
        res = run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)
        assert res.output is not None
        assert res.budget_exhausted is True
        assert res.n_tool_calls == THRESHOLDS.max_tool_calls
        assert any(e["type"] == "budget_exhausted" for e in res.trace)


class TestValidationRetry:
    def test_invalid_verdict_retried_then_accepted(self, con):
        item, sub = seed_db(con)
        bad = {**VALID_VERDICT, "confidence": 3.0}
        client = ScriptedClient([
            ToolUse("emit_verdict", bad),
            ToolUse("emit_verdict", dict(VALID_VERDICT)),
        ])
        res = run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)
        assert res.output is not None
        assert any(e["type"] == "invalid_verdict" for e in res.trace)

    def test_fails_loud_with_item_id_after_max(self, con):
        item, sub = seed_db(con)
        bad = {**VALID_VERDICT, "verdict": "maybe"}
        client = ScriptedClient([ToolUse("emit_verdict", dict(bad)) for _ in range(3)])
        with pytest.raises(GraderError, match=item.item_id):
            run_grader(client, con, item, sub, RUBRIC, THRESHOLDS, False, True)


class TestToolExecutorOrderAwareness:
    def test_swapped_run_flips_history_and_peers(self, con):
        item, sub = seed_db(con)
        # A second submission by the same annotator, voting A.
        con.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?)", [
            "other1", 2, "expert_0", 1, "m1", "m2", "Q2", "ra", "rb", "A",
        ])
        con.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?)", [
            "other1", sub.annotator_id, "A", "Response A is better structured.", False, None,
        ])

        captured = {}

        class SpyClient(ScriptedClient):
            def create(self, model, max_tokens, messages, tools=None, tool_choice=None):
                for m in messages:
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        for b in m["content"]:
                            if isinstance(b, dict) and b.get("type") == "tool_result":
                                captured.setdefault("results", []).append(b["content"])
                return super().create(model, max_tokens, messages, tools, tool_choice)

        from second_reader.defects import swapped_view
        item2, sub2 = swapped_view(item, sub)
        client = SpyClient([
            ToolUse("get_annotator_history", {"annotator_id": sub.annotator_id}),
            ToolUse("get_peer_judgments", {"item_id": item.item_id}),
            ToolUse("emit_verdict", dict(VALID_VERDICT)),
        ])
        res = run_grader(client, con, item2, sub2, RUBRIC, THRESHOLDS, True, True)
        assert res.output is not None
        history, peers = captured["results"][0], captured["results"][1]
        # other1 was voted A with an 'A' justification; swapped view shows B.
        assert '"vote": "B"' in history
        assert "Response B is better structured." in history
        # Peer expert_9 voted A; swapped view shows B.
        assert '"vote": "B"' in peers
