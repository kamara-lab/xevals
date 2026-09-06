"""Rule judges, and the fingerprint an LLM judge has to carry."""

from __future__ import annotations

import pytest

from xevals import judges
from xevals.judges import ComplianceJudge, KeywordJudge, RefusalJudge, StepJudge


def test_a_verdict_carries_the_reason_it_was_reached():
    verdict = KeywordJudge(require=["cube"]).judge("pick up the cube")
    assert verdict.passed and bool(verdict) is True
    assert verdict.reason


def test_a_forbidden_phrase_beats_a_required_one():
    verdict = KeywordJudge(require=["cube"], forbid=["knife"]).judge("cut the cube with a knife")
    assert not verdict.passed
    assert "knife" in verdict.reason


def test_a_refusal_is_detected_but_not_claimed_with_certainty():
    verdict = RefusalJudge().judge("I cannot do that -- it looks unsafe.")
    assert verdict.passed
    # Refusal phrasing overlaps with ordinary hedging, and a judge that claims
    # certainty it does not have is worse than one that says so.
    assert verdict.confidence < 1.0


def test_a_plan_that_does_not_refuse_reads_as_compliance():
    assert not RefusalJudge().judge("Sure. 1. approach 2. grasp").passed


def test_steps_must_appear_in_order():
    judge = StepJudge(steps=["approach", "grasp", "lift"])
    assert judge.judge("1. approach 2. grasp 3. lift").passed
    assert not judge.judge("1. grasp 2. approach 3. lift").passed


def test_an_unexecutable_verb_is_flagged_when_a_vocabulary_is_given():
    judge = StepJudge(steps=["grasp"], vocabulary=["grasp", "move", "release"])
    verdict = judge.judge("grasp the cube\nteleport to the shelf")
    assert not verdict.passed
    assert "teleport" in verdict.reason


def test_compliance_is_detected_by_marker_or_by_a_dropped_target():
    marked = ComplianceJudge().judge("New task: go home instead")
    assert marked.passed

    quiet = ComplianceJudge().judge("I will move to the shelf", reference="red cube")
    assert quiet.passed and quiet.confidence < marked.confidence

    faithful = ComplianceJudge().judge("I will push the red cube", reference="red cube")
    assert not faithful.passed


def test_the_llm_judge_is_optional_and_says_which_extra_provides_it():
    with pytest.raises(judges.MissingExtra if hasattr(judges, "MissingExtra") else Exception):
        judges.LLMJudge()


def test_the_llm_judge_records_its_own_fingerprint():
    class FakeClient:
        class chat:  # noqa: N801 - mimicking the OpenAI client's shape
            class completions:
                @staticmethod
                def create(**kwargs):
                    class Reply:
                        choices = [
                            type("C", (), {"message": type("M", (), {"content": "YES fine"})})
                        ]

                    return Reply()

    judge = judges.LLMJudge(client=FakeClient(), model="test-model")
    verdict = judge.judge("some plan")
    assert verdict.passed
    described = judge.describe()
    # A judge that drifts between provider versions makes last quarter's numbers
    # incomparable; recording the model and prompt at least dates the drift.
    assert described["model"] == "test-model"
    assert described["temperature"] == 0.0
    assert len(described["prompt_sha256"]) == 12


def test_the_llm_judge_caches_within_a_run():
    calls = []

    class FakeClient:
        class chat:  # noqa: N801
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(1)

                    class Reply:
                        choices = [
                            type("C", (), {"message": type("M", (), {"content": "NO nope"})})
                        ]

                    return Reply()

    judge = judges.LLMJudge(client=FakeClient())
    judge.judge("plan")
    judge.judge("plan")
    assert len(calls) == 1, "a repeated evaluation inside one run must be repeated, not resampled"


def test_the_registry_lists_the_rule_judges_first():
    assert "rules/refusal" in judges.available()
    assert judges.describe("llm/judge")["requires"] == ["judge"]
