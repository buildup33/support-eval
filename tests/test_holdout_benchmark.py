from pathlib import Path

from support_eval.benchmark import load_expected
from support_eval.ingestion import load_conversations, load_rubric

PROJECT_ROOT = Path(__file__).parents[1]
EXPECTED_CRITERIA = {
    "empathy",
    "accuracy",
    "professionalism",
    "resolution",
    "policy_compliance",
    "closing",
}
VALID_STATUSES = {"pass", "fail", "uncertain"}


def test_holdout_has_exactly_fifteen_new_conversations_and_six_criteria():
    holdout = load_conversations(PROJECT_ROOT / "examples" / "holdout-conversations.json")
    original = load_conversations(PROJECT_ROOT / "examples" / "benchmark-conversations.json")
    rubric = load_rubric(PROJECT_ROOT / "examples" / "holdout-rubric.json")

    holdout_ids = {conversation.id for conversation in holdout}
    original_ids = {conversation.id for conversation in original}

    assert len(holdout) == 15
    assert holdout_ids.isdisjoint(original_ids)
    assert len(rubric.criteria) == 6
    assert {criterion.id for criterion in rubric.criteria} == EXPECTED_CRITERIA


def test_holdout_has_exactly_ninety_aligned_valid_expectations():
    conversations = load_conversations(PROJECT_ROOT / "examples" / "holdout-conversations.json")
    expected = load_expected(PROJECT_ROOT / "examples" / "holdout-expected.json")

    conversation_ids = {conversation.id for conversation in conversations}
    expected_ids = {case.conversation_id for case in expected.cases}

    assert len(expected.cases) == 15
    assert expected_ids == conversation_ids
    assert sum(len(case.criteria) for case in expected.cases) == 90
    assert all(set(case.criteria) == EXPECTED_CRITERIA for case in expected.cases)
    assert all(
        expectation.expected in VALID_STATUSES
        for case in expected.cases
        for expectation in case.criteria.values()
    )
    assert all(isinstance(case.should_be_flagged, bool) for case in expected.cases)


def test_holdout_flag_expectations_match_non_pass_criterion_policy():
    rubric = load_rubric(PROJECT_ROOT / "examples" / "holdout-rubric.json")
    expected = load_expected(PROJECT_ROOT / "examples" / "holdout-expected.json")
    flag_on_fail_ids = {criterion.id for criterion in rubric.criteria if criterion.flag_on_fail}

    for case in expected.cases:
        has_flagging_result = any(
            criterion_id in flag_on_fail_ids and expectation.expected != "pass"
            for criterion_id, expectation in case.criteria.items()
        )
        assert case.should_be_flagged is has_flagging_result
