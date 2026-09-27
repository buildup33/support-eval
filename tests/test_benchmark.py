from pathlib import Path

from support_eval.benchmark import (
    BenchmarkExpected,
    ExpectedConversation,
    ExpectedCriterion,
    compare_benchmark,
    load_expected,
)
from support_eval.ingestion import load_conversations, load_rubric
from support_eval.models import (
    ConversationResult,
    ReportResult,
    ReportSummary,
    ScoredCriterionEvaluation,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _evaluation(criterion_id: str, status: str) -> ScoredCriterionEvaluation:
    return ScoredCriterionEvaluation.model_validate(
        {
            "criterion_id": criterion_id,
            "status": status,
            "rating": None,
            "rationale": "Deterministic test fixture.",
            "evidence": [],
            "confidence": 1,
        }
    )


def test_benchmark_examples_cover_every_conversation_and_criterion():
    conversations = load_conversations(PROJECT_ROOT / "examples" / "benchmark-conversations.json")
    rubric = load_rubric(PROJECT_ROOT / "examples" / "benchmark-rubric.json")
    expected = load_expected(PROJECT_ROOT / "examples" / "benchmark-expected.json")

    conversation_ids = {conversation.id for conversation in conversations}
    criterion_ids = {criterion.id for criterion in rubric.criteria}

    assert len(conversations) == 10
    assert len(expected.cases) == 10
    assert {case.conversation_id for case in expected.cases} == conversation_ids
    assert all(set(case.criteria) == criterion_ids for case in expected.cases)


def test_compare_benchmark_reports_each_disagreement_category():
    expected_statuses = {
        "matched": "pass",
        "false-pass": "fail",
        "false-failure": "pass",
        "expected-uncertain": "uncertain",
        "actual-uncertain": "fail",
    }
    actual_statuses = {
        "matched": "pass",
        "false-pass": "pass",
        "false-failure": "fail",
        "expected-uncertain": "fail",
        "actual-uncertain": "uncertain",
    }
    expected = BenchmarkExpected(
        rubric_file="fixture-rubric.json",
        cases=[
            ExpectedConversation(
                conversation_id="fixture-1",
                should_be_flagged=True,
                criteria={
                    criterion_id: ExpectedCriterion(
                        expected=status,
                        justification="Independent fixture expectation.",
                    )
                    for criterion_id, status in expected_statuses.items()
                },
            )
        ],
    )
    report = ReportResult(
        rubric_name="Fixture rubric",
        source_file="fixture-conversations.json",
        summary=ReportSummary(
            conversation_count=1,
            flagged_count=0,
            average_score=50,
        ),
        conversations=[
            ConversationResult(
                conversation_id="fixture-1",
                score=50,
                flagged=False,
                flags=[],
                evaluations=[
                    _evaluation(criterion_id, status)
                    for criterion_id, status in actual_statuses.items()
                ],
            )
        ],
    )

    comparison = compare_benchmark(report, expected)

    assert comparison.total_criteria == 5
    assert comparison.criterion_agreements == 1
    assert comparison.criterion_agreement_percentage == 20
    assert comparison.false_passes == 1
    assert comparison.false_failures == 1
    assert comparison.uncertain_disagreements == 2
    assert comparison.flag_agreements == 0
    assert comparison.flag_agreement_percentage == 0
    assert comparison.disagreements_by_conversation == {
        "fixture-1": [
            "false-pass: expected fail, got pass",
            "false-failure: expected pass, got fail",
            "expected-uncertain: expected uncertain, got fail",
            "actual-uncertain: expected fail, got uncertain",
            "flag: expected True, got False",
        ]
    }


def test_compare_benchmark_reports_perfect_agreement():
    expected = BenchmarkExpected(
        rubric_file="fixture-rubric.json",
        cases=[
            ExpectedConversation(
                conversation_id="fixture-1",
                should_be_flagged=False,
                criteria={
                    "quality": ExpectedCriterion(
                        expected="pass", justification="Expected quality behavior."
                    )
                },
            )
        ],
    )
    report = ReportResult(
        rubric_name="Fixture rubric",
        source_file="fixture-conversations.json",
        summary=ReportSummary(
            conversation_count=1,
            flagged_count=0,
            average_score=100,
        ),
        conversations=[
            ConversationResult(
                conversation_id="fixture-1",
                score=100,
                flagged=False,
                flags=[],
                evaluations=[_evaluation("quality", "pass")],
            )
        ],
    )

    comparison = compare_benchmark(report, expected)

    assert comparison.criterion_agreement_percentage == 100
    assert comparison.flag_agreement_percentage == 100
    assert comparison.disagreements_by_conversation == {}
