import pytest
from pydantic import ValidationError

from support_eval.models import (
    Conversation,
    CriterionEvaluation,
    EvaluationResult,
    EvidenceReference,
    Message,
    ResolvedEvidence,
    Rubric,
    RubricCriterion,
)
from support_eval.scoring import EvaluationValidationError, score_evaluation, validate_evaluation


def _rubric():
    return Rubric(
        name="Test",
        flag_threshold=70,
        criteria=[
            RubricCriterion(
                id="a",
                name="A",
                description="A criterion",
                weight=1,
                pass_threshold=0.5,
                keywords=["a"],
            ),
            RubricCriterion(
                id="b",
                name="B",
                description="B criterion",
                weight=3,
                pass_threshold=0.5,
                keywords=["b"],
            ),
        ],
    )


def _conversation():
    return Conversation(
        id="one",
        messages=[
            Message(role="customer", content="The first piece of evidence is here."),
            Message(role="agent", content="The ambiguous evidence is in this reply."),
        ],
    )


def test_weighted_scoring_and_flags_are_outside_evaluator():
    evaluation = EvaluationResult(
        conversation_id="one",
        criteria=[
            CriterionEvaluation(
                criterion_id="a", status="pass", rating=1, rationale="yes", confidence=1
            ),
            CriterionEvaluation(
                criterion_id="b", status="fail", rating=0, rationale="no", confidence=1
            ),
        ],
    )
    result = score_evaluation(evaluation, _rubric(), _conversation())
    assert result.score == 25
    assert result.flagged is True
    assert "Failed criterion: B" in result.flags
    assert "Overall score below 70" in result.flags


def test_validation_rejects_missing_criterion():
    evaluation = EvaluationResult(
        conversation_id="one",
        criteria=[
            CriterionEvaluation(
                criterion_id="a", status="pass", rating=1, rationale="yes", confidence=1
            )
        ],
    )
    with pytest.raises(EvaluationValidationError, match="missing: b"):
        validate_evaluation(evaluation, _conversation(), _rubric())


def test_llm_statuses_have_deterministic_scores_and_uncertain_is_flagged():
    evaluation = EvaluationResult(
        conversation_id="one",
        criteria=[
            CriterionEvaluation(
                criterion_id="a",
                status="pass",
                rationale="Clear evidence",
                evidence=[EvidenceReference(message_id="m0001")],
                confidence=0.9,
            ),
            CriterionEvaluation(
                criterion_id="b",
                status="uncertain",
                rationale="Not enough evidence",
                evidence=[EvidenceReference(message_id="m0002")],
                confidence=0.4,
            ),
        ],
    )
    result = score_evaluation(evaluation, _rubric(), _conversation())
    assert result.score == 62.5
    assert "Failed criterion: B" in result.flags


def _single_criterion_rubric():
    return Rubric(
        name="Evidence QA",
        criteria=[
            RubricCriterion(
                id="evidence",
                name="Evidence",
                description="Use exact evidence",
                weight=1,
                pass_threshold=0.5,
                keywords=["evidence"],
            )
        ],
    )


def _evidence_evaluation(*references):
    return EvaluationResult(
        conversation_id="evidence-case",
        criteria=[
            CriterionEvaluation(
                criterion_id="evidence",
                status="pass",
                rationale="Synthetic evidence fixture.",
                evidence=list(references),
                confidence=1,
            )
        ],
    )


def _evidence_conversation():
    return Conversation(
        id="evidence-case",
        messages=[
            Message(role="customer", content="Order 42 is delayed."),
            Message(role="agent", content="I opened trace TR-900 for your order."),
        ],
    )


def test_message_id_only_resolves_application_owned_evidence():
    result = score_evaluation(
        _evidence_evaluation(EvidenceReference(message_id="m0001")),
        _single_criterion_rubric(),
        _evidence_conversation(),
    )
    assert result.evaluations[0].evidence == [
        ResolvedEvidence(
            message_id="m0001",
            role="customer",
            text="Order 42 is delayed.",
        )
    ]


def test_multiple_evidence_references_are_resolved_in_order():
    result = score_evaluation(
        _evidence_evaluation(
            EvidenceReference(message_id="m0001"),
            EvidenceReference(message_id="m0002"),
        ),
        _single_criterion_rubric(),
        _evidence_conversation(),
    )
    assert [reference.message_id for reference in result.evaluations[0].evidence] == [
        "m0001",
        "m0002",
    ]


def test_nonexistent_evidence_message_id_is_rejected():
    with pytest.raises(EvaluationValidationError, match="message 'm9999' does not exist"):
        score_evaluation(
            _evidence_evaluation(EvidenceReference(message_id="m9999")),
            _single_criterion_rubric(),
            _evidence_conversation(),
        )


def test_message_id_from_another_conversation_is_rejected_without_key_error():
    other_conversation = Conversation(
        id="evidence-case",
        messages=[Message(role="customer", content="Only one message exists here.")],
    )
    with pytest.raises(
        EvaluationValidationError,
        match="conversation 'evidence-case'.*criterion 'evidence'.*message 'm0002'.*does not exist",
    ):
        score_evaluation(
            _evidence_evaluation(EvidenceReference(message_id="m0002")),
            _single_criterion_rubric(),
            other_conversation,
        )


def test_duplicate_message_ids_are_deduplicated_preserving_first_seen_order():
    result = score_evaluation(
        _evidence_evaluation(
            EvidenceReference(message_id="m0002"),
            EvidenceReference(message_id="m0001"),
            EvidenceReference(message_id="m0002"),
        ),
        _single_criterion_rubric(),
        _evidence_conversation(),
    )
    assert [reference.message_id for reference in result.evaluations[0].evidence] == [
        "m0002",
        "m0001",
    ]


def test_empty_evidence_list_is_explicitly_valid():
    result = score_evaluation(
        _evidence_evaluation(),
        _single_criterion_rubric(),
        _evidence_conversation(),
    )
    assert result.score == 100


def test_punctuation_casing_unicode_and_long_text_need_no_model_copying():
    text = "Élodie said: ‘Keep CASE, punctuation—exactly!’ 🚚 " + ("long-text " * 100).rstrip()
    conversation = Conversation(
        id="evidence-case",
        messages=[Message(role="agent", content=text)],
    )
    result = score_evaluation(
        _evidence_evaluation(EvidenceReference(message_id="m0001")),
        _single_criterion_rubric(),
        conversation,
    )
    assert result.evaluations[0].evidence[0].text == text


def test_model_generated_quote_field_is_rejected_as_malformed():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceReference.model_validate(
            {"message_id": "m0001", "quote": "obsolete generated text"}
        )


@pytest.mark.parametrize("invalid_message_id", ["", "2", "message-2", "m2"])
def test_malformed_message_references_fail_model_validation(invalid_message_id):
    with pytest.raises(ValidationError, match="message_id"):
        EvidenceReference(message_id=invalid_message_id)


def test_more_than_three_evidence_references_is_rejected():
    with pytest.raises(ValidationError, match="at most 3 items"):
        _evidence_evaluation(
            EvidenceReference(message_id="m0001"),
            EvidenceReference(message_id="m0002"),
            EvidenceReference(message_id="m0003"),
            EvidenceReference(message_id="m0004"),
        )
