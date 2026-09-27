from __future__ import annotations

from support_eval.models import (
    Conversation,
    ConversationResult,
    EvaluationResult,
    ResolvedEvidence,
    Rubric,
    ScoredCriterionEvaluation,
)
from support_eval.transcript import message_id


class EvaluationValidationError(ValueError):
    """Raised when evaluator output does not match the requested conversation and rubric."""


def validate_evaluation(
    evaluation: EvaluationResult, conversation: Conversation, rubric: Rubric
) -> None:
    if evaluation.conversation_id != conversation.id:
        returned_id = evaluation.conversation_id
        raise EvaluationValidationError(
            f"Evaluator returned conversation '{returned_id}' for '{conversation.id}'"
        )
    expected = [criterion.id for criterion in rubric.criteria]
    actual = [result.criterion_id for result in evaluation.criteria]
    if len(actual) != len(set(actual)):
        raise EvaluationValidationError("Evaluator returned duplicate criterion results")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise EvaluationValidationError("Evaluator criterion mismatch (" + "; ".join(details) + ")")

    valid_message_ids = {
        message_id(index) for index, _ in enumerate(conversation.messages, start=1)
    }
    for result in evaluation.criteria:
        for reference in result.evidence:
            if reference.message_id not in valid_message_ids:
                raise EvaluationValidationError(
                    f"Evidence validation failed for conversation '{conversation.id}', criterion "
                    f"'{result.criterion_id}': message '{reference.message_id}' does not exist"
                )


def _resolved_evaluations(
    evaluation: EvaluationResult, conversation: Conversation
) -> list[ScoredCriterionEvaluation]:
    messages_by_id = {
        message_id(index): message for index, message in enumerate(conversation.messages, start=1)
    }
    resolved: list[ScoredCriterionEvaluation] = []
    for result in evaluation.criteria:
        seen: set[str] = set()
        evidence: list[ResolvedEvidence] = []
        for reference in result.evidence:
            if reference.message_id in seen:
                continue
            message = messages_by_id[reference.message_id]
            evidence.append(
                ResolvedEvidence(
                    message_id=reference.message_id,
                    role=message.role,
                    text=message.content,
                )
            )
            seen.add(reference.message_id)
        resolved.append(
            ScoredCriterionEvaluation(
                criterion_id=result.criterion_id,
                status=result.status,
                rating=result.rating,
                rationale=result.rationale,
                evidence=evidence,
                confidence=result.confidence,
            )
        )
    return resolved


def _rating_for(result_status: str, evaluator_rating: float | None) -> float:
    if evaluator_rating is not None:
        return evaluator_rating
    return {"pass": 1.0, "uncertain": 0.5, "fail": 0.0}[result_status]


def score_evaluation(
    evaluation: EvaluationResult, rubric: Rubric, conversation: Conversation
) -> ConversationResult:
    validate_evaluation(evaluation, conversation, rubric)
    by_id = {
        result.criterion_id: result for result in _resolved_evaluations(evaluation, conversation)
    }
    ratings = {
        criterion_id: _rating_for(result.status, result.rating)
        for criterion_id, result in by_id.items()
    }
    total_weight = sum(criterion.weight for criterion in rubric.criteria)
    weighted_rating = sum(ratings[criterion.id] * criterion.weight for criterion in rubric.criteria)
    score = round((weighted_rating / total_weight) * 100, 2)
    flags = [
        f"Failed criterion: {criterion.name}"
        for criterion in rubric.criteria
        if criterion.flag_on_fail
        and (
            by_id[criterion.id].status != "pass" or ratings[criterion.id] < criterion.pass_threshold
        )
    ]
    if score < rubric.flag_threshold:
        flags.append(f"Overall score below {rubric.flag_threshold:g}")
    return ConversationResult(
        conversation_id=evaluation.conversation_id,
        score=score,
        flagged=bool(flags),
        flags=flags,
        evaluations=[by_id[criterion.id] for criterion in rubric.criteria],
    )
