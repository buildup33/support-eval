from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from support_eval.evaluator import Evaluator
from support_eval.models import (
    Conversation,
    ConversationResult,
    CriterionEvaluation,
    EvaluationResult,
    EvidenceReference,
    Rubric,
    StrictModel,
)
from support_eval.scoring import EvaluationValidationError, score_evaluation

CHECKPOINT_FILENAME = ".support-eval-checkpoint.json"
CHECKPOINT_FORMAT_VERSION = 2
EVALUATION_CONTRACT_VERSION = "application-owned-message-evidence-v2"
LEGACY_EVALUATION_CONTRACT_VERSION = "structured-evidence-references-v1"


class CheckpointError(ValueError):
    """Raised when a batch checkpoint cannot be safely loaded or reused."""


class CheckpointIdentity(StrictModel):
    conversations_fingerprint: str
    rubric_fingerprint: str
    provider: str
    provider_fingerprint: str
    model: str
    model_fingerprint: str
    contract_version: str
    contract_fingerprint: str


class BatchCheckpoint(StrictModel):
    format_version: Literal[2]
    identity: CheckpointIdentity
    completed: list[ConversationResult]


class LegacyEvidenceReference(StrictModel):
    message_id: str
    quote: str | None


class LegacyCriterionEvaluation(StrictModel):
    criterion_id: str
    status: Literal["pass", "fail", "uncertain"]
    rating: float | None = None
    rationale: str
    evidence: list[LegacyEvidenceReference]
    confidence: float


class LegacyConversationResult(StrictModel):
    conversation_id: str
    score: float
    flagged: bool
    flags: list[str]
    evaluations: list[LegacyCriterionEvaluation]


class LegacyBatchCheckpoint(StrictModel):
    format_version: Literal[1]
    identity: CheckpointIdentity
    completed: list[LegacyConversationResult]


def _fingerprint(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_checkpoint_identity(
    conversations: list[Conversation], rubric: Rubric, evaluator: Evaluator
) -> CheckpointIdentity:
    provider = evaluator.provider_id
    model = evaluator.model_id
    return CheckpointIdentity(
        conversations_fingerprint=_fingerprint(
            [conversation.model_dump(mode="json") for conversation in conversations]
        ),
        rubric_fingerprint=_fingerprint(rubric.model_dump(mode="json")),
        provider=provider,
        provider_fingerprint=_fingerprint(provider),
        model=model,
        model_fingerprint=_fingerprint(model),
        contract_version=EVALUATION_CONTRACT_VERSION,
        contract_fingerprint=_fingerprint(EVALUATION_CONTRACT_VERSION),
    )


def _identity_mismatch(saved: CheckpointIdentity, current: CheckpointIdentity) -> str | None:
    comparisons = [
        (
            "normalized conversations",
            saved.conversations_fingerprint,
            current.conversations_fingerprint,
        ),
        ("rubric", saved.rubric_fingerprint, current.rubric_fingerprint),
        ("evaluator/provider", saved.provider_fingerprint, current.provider_fingerprint),
        ("model", saved.model_fingerprint, current.model_fingerprint),
        ("evaluation contract", saved.contract_fingerprint, current.contract_fingerprint),
    ]
    for label, saved_value, current_value in comparisons:
        if saved_value != current_value:
            return label
    return None


def _legacy_identity_mismatch(saved: CheckpointIdentity, current: CheckpointIdentity) -> str | None:
    comparisons = [
        (
            "normalized conversations",
            saved.conversations_fingerprint,
            current.conversations_fingerprint,
        ),
        ("rubric", saved.rubric_fingerprint, current.rubric_fingerprint),
        ("evaluator/provider", saved.provider_fingerprint, current.provider_fingerprint),
        ("model", saved.model_fingerprint, current.model_fingerprint),
    ]
    for label, saved_value, current_value in comparisons:
        if saved_value != current_value:
            return label
    return None


def _identity_is_internally_consistent(identity: CheckpointIdentity) -> bool:
    return (
        identity.provider_fingerprint == _fingerprint(identity.provider)
        and identity.model_fingerprint == _fingerprint(identity.model)
        and identity.contract_fingerprint == _fingerprint(identity.contract_version)
    )


def load_checkpoint(
    path: Path,
    identity: CheckpointIdentity,
    conversations: list[Conversation],
    rubric: Rubric,
) -> tuple[list[ConversationResult], bool]:
    if not path.exists():
        return [], False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Checkpoint '{path}' is corrupted or unreadable") from exc

    format_version = payload.get("format_version") if isinstance(payload, dict) else None
    try:
        if format_version == CHECKPOINT_FORMAT_VERSION:
            checkpoint = BatchCheckpoint.model_validate(payload)
        elif format_version == 1:
            legacy_checkpoint = LegacyBatchCheckpoint.model_validate(payload)
            migrated = _migrate_legacy_checkpoint(
                path, legacy_checkpoint, identity, conversations, rubric
            )
            return migrated, True
        else:
            raise CheckpointError(
                f"Checkpoint '{path}' uses unsupported format version {format_version!r}"
            )
    except ValidationError as exc:
        raise CheckpointError(f"Checkpoint '{path}' is corrupted or unreadable") from exc

    if not _identity_is_internally_consistent(checkpoint.identity):
        raise CheckpointError(f"Checkpoint '{path}' has inconsistent identity fingerprints")

    mismatch = _identity_mismatch(checkpoint.identity, identity)
    if mismatch:
        raise CheckpointError(
            f"Checkpoint '{path}' cannot be resumed because the {mismatch} fingerprint differs"
        )

    conversations_by_id = {conversation.id: conversation for conversation in conversations}
    completed_ids = [result.conversation_id for result in checkpoint.completed]
    if len(completed_ids) != len(set(completed_ids)):
        raise CheckpointError(f"Checkpoint '{path}' contains duplicate conversation results")

    for saved_result in checkpoint.completed:
        conversation = conversations_by_id.get(saved_result.conversation_id)
        if conversation is None:
            raise CheckpointError(
                f"Checkpoint '{path}' contains an unknown conversation "
                f"'{saved_result.conversation_id}'"
            )
        try:
            evaluation = EvaluationResult(
                conversation_id=saved_result.conversation_id,
                criteria=[
                    CriterionEvaluation(
                        criterion_id=item.criterion_id,
                        status=item.status,
                        rating=item.rating,
                        rationale=item.rationale,
                        evidence=[
                            EvidenceReference(message_id=reference.message_id)
                            for reference in item.evidence
                        ],
                        confidence=item.confidence,
                    )
                    for item in saved_result.evaluations
                ],
            )
            recomputed = score_evaluation(evaluation, rubric, conversation)
        except (EvaluationValidationError, ValidationError) as exc:
            raise CheckpointError(
                f"Checkpoint '{path}' contains an invalid result for "
                f"'{saved_result.conversation_id}'"
            ) from exc
        if recomputed != saved_result:
            raise CheckpointError(
                f"Checkpoint '{path}' contains a result inconsistent with deterministic scoring "
                f"for '{saved_result.conversation_id}'"
            )
    return checkpoint.completed, False


def _migrate_legacy_checkpoint(
    path: Path,
    checkpoint: LegacyBatchCheckpoint,
    identity: CheckpointIdentity,
    conversations: list[Conversation],
    rubric: Rubric,
) -> list[ConversationResult]:
    if not _identity_is_internally_consistent(checkpoint.identity):
        raise CheckpointError(f"Checkpoint '{path}' has inconsistent identity fingerprints")
    if checkpoint.identity.contract_version != LEGACY_EVALUATION_CONTRACT_VERSION:
        raise CheckpointError(f"Checkpoint '{path}' uses an unsupported legacy evaluation contract")
    mismatch = _legacy_identity_mismatch(checkpoint.identity, identity)
    if mismatch:
        raise CheckpointError(
            f"Checkpoint '{path}' cannot be migrated because the {mismatch} fingerprint differs"
        )

    conversations_by_id = {conversation.id: conversation for conversation in conversations}
    migrated: list[ConversationResult] = []
    completed_ids: set[str] = set()
    for legacy_result in checkpoint.completed:
        if legacy_result.conversation_id in completed_ids:
            raise CheckpointError(f"Checkpoint '{path}' contains duplicate conversation results")
        conversation = conversations_by_id.get(legacy_result.conversation_id)
        if conversation is None:
            raise CheckpointError(
                f"Checkpoint '{path}' contains an unknown conversation "
                f"'{legacy_result.conversation_id}'"
            )
        try:
            evaluation = EvaluationResult(
                conversation_id=legacy_result.conversation_id,
                criteria=[
                    CriterionEvaluation(
                        criterion_id=item.criterion_id,
                        status=item.status,
                        rating=item.rating,
                        rationale=item.rationale,
                        evidence=[
                            EvidenceReference(message_id=reference.message_id)
                            for reference in item.evidence
                        ],
                        confidence=item.confidence,
                    )
                    for item in legacy_result.evaluations
                ],
            )
            recomputed = score_evaluation(evaluation, rubric, conversation)
        except (EvaluationValidationError, ValidationError) as exc:
            raise CheckpointError(
                f"Checkpoint '{path}' cannot safely migrate result "
                f"'{legacy_result.conversation_id}'"
            ) from exc
        if (
            recomputed.score != legacy_result.score
            or recomputed.flagged != legacy_result.flagged
            or recomputed.flags != legacy_result.flags
        ):
            raise CheckpointError(
                f"Checkpoint '{path}' cannot safely migrate inconsistent scoring for "
                f"'{legacy_result.conversation_id}'"
            )
        migrated.append(recomputed)
        completed_ids.add(legacy_result.conversation_id)
    return migrated


def write_checkpoint(
    path: Path,
    identity: CheckpointIdentity,
    completed: list[ConversationResult],
) -> None:
    checkpoint = BatchCheckpoint(
        format_version=CHECKPOINT_FORMAT_VERSION,
        identity=identity,
        completed=completed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
