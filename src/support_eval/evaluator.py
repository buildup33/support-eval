from __future__ import annotations

import re
from typing import Protocol

from support_eval.models import (
    Conversation,
    CriterionEvaluation,
    EvaluationResult,
    EvidenceReference,
    Rubric,
)
from support_eval.transcript import message_id


class Evaluator(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def evaluate(self, conversation: Conversation, rubric: Rubric) -> EvaluationResult:
        """Evaluate one normalized conversation against a validated rubric."""
        ...


class FakeEvaluator:
    """Offline evaluator that deterministically measures configured keyword coverage."""

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "deterministic-keyword-v1"

    def evaluate(self, conversation: Conversation, rubric: Rubric) -> EvaluationResult:
        agent_messages = [
            (message_id(index), message.content)
            for index, message in enumerate(conversation.messages, start=1)
            if message.role == "agent"
        ]
        transcript = "\n".join(content for _, content in agent_messages).casefold()
        results: list[CriterionEvaluation] = []
        for criterion in rubric.criteria:
            matches = [keyword for keyword in criterion.keywords if keyword in transcript]
            evidence: list[EvidenceReference] = []
            referenced_message_ids: set[str] = set()
            for keyword in matches:
                for current_message_id, content in agent_messages:
                    match = re.search(re.escape(keyword), content, flags=re.IGNORECASE)
                    if match and current_message_id not in referenced_message_ids:
                        evidence.append(EvidenceReference(message_id=current_message_id))
                        referenced_message_ids.add(current_message_id)
                        break
                if len(evidence) == 3:
                    break
            rating = len(matches) / len(criterion.keywords)
            if matches:
                rationale = (
                    f"Matched {len(matches)} of {len(criterion.keywords)} configured keywords."
                )
            else:
                rationale = "Matched 0 configured keywords."
            results.append(
                CriterionEvaluation(
                    criterion_id=criterion.id,
                    status="pass" if rating >= criterion.pass_threshold else "fail",
                    rating=rating,
                    rationale=rationale,
                    evidence=evidence,
                    confidence=1,
                )
            )
        return EvaluationResult(conversation_id=conversation.id, criteria=results)
