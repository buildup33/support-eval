from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Message(StrictModel):
    role: Literal["customer", "agent", "system"]
    content: str = Field(min_length=1)
    timestamp: str | None = None


class Conversation(StrictModel):
    id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RubricCriterion(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(gt=0)
    pass_threshold: float = Field(ge=0, le=1)
    keywords: list[str] = Field(min_length=1)
    flag_on_fail: bool = True

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip().casefold() for value in values if value.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty keyword is required")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("keywords must be unique within a criterion")
        return cleaned


class Rubric(StrictModel):
    name: str = Field(min_length=1)
    criteria: list[RubricCriterion] = Field(min_length=1)
    flag_threshold: float = Field(default=70, ge=0, le=100)

    @model_validator(mode="after")
    def validate_unique_criterion_ids(self) -> Rubric:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        return self


class EvidenceReference(StrictModel):
    message_id: str = Field(pattern=r"^m\d{4,}$")


class ResolvedEvidence(StrictModel):
    message_id: str = Field(pattern=r"^m\d{4,}$")
    role: Literal["customer", "agent", "system"]
    text: str = Field(min_length=1)


class CriterionEvaluation(StrictModel):
    criterion_id: str = Field(min_length=1)
    status: Literal["pass", "fail", "uncertain"]
    rating: float | None = Field(default=None, ge=0, le=1)
    rationale: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1)


class LLMCriterionEvaluation(StrictModel):
    criterion_id: str = Field(min_length=1)
    result: Literal["pass", "fail", "uncertain"]
    explanation: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(max_length=3)
    confidence: float = Field(ge=0, le=1)


class LLMEvaluationResponse(StrictModel):
    conversation_id: str = Field(min_length=1)
    criteria: list[LLMCriterionEvaluation] = Field(min_length=1)


class EvaluationResult(StrictModel):
    conversation_id: str = Field(min_length=1)
    criteria: list[CriterionEvaluation] = Field(min_length=1)


class ScoredCriterionEvaluation(StrictModel):
    criterion_id: str = Field(min_length=1)
    status: Literal["pass", "fail", "uncertain"]
    rating: float | None = Field(default=None, ge=0, le=1)
    rationale: str = Field(min_length=1)
    evidence: list[ResolvedEvidence] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1)


class ConversationResult(StrictModel):
    conversation_id: str
    score: float = Field(ge=0, le=100)
    flagged: bool
    flags: list[str]
    evaluations: list[ScoredCriterionEvaluation]


class ReportSummary(StrictModel):
    conversation_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    average_score: float = Field(ge=0, le=100)


class ReportResult(StrictModel):
    rubric_name: str
    source_file: str
    summary: ReportSummary
    conversations: list[ConversationResult]
