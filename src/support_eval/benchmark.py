from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from support_eval.models import ReportResult, StrictModel


class ExpectedCriterion(StrictModel):
    expected: Literal["pass", "fail", "uncertain"]
    justification: str = Field(min_length=1)


class ExpectedConversation(StrictModel):
    conversation_id: str = Field(min_length=1)
    should_be_flagged: bool
    criteria: dict[str, ExpectedCriterion] = Field(min_length=1)


class BenchmarkExpected(StrictModel):
    rubric_file: str = Field(min_length=1)
    cases: list[ExpectedConversation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_conversation_ids(self) -> BenchmarkExpected:
        ids = [case.conversation_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark conversation ids must be unique")
        return self


class BenchmarkComparison(StrictModel):
    total_criteria: int = Field(ge=0)
    criterion_agreements: int = Field(ge=0)
    criterion_agreement_percentage: float = Field(ge=0, le=100)
    false_passes: int = Field(ge=0)
    false_failures: int = Field(ge=0)
    uncertain_disagreements: int = Field(ge=0)
    flag_agreements: int = Field(ge=0)
    total_flags: int = Field(ge=0)
    flag_agreement_percentage: float = Field(ge=0, le=100)
    disagreements_by_conversation: dict[str, list[str]]


def load_expected(path: str | Path) -> BenchmarkExpected:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    return BenchmarkExpected.model_validate(payload)


def load_results(path: str | Path) -> ReportResult:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    return ReportResult.model_validate(payload)


def compare_benchmark(report: ReportResult, expected: BenchmarkExpected) -> BenchmarkComparison:
    actual_by_id = {result.conversation_id: result for result in report.conversations}
    expected_ids = {case.conversation_id for case in expected.cases}
    total_criteria = 0
    agreements = 0
    false_passes = 0
    false_failures = 0
    uncertain_disagreements = 0
    flag_agreements = 0
    disagreements: dict[str, list[str]] = {}

    for expected_case in expected.cases:
        case_disagreements: list[str] = []
        actual_case = actual_by_id.get(expected_case.conversation_id)
        if actual_case is None:
            total_criteria += len(expected_case.criteria)
            case_disagreements.append("conversation missing from generated results")
            disagreements[expected_case.conversation_id] = case_disagreements
            continue

        actual_criteria = {
            evaluation.criterion_id: evaluation.status for evaluation in actual_case.evaluations
        }
        for criterion_id, expectation in expected_case.criteria.items():
            total_criteria += 1
            actual_status = actual_criteria.get(criterion_id)
            if actual_status == expectation.expected:
                agreements += 1
                continue
            if expectation.expected == "fail" and actual_status == "pass":
                false_passes += 1
            if expectation.expected == "pass" and actual_status == "fail":
                false_failures += 1
            if "uncertain" in {expectation.expected, actual_status}:
                uncertain_disagreements += 1
            case_disagreements.append(
                f"{criterion_id}: expected {expectation.expected}, got {actual_status or 'missing'}"
            )

        if actual_case.flagged == expected_case.should_be_flagged:
            flag_agreements += 1
        else:
            case_disagreements.append(
                f"flag: expected {expected_case.should_be_flagged}, got {actual_case.flagged}"
            )
        if case_disagreements:
            disagreements[expected_case.conversation_id] = case_disagreements

    for unexpected_id in sorted(set(actual_by_id) - expected_ids):
        disagreements[unexpected_id] = ["unexpected conversation in generated results"]

    total_flags = len(expected.cases)
    criterion_percentage = round(agreements / total_criteria * 100, 2) if total_criteria else 0
    flag_percentage = round(flag_agreements / total_flags * 100, 2) if total_flags else 0
    return BenchmarkComparison(
        total_criteria=total_criteria,
        criterion_agreements=agreements,
        criterion_agreement_percentage=criterion_percentage,
        false_passes=false_passes,
        false_failures=false_failures,
        uncertain_disagreements=uncertain_disagreements,
        flag_agreements=flag_agreements,
        total_flags=total_flags,
        flag_agreement_percentage=flag_percentage,
        disagreements_by_conversation=disagreements,
    )


def compare_files(results_path: str | Path, expected_path: str | Path) -> BenchmarkComparison:
    return compare_benchmark(load_results(results_path), load_expected(expected_path))
