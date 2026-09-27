from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from support_eval.checkpoint import (
    CHECKPOINT_FILENAME,
    LEGACY_EVALUATION_CONTRACT_VERSION,
    CheckpointError,
    build_checkpoint_identity,
)
from support_eval.evaluator import FakeEvaluator
from support_eval.ingestion import load_conversations, load_rubric
from support_eval.models import Conversation, EvaluationResult, Rubric
from support_eval.pipeline import analyze
from support_eval.scoring import score_evaluation

PROJECT_ROOT = Path(__file__).parents[1]
CONVERSATIONS = PROJECT_ROOT / "examples" / "conversations.csv"
RUBRIC = PROJECT_ROOT / "examples" / "rubric.json"


class TrackingEvaluator:
    def __init__(
        self,
        *,
        provider: str = "tracking",
        model: str = "tracking-v1",
        fail_on_call: int | None = None,
        secret: str = "checkpoint-secret-must-not-appear",
    ) -> None:
        self._provider = provider
        self._model = model
        self._fail_on_call = fail_on_call
        self._secret = secret
        self.calls: list[str] = []
        self._delegate = FakeEvaluator()

    @property
    def provider_id(self) -> str:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model

    def evaluate(self, conversation: Conversation, rubric: Rubric) -> EvaluationResult:
        self.calls.append(conversation.id)
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise RuntimeError("synthetic evaluator failure")
        return self._delegate.evaluate(conversation, rubric)


def _fingerprint(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _write_legacy_checkpoint(path: Path, evaluator: TrackingEvaluator) -> str:
    conversations = load_conversations(CONVERSATIONS)
    rubric = load_rubric(RUBRIC)
    identity = build_checkpoint_identity(conversations, rubric, evaluator).model_dump()
    identity["contract_version"] = LEGACY_EVALUATION_CONTRACT_VERSION
    identity["contract_fingerprint"] = _fingerprint(LEGACY_EVALUATION_CONTRACT_VERSION)
    evaluation = FakeEvaluator().evaluate(conversations[0], rubric)
    scored = score_evaluation(evaluation, rubric, conversations[0])
    legacy_result = scored.model_dump()
    for criterion in legacy_result["evaluations"]:
        criterion["evidence"] = [
            {"message_id": item["message_id"], "quote": "ignore this generated quote"}
            for item in criterion["evidence"]
        ]
    payload = {
        "format_version": 1,
        "identity": identity,
        "completed": [legacy_result],
    }
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def test_failure_after_n_conversations_preserves_first_n_and_no_credentials(tmp_path):
    evaluator = TrackingEvaluator(fail_on_call=3)

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, evaluator)

    checkpoint_path = tmp_path / CHECKPOINT_FILENAME
    payload_text = checkpoint_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert checkpoint_path.is_file()
    assert [item["conversation_id"] for item in payload["completed"]] == [
        "ticket-1001",
        "ticket-1002",
    ]
    assert evaluator._secret not in payload_text


def test_resume_skips_completed_conversations_and_finishes_artifacts(tmp_path):
    first = TrackingEvaluator(fail_on_call=3)
    with pytest.raises(RuntimeError):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, first)

    resumed = TrackingEvaluator()
    report = analyze(CONVERSATIONS, RUBRIC, tmp_path, resumed)

    assert resumed.calls == ["ticket-1003"]
    assert report.summary.conversation_count == 3
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report.html").is_file()
    assert resumed._secret not in (tmp_path / "results.json").read_text(encoding="utf-8")
    assert resumed._secret not in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert not (tmp_path / CHECKPOINT_FILENAME).exists()


def test_changed_input_fingerprint_refuses_checkpoint_reuse(tmp_path):
    evaluator = TrackingEvaluator(fail_on_call=2)
    with pytest.raises(RuntimeError):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, evaluator)

    changed_input = tmp_path / "changed.csv"
    changed_input.write_text(
        CONVERSATIONS.read_text(encoding="utf-8").replace(
            "I was charged twice", "I was charged three times", 1
        ),
        encoding="utf-8",
    )
    resumed = TrackingEvaluator()

    with pytest.raises(CheckpointError, match="normalized conversations fingerprint differs"):
        analyze(changed_input, RUBRIC, tmp_path, resumed)
    assert resumed.calls == []


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("different-provider", "tracking-v1", "evaluator/provider fingerprint differs"),
        ("tracking", "different-model", "model fingerprint differs"),
    ],
)
def test_evaluator_or_model_mismatch_refuses_checkpoint_reuse(tmp_path, provider, model, expected):
    evaluator = TrackingEvaluator(fail_on_call=2)
    with pytest.raises(RuntimeError):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, evaluator)

    resumed = TrackingEvaluator(provider=provider, model=model)
    with pytest.raises(CheckpointError, match=expected):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, resumed)
    assert resumed.calls == []


def test_corrupted_checkpoint_fails_before_evaluation(tmp_path):
    (tmp_path / CHECKPOINT_FILENAME).write_text("{not-json", encoding="utf-8")
    evaluator = TrackingEvaluator()

    with pytest.raises(CheckpointError, match="corrupted or unreadable"):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, evaluator)
    assert evaluator.calls == []


def test_legacy_quote_checkpoint_migrates_without_trusting_quotes_or_mixing_contracts(
    tmp_path,
):
    checkpoint_path = tmp_path / CHECKPOINT_FILENAME
    evaluator = TrackingEvaluator()
    _write_legacy_checkpoint(checkpoint_path, evaluator)
    resumed = TrackingEvaluator(fail_on_call=1)

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, resumed)

    payload_text = checkpoint_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert resumed.calls == ["ticket-1002"]
    assert payload["format_version"] == 2
    assert payload["identity"]["contract_version"] == "application-owned-message-evidence-v2"
    assert "ignore this generated quote" not in payload_text
    assert '"quote"' not in payload_text
    assert '"text"' in payload_text


def test_unsupported_legacy_contract_is_rejected_without_modifying_checkpoint(tmp_path):
    checkpoint_path = tmp_path / CHECKPOINT_FILENAME
    original = _write_legacy_checkpoint(checkpoint_path, TrackingEvaluator())
    payload = json.loads(original)
    payload["identity"]["contract_version"] = "unknown-legacy-contract"
    payload["identity"]["contract_fingerprint"] = _fingerprint("unknown-legacy-contract")
    unsupported = json.dumps(payload, indent=2) + "\n"
    checkpoint_path.write_text(unsupported, encoding="utf-8")

    with pytest.raises(CheckpointError, match="unsupported legacy evaluation contract"):
        analyze(CONVERSATIONS, RUBRIC, tmp_path, TrackingEvaluator())

    assert checkpoint_path.read_text(encoding="utf-8") == unsupported
