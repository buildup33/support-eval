from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import pytest

from support_eval.evaluator import FakeEvaluator
from support_eval.models import CriterionEvaluation, EvaluationResult, EvidenceReference
from support_eval.web import create_app

CSV_BYTES = b"""conversation_id,role,message,timestamp
ticket-1,customer,I need help,
ticket-1,agent,I understand and can help. This is resolved.,
ticket-2,customer,The issue remains,
ticket-2,agent,I will follow up after I verify it.,
"""
RUBRIC = {
    "name": "Web QA",
    "flag_threshold": 50,
    "criteria": [
        {
            "id": "helpfulness",
            "name": "Helpfulness",
            "description": "Provides useful support.",
            "weight": 1,
            "pass_threshold": 0.5,
            "keywords": ["help", "resolved"],
            "flag_on_fail": True,
        }
    ],
}


@pytest.fixture
def app(tmp_path: Path):  # type: ignore[no-untyped-def]
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: FakeEvaluator(),
        }
    )
    yield application
    application.extensions["support_eval_jobs"].shutdown()


def _csrf_token(client) -> str:  # type: ignore[no-untyped-def]
    page = client.get("/").get_data(as_text=True)
    match = re.search(r'name="_csrf_token" value="([^"]+)"', page)
    assert match is not None
    return match.group(1)


def _post_job(client, *, filename: str = "tickets.csv", content: bytes = CSV_BYTES):  # type: ignore[no-untyped-def]
    return client.post(
        "/jobs",
        data={
            "_csrf_token": _csrf_token(client),
            "conversations": (io.BytesIO(content), filename),
            "rubric": json.dumps(RUBRIC),
        },
        content_type="multipart/form-data",
        headers={"Origin": "http://localhost"},
    )


def _wait_for_state(client, job_id: str, expected: str, timeout: float = 3):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    payload = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        payload = response.get_json()
        if payload["state"] == expected:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach {expected}: {payload}")


def _job_id(response) -> str:  # type: ignore[no-untyped-def]
    return response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]


def test_home_page_has_polished_setup_flow(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Evaluate support conversations" in response.data
    assert b"Drop your file here" in response.data
    assert b"Criteria preview" in response.data
    assert b"AI judgments are beta" in response.data
    sample = client.get("/samples/conversations.csv")
    assert sample.status_code == 200
    assert b"conversation_id,role,message,timestamp" in sample.data


def test_valid_upload_runs_and_renders_native_results(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    response = _post_job(client)

    assert response.status_code == 302
    job_id = _job_id(response)
    status = _wait_for_state(client, job_id, "complete")
    assert status["completed"] == 2
    assert status["total"] == 2

    job_dir = tmp_path / "jobs" / job_id
    assert (job_dir / "conversations.csv").is_file()
    assert (job_dir / "rubric.json").is_file()
    assert (job_dir / "results.json").is_file()
    assert (job_dir / "report.html").is_file()
    assert not (job_dir / ".support-eval-checkpoint.json").exists()

    results = client.get(f"/jobs/{job_id}/results")
    assert b"Criteria performance" in results.data
    assert b"ticket-1" in results.data
    assert b"Download JSON" in results.data

    detail = client.get(f"/jobs/{job_id}/conversations/ticket-1")
    assert b"Criterion judgments" in detail.data
    assert b"message-m0002" in detail.data
    assert b"I understand and can help. This is resolved." in detail.data


def test_downloads_return_generated_artifacts(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    job_id = _job_id(_post_job(client))
    _wait_for_state(client, job_id, "complete")

    json_response = client.get(f"/jobs/{job_id}/download/results.json")
    html_response = client.get(f"/jobs/{job_id}/download/report.html")

    assert json_response.status_code == 200
    assert json_response.json["rubric_name"] == "Web QA"
    assert html_response.status_code == 200
    assert b"Support Eval Report" in html_response.data


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("tickets.txt", CSV_BYTES, b"must be CSV or JSON"),
        ("tickets.csv", b"bad,columns\n1,2\n", b"missing required columns"),
        ("tickets.json", b"not-json", b"Invalid JSON"),
    ],
)
def test_invalid_uploads_are_rejected(app, filename: str, content: bytes, message: bytes) -> None:  # type: ignore[no-untyped-def]
    response = _post_job(app.test_client(), filename=filename, content=content)

    assert response.status_code == 400
    assert message in response.data


def test_invalid_rubric_is_rejected(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    response = client.post(
        "/jobs",
        data={
            "_csrf_token": _csrf_token(client),
            "conversations": (io.BytesIO(CSV_BYTES), "tickets.csv"),
            "rubric": "{bad json",
        },
        content_type="multipart/form-data",
        headers={"Origin": "http://localhost"},
    )

    assert response.status_code == 400
    assert b"Invalid JSON" in response.data


def test_missing_credentials_has_actionable_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "JOBS_ROOT": tmp_path / "jobs"})
    try:
        response = _post_job(app.test_client())
        assert response.status_code == 400
        assert b"GROQ_API_KEY is not set" in response.data
    finally:
        app.extensions["support_eval_jobs"].shutdown()


def test_oversized_upload_returns_friendly_error(tmp_path: Path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "MAX_CONTENT_LENGTH": 100,
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: FakeEvaluator(),
        }
    )
    try:
        response = _post_job(app.test_client())
        assert response.status_code == 413
        assert b"Uploads are limited" in response.data
    finally:
        app.extensions["support_eval_jobs"].shutdown()


def test_uploaded_filename_cannot_control_storage_or_render_html(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    unsafe_name = "evil<script>alert(1)<script>.csv"
    response = _post_job(client, filename=unsafe_name)
    job_id = _job_id(response)
    _wait_for_state(client, job_id, "complete")

    job_dir = tmp_path / "jobs" / job_id
    assert sorted(path.name for path in job_dir.iterdir()) == [
        "conversations.csv",
        "job.json",
        "report.html",
        "results.json",
        "rubric.json",
    ]
    results = client.get(f"/jobs/{job_id}/results")
    assert b"<script>alert(1)<script>" not in results.data


def test_unknown_job_and_artifact_return_404(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    assert client.get("/jobs/not-a-job").status_code == 404
    job_id = _job_id(_post_job(client))
    _wait_for_state(client, job_id, "complete")
    assert client.get(f"/jobs/{job_id}/download/secrets.txt").status_code == 404


def test_failure_preserves_checkpoint_and_resume_skips_completed(tmp_path: Path) -> None:
    class FailOnceEvaluator:
        provider_id = "fake-resumable"
        model_id = "fake-resumable-v1"

        def __init__(self) -> None:
            self.failed = False
            self.calls: list[str] = []
            self.fake = FakeEvaluator()

        def evaluate(self, conversation, rubric):  # type: ignore[no-untyped-def]
            self.calls.append(conversation.id)
            if conversation.id == "ticket-2" and not self.failed:
                self.failed = True
                raise RuntimeError("Temporary provider interruption")
            return self.fake.evaluate(conversation, rubric)

    evaluator = FailOnceEvaluator()
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: evaluator,
        }
    )
    try:
        client = app.test_client()
        job_id = _job_id(_post_job(client))
        failed = _wait_for_state(client, job_id, "failed")
        assert failed["completed"] == 1
        assert (tmp_path / "jobs" / job_id / ".support-eval-checkpoint.json").is_file()

        response = client.post(
            f"/jobs/{job_id}/resume",
            data={"_csrf_token": _csrf_token(client)},
            headers={"Origin": "http://localhost"},
        )
        assert response.status_code == 302
        _wait_for_state(client, job_id, "complete")
        assert evaluator.calls == ["ticket-1", "ticket-2", "ticket-2"]
    finally:
        app.extensions["support_eval_jobs"].shutdown()


def test_failure_error_redacts_credentials(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    secret = "gsk_super_secret_value_that_must_not_appear"
    monkeypatch.setenv("GROQ_API_KEY", secret)

    class LeakyEvaluator:
        provider_id = "test"
        model_id = "test"

        def evaluate(self, _conversation, _rubric):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"Authorization: Bearer {secret}")

    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: LeakyEvaluator(),
        }
    )
    try:
        client = app.test_client()
        job_id = _job_id(_post_job(client))
        failed = _wait_for_state(client, job_id, "failed")
        serialized = json.dumps(failed)
        assert secret not in serialized
        assert "[REDACTED]" in serialized
        assert secret not in (tmp_path / "jobs" / job_id / "job.json").read_text()
    finally:
        app.extensions["support_eval_jobs"].shutdown()


@pytest.fixture
def mixed_status_job(tmp_path: Path):  # type: ignore[no-untyped-def]
    mixed_rubric = {
        "name": "Mixed Status QA",
        "flag_threshold": 50,
        "criteria": [
            {
                "id": status,
                "name": status.title(),
                "description": f"Produces a {status} result for presentation testing.",
                "weight": 1,
                "pass_threshold": 1,
                "keywords": [status],
                "flag_on_fail": True,
            }
            for status in ("pass", "uncertain", "fail")
        ],
    }

    class MixedStatusEvaluator:
        provider_id = "test"
        model_id = "mixed-status"

        def evaluate(self, conversation, _rubric):  # type: ignore[no-untyped-def]
            return EvaluationResult(
                conversation_id=conversation.id,
                criteria=[
                    CriterionEvaluation(
                        criterion_id=status,
                        status=status,
                        rationale=f"The criterion is {status}.",
                        evidence=[EvidenceReference(message_id="m0002")],
                        confidence=0.9,
                    )
                    for status in ("pass", "uncertain", "fail")
                ],
            )

    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: MixedStatusEvaluator(),
        }
    )
    client = app.test_client()
    token = _csrf_token(client)
    response = client.post(
        "/jobs",
        data={
            "_csrf_token": token,
            "conversations": (io.BytesIO(CSV_BYTES.split(b"ticket-2")[0]), "ticket.csv"),
            "rubric": json.dumps(mixed_rubric),
        },
        content_type="multipart/form-data",
        headers={"Origin": "http://localhost"},
    )
    job_id = _job_id(response)
    _wait_for_state(client, job_id, "complete")
    yield client, job_id
    app.extensions["support_eval_jobs"].shutdown()


def test_conversation_keeps_fail_and_uncertain_semantically_distinct(mixed_status_job) -> None:  # type: ignore[no-untyped-def]
    client, job_id = mixed_status_job
    html = client.get(f"/jobs/{job_id}/conversations/ticket-1").get_data(as_text=True)

    assert "1 failed" in html
    assert "1 uncertain" in html
    assert 'class="judgment-card judgment-fail"' in html
    assert 'class="judgment-card judgment-uncertain"' in html
    assert (
        html.index("judgment-fail") < html.index("judgment-uncertain") < html.index("judgment-pass")
    )
    assert "% confidence" not in html
    assert "Flagged for review by deterministic rules" in html
    assert "Failed criterion: Resolution" not in html


def test_evidence_control_maps_to_exact_transcript_message(mixed_status_job) -> None:  # type: ignore[no-untyped-def]
    client, job_id = mixed_status_job
    html = client.get(f"/jobs/{job_id}/conversations/ticket-1").get_data(as_text=True)

    assert 'id="message-m0002"' in html
    assert 'data-evidence-target="message-m0002"' in html
    assert 'aria-controls="message-m0002"' in html
    assert 'aria-label="Show evidence in m0002, agent message"' in html
    assert "m0002 · Agent" in html


def test_conversation_has_accessible_back_to_results_link(mixed_status_job) -> None:  # type: ignore[no-untyped-def]
    client, job_id = mixed_status_job
    html = client.get(f"/jobs/{job_id}/conversations/ticket-1").get_data(as_text=True)

    assert f'href="/jobs/{job_id}/results"' in html
    assert "← Back to results" in html


def test_state_changing_request_requires_csrf_token(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    response = client.post(
        "/jobs",
        data={
            "conversations": (io.BytesIO(CSV_BYTES), "tickets.csv"),
            "rubric": json.dumps(RUBRIC),
        },
        content_type="multipart/form-data",
        headers={"Origin": "http://localhost"},
    )

    assert response.status_code == 403


def test_cross_origin_state_change_is_rejected(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    response = client.post(
        "/jobs",
        data={
            "_csrf_token": _csrf_token(client),
            "conversations": (io.BytesIO(CSV_BYTES), "tickets.csv"),
            "rubric": json.dumps(RUBRIC),
        },
        content_type="multipart/form-data",
        headers={"Origin": "https://malicious.example"},
    )

    assert response.status_code == 403


def test_untrusted_host_is_rejected(app) -> None:  # type: ignore[no-untyped-def]
    response = app.test_client().get("/", headers={"Host": "malicious.example"})

    assert response.status_code == 400


def test_normal_127_loopback_post_is_allowed(app) -> None:  # type: ignore[no-untyped-def]
    client = app.test_client()
    page = client.get("/", base_url="http://127.0.0.1:8000")
    match = re.search(r'name="_csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    assert match is not None
    response = client.post(
        "/jobs",
        base_url="http://127.0.0.1:8000",
        data={
            "_csrf_token": match.group(1),
            "conversations": (io.BytesIO(CSV_BYTES), "tickets.csv"),
            "rubric": json.dumps(RUBRIC),
        },
        content_type="multipart/form-data",
        headers={"Origin": "http://127.0.0.1:8000"},
    )

    assert response.status_code == 302


def test_duplicate_json_ids_fail_before_evaluator_invocation(tmp_path: Path) -> None:
    calls: list[str] = []

    class RecordingEvaluator(FakeEvaluator):
        def evaluate(self, conversation, rubric):  # type: ignore[no-untyped-def]
            calls.append(conversation.id)
            return super().evaluate(conversation, rubric)

    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "JOBS_ROOT": tmp_path / "jobs",
            "REQUIRE_GROQ_API_KEY": False,
            "EVALUATOR_FACTORY": lambda _notify: RecordingEvaluator(),
        }
    )
    duplicate_json = json.dumps(
        {
            "conversations": [
                {"id": "same-id", "messages": [{"role": "customer", "content": "One"}]},
                {"id": "same-id", "messages": [{"role": "customer", "content": "Two"}]},
            ]
        }
    ).encode()
    try:
        response = _post_job(
            application.test_client(),
            filename="duplicates.json",
            content=duplicate_json,
        )
        assert response.status_code == 400
        assert b"Duplicate conversation id &#39;same-id&#39;" in response.data
        assert calls == []
    finally:
        application.extensions["support_eval_jobs"].shutdown()
