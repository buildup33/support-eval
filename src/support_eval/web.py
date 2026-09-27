from __future__ import annotations

import json
import os
import secrets
from collections import Counter
from contextlib import suppress
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlsplit

from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from support_eval.groq import DEFAULT_MODEL
from support_eval.ingestion import InputError, load_conversations, load_rubric
from support_eval.models import ReportResult
from support_eval.transcript import transcript_messages
from support_eval.web_jobs import JobManager

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAMPLE_CSV = """conversation_id,role,message,timestamp
example-1,customer,I cannot access my account,2026-09-20T09:00:00Z
example-1,agent,I understand. I reset the login lock and verified access.,2026-09-20T09:02:00Z
"""
DEFAULT_RUBRIC = {
    "name": "Customer Support Quality",
    "flag_threshold": 70,
    "criteria": [
        {
            "id": "empathy",
            "name": "Empathy",
            "description": "Acknowledges the customer's experience when empathy is appropriate.",
            "weight": 1,
            "pass_threshold": 1,
            "keywords": ["understand", "sorry"],
            "flag_on_fail": False,
        },
        {
            "id": "accuracy",
            "name": "Accuracy",
            "description": (
                "Material claims and guidance are supported, correct, or properly qualified."
            ),
            "weight": 2,
            "pass_threshold": 1,
            "keywords": ["confirmed", "verified"],
            "flag_on_fail": True,
        },
        {
            "id": "resolution",
            "name": "Resolution",
            "description": "Resolves the issue or provides a concrete, owned next step.",
            "weight": 2,
            "pass_threshold": 1,
            "keywords": ["resolved", "follow up"],
            "flag_on_fail": True,
        },
        {
            "id": "professionalism",
            "name": "Professionalism",
            "description": "Communication is respectful and appropriate throughout.",
            "weight": 1,
            "pass_threshold": 1,
            "keywords": ["please", "help"],
            "flag_on_fail": True,
        },
    ],
}


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=secrets.token_hex(24),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        TRUSTED_HOSTS=["127.0.0.1", "localhost"],
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        JOBS_ROOT=Path.cwd() / ".support-eval-web" / "jobs",
        REQUIRE_GROQ_API_KEY=True,
        EVALUATOR_FACTORY=None,
    )
    if config:
        app.config.update(config)

    manager_kwargs: dict[str, Any] = {}
    if app.config["EVALUATOR_FACTORY"] is not None:
        manager_kwargs["evaluator_factory"] = app.config["EVALUATOR_FACTORY"]
    manager = JobManager(Path(app.config["JOBS_ROOT"]), **manager_kwargs)
    app.extensions["support_eval_jobs"] = manager

    @app.context_processor
    def inject_csrf_token() -> dict[str, Any]:
        return {"csrf_token": _csrf_token()}

    @app.before_request
    def protect_state_changes():  # type: ignore[no-untyped-def]
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None

        origin = request.headers.get("Origin")
        referer = request.headers.get("Referer")
        if origin and not _same_origin(origin, request.host_url):
            abort(403)
        if not origin and referer and not _same_origin(referer, request.host_url):
            abort(403)

        expected = session.get("_csrf_token")
        supplied = request.form.get("_csrf_token", "")
        if not isinstance(expected, str) or not secrets.compare_digest(expected, supplied):
            abort(403)
        return None

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            rubric_json=json.dumps(DEFAULT_RUBRIC, indent=2),
            model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        )

    @app.post("/jobs")
    def create_job():  # type: ignore[no-untyped-def]
        upload = request.files.get("conversations")
        rubric_text = request.form.get("rubric", "")
        if upload is None or not upload.filename:
            flash("Choose a CSV or JSON conversation file.", "error")
            return _render_index(rubric_text), 400
        suffix = Path(upload.filename).suffix.casefold()
        if suffix not in {".csv", ".json"}:
            flash("Conversation files must be CSV or JSON.", "error")
            return _render_index(rubric_text), 400
        if app.config["REQUIRE_GROQ_API_KEY"] and not os.environ.get("GROQ_API_KEY", "").strip():
            flash(
                "GROQ_API_KEY is not set. Add it to this PowerShell session, then try again.",
                "error",
            )
            return _render_index(rubric_text), 400

        job_id = secrets.token_hex(16)
        destination = manager.job_dir(job_id)
        destination.mkdir(parents=True, exist_ok=False)
        conversation_filename = f"conversations{suffix}"
        try:
            upload.save(destination / conversation_filename)
            (destination / "rubric.json").write_text(rubric_text, encoding="utf-8")
            conversations = load_conversations(destination / conversation_filename)
            load_rubric(destination / "rubric.json")
        except InputError as exc:
            _remove_invalid_job(destination)
            flash(str(exc), "error")
            return _render_index(rubric_text), 400
        except OSError:
            _remove_invalid_job(destination)
            flash("The uploaded files could not be saved. Please try again.", "error")
            return _render_index(rubric_text), 500

        manager.create_job(
            job_id,
            display_filename=PurePath(upload.filename).name,
            conversation_filename=conversation_filename,
            total=len(conversations),
            prepared=True,
        )
        manager.submit(job_id)
        return redirect(url_for("job_progress", job_id=job_id))

    @app.get("/jobs/<job_id>")
    def job_progress(job_id: str):  # type: ignore[no-untyped-def]
        job = _load_job_or_404(manager, job_id)
        if job.state == "complete":
            return redirect(url_for("job_results", job_id=job_id))
        return render_template("progress.html", job=job, completed=manager.progress(job))

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):  # type: ignore[no-untyped-def]
        job = _load_job_or_404(manager, job_id)
        completed = manager.progress(job)
        return jsonify(
            id=job.id,
            state=job.state,
            message=job.status_message,
            error=job.error_message,
            completed=completed,
            total=job.total,
            percentage=round((completed / job.total) * 100),
            results_url=(
                url_for("job_results", job_id=job.id) if job.state == "complete" else None
            ),
        )

    @app.post("/jobs/<job_id>/resume")
    def resume_job(job_id: str):  # type: ignore[no-untyped-def]
        job = _load_job_or_404(manager, job_id)
        if job.state != "failed":
            flash("Only a stopped evaluation can be resumed.", "error")
            return redirect(url_for("job_progress", job_id=job_id))
        job.state = "queued"
        job.error_message = None
        job.status_message = "Queued to resume from the saved checkpoint"
        manager.save(job)
        manager.submit(job_id)
        return redirect(url_for("job_progress", job_id=job_id))

    @app.get("/jobs/<job_id>/results")
    def job_results(job_id: str):  # type: ignore[no-untyped-def]
        job = _load_job_or_404(manager, job_id)
        if job.state != "complete":
            return redirect(url_for("job_progress", job_id=job_id))
        report, rubric = _load_results(manager, job_id)
        criterion_summary = _criterion_summary(report, rubric)
        return render_template(
            "results.html",
            job=job,
            report=report,
            rubric=rubric,
            criterion_summary=criterion_summary,
        )

    @app.get("/jobs/<job_id>/conversations/<path:conversation_id>")
    def conversation_detail(job_id: str, conversation_id: str):  # type: ignore[no-untyped-def]
        job = _load_job_or_404(manager, job_id)
        report, rubric = _load_results(manager, job_id)
        result = next(
            (item for item in report.conversations if item.conversation_id == conversation_id),
            None,
        )
        if result is None:
            abort(404)
        conversations = load_conversations(manager.job_dir(job_id) / job.conversation_filename)
        conversation = next(item for item in conversations if item.id == conversation_id)
        criterion_by_id = {item.id: item for item in rubric.criteria}
        return render_template(
            "conversation.html",
            job=job,
            result=result,
            messages=transcript_messages(conversation),
            criterion_by_id=criterion_by_id,
        )

    @app.get("/jobs/<job_id>/download/<artifact>")
    def download_artifact(job_id: str, artifact: str):  # type: ignore[no-untyped-def]
        _load_job_or_404(manager, job_id)
        allowed = {"results.json", "report.html"}
        if artifact not in allowed:
            abort(404)
        return send_from_directory(
            manager.job_dir(job_id),
            artifact,
            as_attachment=True,
            download_name=f"support-eval-{artifact}",
        )

    @app.get("/samples/conversations.csv")
    def sample_conversations():  # type: ignore[no-untyped-def]
        return Response(
            SAMPLE_CSV,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=support-eval-sample.csv"},
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error: RequestEntityTooLarge):  # type: ignore[no-untyped-def]
        return render_template(
            "error.html",
            title="File too large",
            message="Uploads are limited to 10 MB for this local MVP.",
        ), 413

    @app.errorhandler(404)
    def not_found(_error):  # type: ignore[no-untyped-def]
        return render_template(
            "error.html",
            title="Page not found",
            message="This evaluation or page does not exist on this machine.",
        ), 404

    @app.errorhandler(403)
    def forbidden(_error):  # type: ignore[no-untyped-def]
        return render_template(
            "error.html",
            title="Request blocked",
            message="This request did not come from the local Support Eval session.",
        ), 403

    def _render_index(rubric_text: str) -> str:
        return render_template(
            "index.html",
            rubric_json=rubric_text or json.dumps(DEFAULT_RUBRIC, indent=2),
            model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        )

    return app


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not isinstance(token, str) or len(token) < 32:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _same_origin(value: str, local_url: str) -> bool:
    try:
        candidate = urlsplit(value)
        local = urlsplit(local_url)
        candidate_port = candidate.port or (443 if candidate.scheme == "https" else 80)
        local_port = local.port or (443 if local.scheme == "https" else 80)
    except ValueError:
        return False
    return (
        candidate.scheme == local.scheme
        and candidate.hostname == local.hostname
        and candidate_port == local_port
    )


def _load_job_or_404(manager: JobManager, job_id: str):  # type: ignore[no-untyped-def]
    try:
        return manager.load(job_id)
    except (FileNotFoundError, ValueError):
        abort(404)


def _load_results(manager: JobManager, job_id: str):  # type: ignore[no-untyped-def]
    destination = manager.job_dir(job_id)
    try:
        report = ReportResult.model_validate_json(
            (destination / "results.json").read_text(encoding="utf-8")
        )
        rubric = load_rubric(destination / "rubric.json")
    except (OSError, ValueError):
        abort(404)
    return report, rubric


def _criterion_summary(report, rubric):  # type: ignore[no-untyped-def]
    rows = []
    for criterion in rubric.criteria:
        statuses = Counter(
            evaluation.status
            for conversation in report.conversations
            for evaluation in conversation.evaluations
            if evaluation.criterion_id == criterion.id
        )
        total = sum(statuses.values())
        rows.append(
            {
                "id": criterion.id,
                "name": criterion.name,
                "pass": statuses["pass"],
                "fail": statuses["fail"],
                "uncertain": statuses["uncertain"],
                "pass_rate": round((statuses["pass"] / total) * 100) if total else 0,
            }
        )
    return rows


def _remove_invalid_job(destination: Path) -> None:
    for filename in ("conversations.csv", "conversations.json", "rubric.json"):
        (destination / filename).unlink(missing_ok=True)
    with suppress(OSError):
        destination.rmdir()


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
