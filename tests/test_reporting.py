from pathlib import Path

from support_eval.models import (
    Conversation,
    ConversationResult,
    CriterionEvaluation,
    EvaluationResult,
    EvidenceReference,
    Message,
    ReportResult,
    ReportSummary,
    Rubric,
    RubricCriterion,
)
from support_eval.reporting import write_html_report, write_json_report
from support_eval.scoring import score_evaluation


def test_reports_escape_html_and_are_self_contained(tmp_path):
    report = ReportResult(
        rubric_name="<Quality>",
        source_file="input.csv",
        summary=ReportSummary(conversation_count=1, flagged_count=1, average_score=0),
        conversations=[
            ConversationResult(
                conversation_id="ticket<script>",
                score=0,
                flagged=True,
                flags=["Needs <review>"],
                evaluations=[],
            )
        ],
    )
    json_path = write_json_report(report, tmp_path / "results.json")
    html_path = write_html_report(report, tmp_path / "report.html")
    html = html_path.read_text(encoding="utf-8")
    assert json_path.is_file()
    assert "&lt;Quality&gt;" in html
    assert "ticket&lt;script&gt;" in html
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html


def test_final_report_writes_replace_temporary_files_atomically(tmp_path, monkeypatch):
    replaced: list[tuple[str, str]] = []
    original_replace = Path.replace

    def track_replace(path, target):
        replaced.append((path.name, Path(target).name))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", track_replace)
    report = ReportResult(
        rubric_name="Atomic",
        source_file="fixture.json",
        summary=ReportSummary(conversation_count=0, flagged_count=0, average_score=0),
        conversations=[],
    )

    write_json_report(report, tmp_path / "results.json")
    write_html_report(report, tmp_path / "report.html")

    assert replaced == [
        ("results.json.tmp", "results.json"),
        ("report.html.tmp", "report.html"),
    ]
    assert not (tmp_path / "results.json.tmp").exists()
    assert not (tmp_path / "report.html.tmp").exists()


def test_report_uses_application_resolved_transcript_evidence(tmp_path):
    conversation = Conversation(
        id="traceable",
        messages=[Message(role="agent", content="The verified transcript sentence.")],
    )
    rubric = Rubric(
        name="Traceable evidence",
        criteria=[
            RubricCriterion(
                id="trace",
                name="Trace",
                description="Evidence must be traceable.",
                weight=1,
                pass_threshold=1,
                keywords=["verified"],
            )
        ],
    )
    evaluation = EvaluationResult(
        conversation_id="traceable",
        criteria=[
            CriterionEvaluation(
                criterion_id="trace",
                status="pass",
                rationale="The message is referenced.",
                evidence=[EvidenceReference(message_id="m0001")],
                confidence=1,
            )
        ],
    )
    scored = score_evaluation(evaluation, rubric, conversation)
    report = ReportResult(
        rubric_name=rubric.name,
        source_file="fixture.json",
        summary=ReportSummary(conversation_count=1, flagged_count=0, average_score=100),
        conversations=[scored],
    )

    json_path = write_json_report(report, tmp_path / "results.json")
    html_path = write_html_report(report, tmp_path / "report.html")

    payload = json_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert '"message_id": "m0001"' in payload
    assert '"role": "agent"' in payload
    assert '"text": "The verified transcript sentence."' in payload
    assert "m0001 (agent): The verified transcript sentence." in html
