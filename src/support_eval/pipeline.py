from __future__ import annotations

from pathlib import Path

from support_eval.checkpoint import (
    CHECKPOINT_FILENAME,
    build_checkpoint_identity,
    load_checkpoint,
    write_checkpoint,
)
from support_eval.evaluator import Evaluator, FakeEvaluator
from support_eval.ingestion import load_conversations, load_rubric
from support_eval.models import ReportResult, ReportSummary
from support_eval.reporting import write_html_report, write_json_report
from support_eval.scoring import score_evaluation


def analyze(
    conversations_path: str | Path,
    rubric_path: str | Path,
    output_dir: str | Path = ".",
    evaluator: Evaluator | None = None,
) -> ReportResult:
    conversations = load_conversations(conversations_path)
    rubric = load_rubric(rubric_path)
    active_evaluator = evaluator or FakeEvaluator()
    destination = Path(output_dir)
    checkpoint_path = destination / CHECKPOINT_FILENAME
    identity = build_checkpoint_identity(conversations, rubric, active_evaluator)
    saved_results, migrated = load_checkpoint(checkpoint_path, identity, conversations, rubric)
    if migrated:
        write_checkpoint(checkpoint_path, identity, saved_results)
    results_by_id = {result.conversation_id: result for result in saved_results}

    for conversation in conversations:
        if conversation.id in results_by_id:
            continue
        evaluation = active_evaluator.evaluate(conversation, rubric)
        results_by_id[conversation.id] = score_evaluation(evaluation, rubric, conversation)
        completed = [results_by_id[item.id] for item in conversations if item.id in results_by_id]
        write_checkpoint(checkpoint_path, identity, completed)

    results = [results_by_id[conversation.id] for conversation in conversations]
    flagged_count = sum(result.flagged for result in results)
    average_score = round(sum(result.score for result in results) / len(results), 2)
    report = ReportResult(
        rubric_name=rubric.name,
        source_file=str(Path(conversations_path)),
        summary=ReportSummary(
            conversation_count=len(results),
            flagged_count=flagged_count,
            average_score=average_score,
        ),
        conversations=results,
    )
    write_json_report(report, destination / "results.json")
    write_html_report(report, destination / "report.html")
    checkpoint_path.unlink(missing_ok=True)
    return report
