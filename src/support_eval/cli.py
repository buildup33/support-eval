from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from support_eval.benchmark import compare_files
from support_eval.checkpoint import CheckpointError
from support_eval.groq import GroqError, GroqEvaluator
from support_eval.ingestion import InputError
from support_eval.pipeline import analyze
from support_eval.scoring import EvaluationValidationError

app = typer.Typer(help="Evaluate support conversations against a local QA rubric.")


class EvaluatorChoice(StrEnum):
    fake = "fake"
    groq = "groq"


@app.callback()
def main() -> None:
    """Run local support conversation quality evaluations."""


@app.command("analyze")
def analyze_command(
    conversations: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="CSV or JSON conversations"
        ),
    ],
    rubric: Annotated[
        Path,
        typer.Option("--rubric", "-r", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", file_okay=False, help="Report output directory"),
    ] = Path("."),
    evaluator: Annotated[
        EvaluatorChoice,
        typer.Option("--evaluator", help="Evaluation provider: fake or groq"),
    ] = EvaluatorChoice.fake,
) -> None:
    """Analyze conversations and write results.json and report.html."""
    try:
        selected_evaluator = GroqEvaluator() if evaluator is EvaluatorChoice.groq else None
        report = analyze(conversations, rubric, output_dir, evaluator=selected_evaluator)
    except (
        GroqError,
        CheckpointError,
        InputError,
        EvaluationValidationError,
        ValidationError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Analyzed {report.summary.conversation_count} conversations; "
        f"flagged {report.summary.flagged_count}."
    )
    typer.echo(f"Wrote {output_dir / 'results.json'}")
    typer.echo(f"Wrote {output_dir / 'report.html'}")


@app.command("compare-benchmark")
def compare_benchmark_command(
    results: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Generated results.json"),
    ],
    expected: Annotated[
        Path,
        typer.Option("--expected", "-e", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Compare generated results with human benchmark expectations."""
    try:
        comparison = compare_files(results, expected)
    except (ValueError, ValidationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        "Criterion agreement: "
        f"{comparison.criterion_agreements}/{comparison.total_criteria} "
        f"({comparison.criterion_agreement_percentage:.2f}%)"
    )
    typer.echo(f"False passes: {comparison.false_passes}")
    typer.echo(f"False failures: {comparison.false_failures}")
    typer.echo(f"Uncertain disagreements: {comparison.uncertain_disagreements}")
    typer.echo(
        f"Flag agreement: {comparison.flag_agreements}/{comparison.total_flags} "
        f"({comparison.flag_agreement_percentage:.2f}%)"
    )
    if comparison.disagreements_by_conversation:
        typer.echo("Disagreements by conversation:")
        for conversation_id, items in comparison.disagreements_by_conversation.items():
            typer.echo(f"  {conversation_id}")
            for item in items:
                typer.echo(f"    - {item}")
    else:
        typer.echo("Disagreements by conversation: none")
