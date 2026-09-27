from pathlib import Path

from typer.testing import CliRunner

from support_eval.cli import app

PROJECT_ROOT = Path(__file__).parents[1]
runner = CliRunner()


def test_exact_analyze_command(tmp_path):
    result = runner.invoke(
        app,
        [
            "analyze",
            str(PROJECT_ROOT / "examples" / "conversations.csv"),
            "--rubric",
            str(PROJECT_ROOT / "examples" / "rubric.json"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Analyzed 3 conversations" in result.output
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report.html").is_file()


def test_groq_cli_selection_requires_environment_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(PROJECT_ROOT / "examples" / "smoke-conversation.json"),
            "--rubric",
            str(PROJECT_ROOT / "examples" / "rubric.json"),
            "--evaluator",
            "groq",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "GROQ_API_KEY is required" in result.output
    assert not (tmp_path / "results.json").exists()
