import json
from pathlib import Path

from support_eval.checkpoint import CHECKPOINT_FILENAME
from support_eval.pipeline import analyze

PROJECT_ROOT = Path(__file__).parents[1]


def test_pipeline_writes_both_reports(tmp_path):
    report = analyze(
        PROJECT_ROOT / "examples" / "conversations.csv",
        PROJECT_ROOT / "examples" / "rubric.json",
        tmp_path,
    )
    assert report.summary.conversation_count == 3
    assert report.summary.flagged_count == 1
    assert [result.flagged for result in report.conversations] == [False, True, False]
    json_path = tmp_path / "results.json"
    html_path = tmp_path / "report.html"
    assert json_path.is_file()
    assert html_path.is_file()
    assert not (tmp_path / CHECKPOINT_FILENAME).exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["conversation_count"] == 3


def test_pipeline_is_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    arguments = (
        PROJECT_ROOT / "examples" / "conversations.csv",
        PROJECT_ROOT / "examples" / "rubric.json",
    )
    analyze(*arguments, first_dir)
    analyze(*arguments, second_dir)
    assert (first_dir / "results.json").read_text(encoding="utf-8") == (
        second_dir / "results.json"
    ).read_text(encoding="utf-8")
