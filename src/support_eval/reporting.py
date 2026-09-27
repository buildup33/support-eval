from __future__ import annotations

import html
from pathlib import Path

from support_eval.models import ReportResult


def _atomic_write_text(destination: Path, content: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_json_report(report: ReportResult, path: str | Path) -> Path:
    destination = Path(path)
    return _atomic_write_text(destination, report.model_dump_json(indent=2) + "\n")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_html_report(report: ReportResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for conversation in report.conversations:
        status = "Flagged" if conversation.flagged else "Pass"
        css_class = "flagged" if conversation.flagged else "pass"
        flags = "; ".join(conversation.flags) or "None"
        rows.append(
            "<tr>"
            f'<td><a href="#conversation-{_escape(conversation.conversation_id)}">'
            f"{_escape(conversation.conversation_id)}</a></td>"
            f"<td>{conversation.score:.2f}</td>"
            f'<td><span class="badge {css_class}">{status}</span></td>'
            f"<td>{_escape(flags)}</td>"
            "</tr>"
        )
        criterion_rows = []
        for result in conversation.evaluations:
            evidence = (
                "; ".join(
                    f"{reference.message_id} ({reference.role}): {reference.text}"
                    for reference in result.evidence
                )
                or "None"
            )
            rating = f"{result.rating:.2f}" if result.rating is not None else "Derived"
            criterion_rows.append(
                "<tr>"
                f"<td>{_escape(result.criterion_id)}</td>"
                f"<td>{_escape(result.status)}</td>"
                f"<td>{rating}</td>"
                f"<td>{result.confidence:.2f}</td>"
                f"<td>{_escape(result.rationale)}</td>"
                f"<td>{_escape(evidence)}</td>"
                "</tr>"
            )
        details.append(
            f'<section id="conversation-{_escape(conversation.conversation_id)}">'
            f"<h2>{_escape(conversation.conversation_id)} — {conversation.score:.2f}</h2>"
            "<table><thead><tr><th>Criterion</th><th>Result</th><th>Rating</th>"
            "<th>Confidence</th><th>Rationale</th><th>Evidence</th></tr></thead><tbody>"
            + "".join(criterion_rows)
            + "</tbody></table></section>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support Eval Report</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
    .card {{ background: #f3f4f6; border-radius: .5rem; padding: 1rem; min-width: 10rem; }}
    .card strong {{ display: block; font-size: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: .65rem; text-align: left; }}
    th {{ background: #f9fafb; }}
    .badge {{ border-radius: 999px; color: white; padding: .2rem .55rem; font-size: .85rem; }}
    .flagged {{ background: #b91c1c; }} .pass {{ background: #047857; }}
    a {{ color: #1d4ed8; }} section {{ scroll-margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Support Eval Report</h1>
  <p>Rubric: <strong>{_escape(report.rubric_name)}</strong><br>
     Source: <code>{_escape(report.source_file)}</code></p>
  <div class="summary">
    <div class="card"><strong>{report.summary.conversation_count}</strong>Conversations</div>
    <div class="card"><strong>{report.summary.flagged_count}</strong>Flagged</div>
    <div class="card"><strong>{report.summary.average_score:.2f}</strong>Average score</div>
  </div>
  <h2>Overview</h2>
  <table><thead><tr><th>Conversation</th><th>Score</th><th>Status</th><th>Flags</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
  {"".join(details)}
</body>
</html>
"""
    return _atomic_write_text(destination, document)
