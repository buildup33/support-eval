# Contributing to Support Eval

Support Eval is a small Python 3.11+ modular monolith. Keep changes focused, readable, and within the
documented local MVP scope.

## Development setup

Create and activate a Python 3.11+ virtual environment, then run:

```console
python -m pip install -e ".[dev]"
```

Run the complete checks before opening a pull request:

```console
pytest
ruff check .
ruff format --check .
```

Tests must make zero real provider requests. Use mocked HTTP responses or `FakeEvaluator`, and never
put API keys or customer conversations in fixtures, logs, issues, or commits.

## Architecture map

```text
CLI / local Flask UI
        ↓
ingestion.py → canonical models.py
        ↓
evaluator.py / groq.py
        ↓
scoring.py → checkpoint.py
        ↓
reporting.py → web review / JSON / HTML
```

`pipeline.py` orchestrates the shared workflow used by both entry points. `transcript.py` owns stable
message IDs and evidence resolution. `benchmark.py` compares generated results with frozen human
expectations.

Scoring, weights, thresholds, flags, and evidence-message validation must remain deterministic
application behavior. Provider output must be validated before scoring. Evaluator or prompt changes
need mocked regression tests and careful review against the benchmark contract; do not rewrite
benchmark or holdout expectations to fit model output.

## Pull request checklist

- The change stays within the documented scope.
- Tests cover new or changed behavior and make no real API calls.
- `pytest`, `ruff check .`, and `ruff format --check .` pass.
- Evaluator output remains validated before deterministic scoring.
- No secrets, customer data, generated reports, jobs, or checkpoints are included.
- User-facing behavior or data flow is documented when it changes.
