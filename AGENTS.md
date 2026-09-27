# Support Eval Development Guide

## Product goal

Build a useful, open-source customer-support QA tool in a seven-day MVP. It accepts CSV or JSON conversations and a custom QA rubric, normalizes the input, evaluates each conversation, calculates deterministic scores and flags, and produces readable JSON and self-contained HTML reports.

## MVP boundaries

- Target Python 3.11 or newer.
- Build a simple modular monolith with a local CLI, thin local web UI, and reusable core evaluation engine.
- Keep dependencies minimal and justify every runtime dependency.
- Keep the fake deterministic evaluator as the default for tests and offline use.
- Keep Groq as the only real LLM evaluator for the public MVP, behind the evaluator interface and configured only by environment variables.
- Keep scoring and flagging deterministic and separate from evaluator output.
- Validate evaluator results before scoring or reporting them.
- Treat the canonical conversation and evaluation models as boundaries between loading, evaluation, scoring, and reporting.
- Support local CSV and JSON conversation inputs and a user-provided rubric.
- Produce machine-readable JSON and a self-contained HTML report.
- Prefer clear, testable code over abstractions intended only for hypothetical future needs.

## Intended flow

`CLI or local web UI -> load conversations and rubric -> normalize to canonical models -> evaluator interface -> validate evaluation -> deterministic scoring and flags -> JSON and HTML reports`

## Explicitly out of scope for the MVP

- Hosted SaaS backend
- Databases or persistent project storage
- Authentication, authorization, or user management
- Billing, subscriptions, or usage metering
- Third-party support-platform integrations
- Scheduled evaluations or distributed background-job infrastructure
- Team collaboration, alerts, or notifications
- Additional LLM providers beyond Groq
- Microservices, message queues, vector databases, or Kubernetes
- Unnecessary frameworks or speculative infrastructure

## Development rules

- Preserve the CLI/core boundary so a hosted product can reuse the engine later without building SaaS concerns now.
- Add only work needed for the documented MVP acceptance path.
- Do not introduce deferred SaaS features behind placeholders, stubs, or premature interfaces.
- Keep modules focused, but avoid excessive layering and indirection.
- Make local runs reproducible and deterministic.
- Fail with actionable messages for malformed input, invalid rubrics, and invalid evaluator output.
- Add tests in proportion to risk, prioritizing normalization, validation, scoring, flags, and report generation.
- Keep reports local and self-contained; do not require external assets or network access.
- Never hardcode, persist, or print provider API keys.
- Update product documentation when scope or behavior changes.

## Scope check for every change

Before implementation, confirm that the change directly supports input loading, normalization, rubric evaluation, validation, scoring, flagging, reporting, local CLI/web usability, or tests/documentation for those capabilities. If it does not, defer it.
