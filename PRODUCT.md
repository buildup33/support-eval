# Support Eval

## Product summary

Support Eval is an open-source tool, with a possible future SaaS product, for evaluating customer-support conversations against a team's own quality-assurance rubric. The MVP turns CSV or JSON conversation exports into consistent scores, flags, and readable JSON and HTML reports.

## Target user

The initial user is a support lead, QA owner, support operations practitioner, or small-team operator who needs repeatable conversation reviews but cannot justify a complex enterprise QA platform. The local web UI makes the workflow accessible in a browser, while the CLI remains available for technical and automated workflows.

## Problem

Manual support QA is slow, inconsistent, and difficult to repeat across many conversations. Existing workflows often rely on spreadsheets, ad hoc sampling, or platforms that are too expensive or rigid for small teams. Teams need a transparent way to apply their own rubric, identify problematic conversations, and inspect how scores were derived.

## Seven-day MVP

The MVP provides one complete local workflow:

1. Load customer-support conversations from CSV or JSON and load a custom rubric from the CLI or local web UI.
2. Normalize source data into canonical conversation models.
3. Evaluate conversations through a replaceable evaluator interface, using the default fake deterministic evaluator or the optional Groq evaluator.
4. Validate evaluation output before it affects results.
5. Apply deterministic scoring and flagging rules.
6. Write a machine-readable JSON report and a self-contained, human-readable HTML report, with native local web result views.

The MVP should be straightforward to install and run with Python 3.11+, remain deterministic without network access when using the fake evaluator, and be useful for inspecting the full evaluation pipeline. It does not claim production-grade semantic judgment while using the fake evaluator.

## Open-source product

The open-source edition is the foundation of the product: a genuinely useful local web UI, Python CLI, and reusable core evaluation engine. It owns input loading, canonical models, the evaluator contract, output validation, scoring, flags, evidence, and local reports. Its value should not depend on a hosted account or paid service.

## Future SaaS product

A future hosted edition may reuse the core engine while adding managed execution, saved history and projects, scheduled or continuous QA, support-platform integrations, team workflows, alerts, and billing. These are product directions, not MVP commitments, and the local engine should not be distorted by speculative SaaS architecture.

## Explicitly deferred

The following are not part of the seven-day MVP:

- Hosted web application or hosted API (the browser UI is local-only)
- Saved projects, database persistence, or cross-run history
- Authentication, authorization, accounts, or organizations
- Billing, subscriptions, quotas, or usage metering
- Scheduled jobs or distributed background workers
- Support-platform or other third-party integrations
- Team assignment, collaboration, comments, or audit workflows
- Alerts and notifications
- Additional LLM providers beyond Groq or guarantees of production AI evaluation quality
- Model training, embeddings, semantic search, or vector databases
- Microservices, distributed infrastructure, queues, or Kubernetes
- Enterprise administration, compliance, or advanced analytics

## MVP success criteria

- A user can run one local command against supported conversation data and a rubric.
- Valid inputs produce deterministic scores and flags for each conversation.
- Invalid inputs or evaluator results fail clearly rather than silently corrupting scores.
- JSON output is suitable for downstream tooling.
- The HTML report is readable offline and requires no external assets.
- The core workflow remains understandable and maintainable by one developer.
