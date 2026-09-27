# Security policy

## Supported versions

Support Eval is currently a `0.x` project. Security fixes are applied to the latest released version
and the current default branch; older releases do not receive guaranteed backports.

## Reporting a vulnerability

Once the public repository exists, enable GitHub private vulnerability reporting. Use the
repository's **Security → Report a vulnerability** flow to report suspected vulnerabilities
privately. No security email is currently published, and one should not be invented.

Please include a concise description, affected version or commit, reproduction steps, and likely
impact. Allow a reasonable opportunity to investigate and release a fix before public disclosure.

Do not place API keys, credentials, real customer conversations, regulated data, or sensitive
reproduction artifacts in public issues. Use synthetic or redacted examples.

## Deployment boundary

The current Flask application is a localhost-only development server bound to `127.0.0.1`. It has
no authentication and must not be exposed directly to the public internet. Groq evaluation sends
normalized conversation content, the rubric, and included conversation metadata to a third-party
processor; review the privacy guidance in the README before processing customer data.
