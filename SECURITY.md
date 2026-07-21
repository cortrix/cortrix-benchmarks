# Security Policy

## Supported versions

Before the first public release, security support applies to the latest commit on the default branch only. Other branches, tags, releases, and older commits are not supported. This policy will be replaced with an explicit supported-version table before the first release.

## Reporting a vulnerability

Report suspected vulnerabilities privately to [security@cortrix.ai](mailto:security@cortrix.ai). Do not open a public benchmark challenge or other issue for an unpatched vulnerability, credentials, private infrastructure details, or sensitive logs.

The Cortrix Security Response Team will acknowledge receipt within 5 business days. An acknowledgment confirms intake and initial routing; it is not a remediation or release deadline.

Include the affected 40-character repository commit, runner or bundle identity, impact, minimal reproduction, and sanitized evidence. Do not attach restricted datasets, secrets, model caches, raw runtime databases, or sensitive logs.

Never commit API keys, tokens, `.env` files, cloud credentials, dataset caches, raw runs, model caches, runtime databases, or logs.

## Coordinated disclosure

Do not disclose the vulnerability publicly before a fix is available or a mutually agreed disclosure date is reached. The response team will coordinate severity assessment, remediation status, and disclosure timing with the reporter.
