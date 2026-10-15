# Security Policy

## Supported versions

tulpa is pre-1.0. Security fixes are provided for the latest release and the
current default branch.

| Version | Supported |
| --- | --- |
| latest release | yes |
| current default branch | yes |
| older snapshots | no |

## Reporting a vulnerability

Do not open a public issue containing vulnerability details, credentials,
private conversation data, audio, provider responses, or exploit code.

Use a private GitHub Security Advisory for the repository. If advisories are not
yet enabled, contact the maintainers through a private method shown on their
GitHub profiles and ask for a secure reporting channel without including exploit
details in the first public message.

Include when available:

- affected version, tag, or commit;
- affected endpoint, file, and configuration;
- reproduction steps or proof of concept;
- expected impact and required privileges;
- whether source messages, Facts, prompts, credentials, audio, or diagnostics
  can be exposed;
- a safe way to contact you for follow-up.

Maintainers will acknowledge a complete report as soon as practical, normally
within 7 days. The target coordinated disclosure window is 90 days from
acknowledgement. A fix may ship sooner; the date may be adjusted with the
reporter when remediation or downstream coordination requires it.

## Scope notes

The following deserve special care in tulpa:

- AI service keys and other secrets the desktop app stores in its data
  directory;
- conversation history, memories, screen and computer-audio perception, and
  microphone audio;
- full LLM prompts/responses kept in the local call logs;
- the local service API, which listens on `127.0.0.1` and is protected by a
  per-installation key;
- imported Live2D models, plugins, and Agent integrations that run or load
  third-party content.

## Out of scope

- social engineering or denial-of-service traffic that risks other users;
- reports based only on modified builds or a local service deliberately exposed
  beyond localhost;
- leaked test credentials that are already revoked and cannot access data;
- model hallucinations without a security boundary impact.

Good-faith research that follows this policy will be handled through coordinated
disclosure. Do not access, retain, or disclose data beyond what is necessary to
demonstrate the issue.
