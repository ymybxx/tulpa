# Hermes Agent task setup

tulpa can delegate long-running work to an external Hermes Profile and bring
the result or a required user interaction back through normal Messages and voice.
Hermes remains a separate application: start the Hermes Desktop **Messaging Gateway /
API Server** yourself. tulpa does not install, launch, restart, or supervise it.

Check the endpoint before configuring it:

```bash
export HERMES_API_KEY=replace-with-a-current-key
curl -sS http://127.0.0.1:8642/health
curl -sS \
  -H "Authorization: Bearer $HERMES_API_KEY" \
  http://127.0.0.1:8642/v1/capabilities
```

This integration uses the Runs control plane beginning at `/v1/runs`. The
OpenAI-compatible Chat endpoint exposed by the same API Server is not a Run
submission, status, or approval API.

Users configure one complete Agent URL and API key. The default Hermes profile uses
`http://127.0.0.1:8642`; a multiplexed profile can be addressed directly as
`http://127.0.0.1:8642/p/coder`. Profile names, route prefixes, credential references,
and Target IDs remain internal implementation details.

```dotenv
FISH_MEMORY_AGENT_TASKS_ENABLED=true
FISH_MEMORY_AGENT_URL=http://127.0.0.1:8642
FISH_MEMORY_AGENT_API_KEY=replace-with-a-current-key
# Optional; defaults to five minutes.
FISH_MEMORY_AGENT_RUN_STALE_TIMEOUT_SECONDS=300
```

The stale timeout measures how long tulpa has gone without a trustworthy
provider status. Before failing a stale Run, tulpa performs one final
authoritative `GET /v1/runs/{id}` when an external Run ID is available. A returned
`running` status refreshes the Run and does not kill legitimate long work;
`waiting_user` is also excluded while the Agent is waiting for the user. If the final
status request still fails or remains unknown, the Run becomes failed so it cannot
hold a worker slot indefinitely.

The Fish management UI inherits this global configuration by default and can override
the URL and key. A same-origin URL may reuse the global key; a different host or port
requires a new key. Fish-specific keys are stored in an internal mode-0600 secrets file,
never in the ordinary database or an API response.
An override applies only to newly created logical Tasks; existing Tasks retain their
original internal Target and Hermes Session. Global credentials are retained by
normalized URL in the same internal secrets file so those Tasks survive a global URL
change. Explicitly clearing the Desktop Agent key, or removing
`FISH_MEMORY_AGENT_API_KEY`, revokes all retained global credentials; Tasks that depend
on them can no longer continue.

Never put a real key in documentation, logs, or ordinary database configuration.
Rotate any key pasted into a chat, screenshot, or other untrusted location.

