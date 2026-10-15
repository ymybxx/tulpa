import path from "node:path";

import type { ComputerHistoryInteractionEvent } from "./perception-manager";

/**
 * Persisted in place of recursive AX/selection text. Keeping a small semantic
 * marker lets the summarizer retain the fact that the user inspected Computer
 * History without feeding the history file back into itself.
 */
export const COMPUTER_HISTORY_SELF_OBSERVATION_MARKER =
  "[Fish Memory: own Computer History content omitted]";

interface ObservationContext {
  app: string;
  window: string;
}

/**
 * Guards only high-confidence observations of this recorder's own JSONL.
 *
 * A filename alone is intentionally insufficient: developers commonly edit
 * unrelated files named events.jsonl. Detection requires the Fish event
 * envelope plus either the managed root path or a known macOS viewer focused
 * on that filename. Once confirmed, the context is remembered so a subsequent
 * short selection does not reintroduce the file body.
 */
export class ComputerHistorySelfObservationGuard {
  private readonly normalizedRootPath: string;
  private activeContext: ObservationContext | null = null;

  constructor(rootPath: string) {
    this.normalizedRootPath = normalizePathForDetection(path.resolve(rootPath));
  }

  reset(): void {
    this.activeContext = null;
  }

  restoreFromRecentEvents(eventsNewestFirst: readonly ComputerHistoryInteractionEvent[]): void {
    this.activeContext = null;
    let recentContext: ObservationContext | null = null;
    for (const event of eventsNewestFirst) {
      const context = observationContext(event);
      if (!context) break;
      if (!recentContext) recentContext = context;
      if (!sameContext(recentContext, context)) break;
      if (hasRedactionMarker(event) || this.isManagedHistoryObservation(event)) {
        this.activeContext = recentContext;
        return;
      }
    }
  }

  sanitize(event: ComputerHistoryInteractionEvent): ComputerHistoryInteractionEvent {
    const context = observationContext(event);
    if (!context || (this.activeContext && !sameContext(this.activeContext, context))) {
      this.activeContext = null;
    }

    const directlyDetected = this.isManagedHistoryObservation(event);
    if (directlyDetected && context) this.activeContext = context;
    const shouldRedact = directlyDetected
      || Boolean(context && this.activeContext && sameContext(this.activeContext, context));
    if (!shouldRedact) return event;

    return {
      ...event,
      ...(event.ax
        ? {
            ax: {
              ...event.ax,
              text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
            },
          }
        : {}),
      ...(event.selection
        ? {
            selection: {
              truncated: event.selection.truncated,
              text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
              ...(event.selection.target
                ? { target: { ...event.selection.target } }
                : {}),
            },
          }
        : {}),
    };
  }

  private isManagedHistoryObservation(event: ComputerHistoryInteractionEvent): boolean {
    if (hasRedactionMarker(event)) return true;
    const body = observationBody(event);
    if (!hasStoredFishEventEnvelope(body)) return false;
    return this.containsManagedEventsPath(observationLocationAndBody(event, body))
      || (isKnownMacFileViewer(event) && focusesEventsJsonl(event));
  }

  private containsManagedEventsPath(value: string): boolean {
    const normalized = normalizePathForDetection(value);
    let index = normalized.indexOf(this.normalizedRootPath);
    while (index >= 0) {
      const suffix = normalized.slice(
        index + this.normalizedRootPath.length,
        index + this.normalizedRootPath.length + 512,
      );
      if (/^\/(?:[^/\s]+\/)*events\.jsonl(?:[^a-z0-9._-]|$)/u.test(suffix)) return true;
      index = normalized.indexOf(this.normalizedRootPath, index + 1);
    }
    return false;
  }
}

function observationBody(event: ComputerHistoryInteractionEvent): string {
  return [event.ax?.text, event.selection?.text]
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .join("\n");
}

function observationLocationAndBody(
  event: ComputerHistoryInteractionEvent,
  body: string,
): string {
  return [
    event.window.url,
    event.window.title,
    event.mouse?.target?.title,
    event.mouse?.target?.description,
    event.keyboard?.target?.title,
    event.keyboard?.target?.description,
    event.selection?.target?.title,
    event.selection?.target?.description,
    body,
  ]
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .join("\n");
}

function observationContext(event: ComputerHistoryInteractionEvent): ObservationContext | null {
  const app = (event.app.bundleIdentifier ?? event.app.name)?.trim().toLocaleLowerCase();
  const window = (event.window.url ?? event.window.title)?.trim().toLocaleLowerCase();
  if (!app || !window) return null;
  return {
    app,
    window: normalizeEventsWindow(window),
  };
}

function normalizeEventsWindow(value: string): string {
  if (/(?:^|[/\\])events\.jsonl(?:\s+[—-]\s+[^/\\]+)?$/iu.test(value)) {
    return "events.jsonl";
  }
  return value;
}

function sameContext(left: ObservationContext, right: ObservationContext): boolean {
  return left.app === right.app && left.window === right.window;
}

function isKnownMacFileViewer(event: ComputerHistoryInteractionEvent): boolean {
  const bundleIdentifier = event.app.bundleIdentifier?.toLocaleLowerCase();
  if (bundleIdentifier === "com.apple.textedit" || bundleIdentifier === "com.apple.finder") {
    return true;
  }
  const name = event.app.name?.trim().toLocaleLowerCase();
  return name === "textedit" || name === "文本编辑" || name === "finder" || name === "访达";
}

function focusesEventsJsonl(event: ComputerHistoryInteractionEvent): boolean {
  const candidates = [
    event.window.title,
    event.window.url,
    event.mouse?.target?.title,
    event.mouse?.target?.description,
    event.keyboard?.target?.title,
    event.keyboard?.target?.description,
    event.selection?.target?.title,
    event.selection?.target?.description,
  ];
  return candidates.some((candidate) =>
    typeof candidate === "string"
    && /(?:^|[/\\])events\.jsonl(?:\s+[—-]\s+[^/\\]+)?$/iu.test(candidate.trim()));
}

function hasRedactionMarker(event: ComputerHistoryInteractionEvent): boolean {
  return event.ax?.text === COMPUTER_HISTORY_SELF_OBSERVATION_MARKER
    || event.selection?.text === COMPUTER_HISTORY_SELF_OBSERVATION_MARKER;
}

function hasStoredFishEventEnvelope(value: string): boolean {
  if (!value) return false;
  // AX serialization can contain either literal JSON or a JSON string copied
  // from an earlier recursive capture. Normalize only quoting for detection;
  // the original content is never parsed or persisted by this helper.
  const normalized = value.replace(/\\+"/gu, "\"");
  return /"schemaVersion"\s*:\s*1\s*,\s*"ordinal"\s*:\s*[1-9]\d*\s*,\s*"id"\s*:\s*"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"\s*,\s*"timestamp"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z"\s*,\s*"kind"\s*:\s*"(?:session\.started|window\.changed|mouse\.click|mouse\.drag|mouse\.context_menu|selection\.changed|keyboard\.shortcut|keyboard\.text_input|keyboard\.submit)"\s*,\s*"app"\s*:\s*\{[^{}]{0,1024}\}\s*,\s*"window"\s*:\s*\{/iu.test(normalized);
}

function normalizePathForDetection(value: string): string {
  let decoded = value;
  try {
    decoded = decodeURIComponent(value);
  } catch {
    // Keep malformed untrusted AX text literal; detection must never make
    // recording fail.
  }
  return decoded
    .replace(/file:\/\//giu, "")
    .replace(/\\/gu, "/")
    .replace(/\/{2,}/gu, "/")
    .toLocaleLowerCase();
}
