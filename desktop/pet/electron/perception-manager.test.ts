import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  JsonLineDecoder,
  MAX_VISION_CANDIDATE_JPEG_BYTES,
  parseComputerHistoryInteractionEvent,
  parseComputerHistorySuppressedEvents,
  parsePerceptionEnvelope,
  parseVisionCandidateMetadata,
  parseVisionCandidatePayload,
  PerceptionManager,
  type PerceptionHelperProcess,
  type PerceptionProcessSpawner,
} from "./perception-manager";

afterEach(() => {
  vi.useRealTimers();
});

describe("perception helper protocol", () => {
  it("decodes chunked JSON lines without merging messages", () => {
    const decoder = new JsonLineDecoder();
    expect(decoder.push('{"a":1}\n{"a"')).toEqual(['{"a":1}']);
    expect(decoder.push(':2}\r\n')).toEqual(['{"a":2}']);
  });

  it("preserves UTF-8 characters split across helper chunks", () => {
    const decoder = new JsonLineDecoder();
    const encoded = Buffer.from('{"app":"终端"}\n');
    const split = encoded.indexOf(Buffer.from("终")) + 1;

    expect(decoder.push(encoded.subarray(0, split))).toEqual([]);
    expect(decoder.push(encoded.subarray(split))).toEqual(['{"app":"终端"}']);
  });

  it("rejects unsupported protocol versions", () => {
    expect(() => parsePerceptionEnvelope(JSON.stringify({
      protocolVersion: 2,
      sequence: 1,
      sentAt: 10,
      event: "hello",
      body: {},
    }))).toThrow("Unsupported perception protocol version");
  });

  it("strictly parses candidate identity, app and five-second timestamps", () => {
    const candidate = visionMetadata();
    expect(parseVisionCandidateMetadata(candidate)).toEqual(candidate);
    expect(() => parseVisionCandidateMetadata({
      ...candidate,
      candidateId: "not-a-uuid",
    })).toThrow("Invalid vision candidate metadata");
    expect(() => parseVisionCandidateMetadata({
      ...candidate,
      appBundleId: "COM.Example.App",
    })).toThrow("Invalid vision candidate metadata");
    expect(() => parseVisionCandidateMetadata({
      ...candidate,
      expiresAt: candidate.observedAt + 6,
    })).toThrow("Invalid vision candidate metadata");
    expect(() => parseVisionCandidateMetadata({
      ...candidate,
      extra: true,
    })).toThrow("Invalid vision candidate metadata");
  });

  it("strictly parses a bounded JPEG payload", () => {
    const candidate = visionMetadata();
    const jpegBase64 = jpegCandidateBase64();
    expect(parseVisionCandidatePayload({
      ...candidate,
      mimeType: "image/jpeg",
      jpegBase64,
    })).toEqual({ ...candidate, mimeType: "image/jpeg", jpegBase64 });
    expect(() => parseVisionCandidatePayload({
      ...candidate,
      mimeType: "image/png",
      jpegBase64,
    })).toThrow("Invalid vision candidate payload");
  });

  it("strictly parses typed computer-history interactions including a 64 KiB AX tree", () => {
    const value = interactionEvent({
      ax: { mode: "fullTree", text: "\\".repeat(64 * 1024), truncated: true },
    });
    expect(parseComputerHistoryInteractionEvent(value)).toEqual(value);
    const { mouse: _mouse, ...base } = interactionEvent();
    const selection = {
      ...base,
      kind: "selection.changed",
      selection: {
        text: "selected",
        truncated: false,
        selectedRange: { location: 2, length: 8 },
        target: {
          role: "textArea",
          title: "Editor",
          description: "Current editor",
        },
      },
    };
    expect(parseComputerHistoryInteractionEvent(selection)).toEqual(selection);
    const submit = {
      ...base,
      kind: "keyboard.submit",
      keyboard: {
        keyEquivalent: "return",
        modifiers: [],
        target: { role: "AXTextArea", title: "Composer" },
      },
    };
    expect(parseComputerHistoryInteractionEvent(submit)).toEqual(submit);
    expect(() => parseComputerHistoryInteractionEvent({
      ...submit,
      keyboard: { keyEquivalent: "return", text: "\n" },
    })).toThrow("Invalid computer history event");
    expect(() => parseComputerHistoryInteractionEvent({
      ...submit,
      keyboard: { keyEquivalent: "tab" },
    })).toThrow("Invalid computer history event");
    expect(() => parseComputerHistoryInteractionEvent({
      ...value,
      kind: "mouse.unknown",
    })).toThrow("Invalid computer history event");
    expect(() => parseComputerHistoryInteractionEvent({
      ...value,
      kind: "keyboard.text_input",
      keyboard: { text: "a" },
    })).toThrow("Invalid computer history event");
  });

  it("strictly parses a bounded suppressed-event count", () => {
    expect(parseComputerHistorySuppressedEvents({
      count: 4,
      timestamp: 1_788_200_000,
    })).toEqual({ count: 4, timestamp: 1_788_200_000 });
    expect(() => parseComputerHistorySuppressedEvents({
      count: 0,
      timestamp: 1_788_200_000,
    })).toThrow("Invalid suppressed interaction count");
    expect(parseComputerHistorySuppressedEvents({
      count: 1,
      timestamp: 253_402_300_799.999,
    }).timestamp).toBe(253_402_300_799.999);
    expect(() => parseComputerHistorySuppressedEvents({
      count: 1,
      timestamp: 253_402_300_800,
    })).toThrow("Invalid suppressed interaction count");
  });
});

describe("PerceptionManager", () => {
  it("publishes interaction_event only on its dedicated additive channel", () => {
    const harness = createHarness();
    harness.manager.start({
      workspace: false,
      screen: false,
      accessibility: false,
      camera: false,
      computerHistory: true,
    });
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const interactions: unknown[] = [];
    const suppressed: unknown[] = [];
    const protocolErrors: Error[] = [];
    const genericEvents: unknown[] = [];
    harness.manager.on("interaction-event", (value) => interactions.push(value));
    harness.manager.on("interaction-events-suppressed", (value) => suppressed.push(value));
    harness.manager.on("protocol-error", (error) => protocolErrors.push(error));
    harness.manager.on("event", (value) => genericEvents.push(value));

    child.send(event("interaction_event", interactionEvent()));
    child.send(event("interaction_events_suppressed", {
      count: 3,
      timestamp: 1_788_200_001,
    }));
    child.send(event("interaction_event", {
      ...interactionEvent(),
      kind: "not-supported",
      timestamp: 1_788_200_002,
    }));

    expect(interactions).toEqual([interactionEvent()]);
    expect(suppressed).toEqual([
      { count: 3, timestamp: 1_788_200_001 },
      { count: 1, timestamp: 1_788_200_002 },
    ]);
    expect(protocolErrors.map((error) => error.message)).toEqual([
      "Invalid computer history event",
    ]);
    expect(genericEvents).toEqual([]);
    harness.manager.stop();
  });

  it("configures lazy modules only after the helper hello", () => {
    const harness = createHarness();
    harness.manager.start({ workspace: false, screen: true, accessibility: false, camera: false });
    const child = harness.children[0]!;

    expect(child.stdinData()).toBe("");
    child.send(event("hello", { capabilities: capabilities() }));

    const command = JSON.parse(child.stdinData().trim()) as Record<string, unknown>;
    expect(command).toMatchObject({
      protocolVersion: 1,
      command: "configure",
      environmentFrameIntervalMs: 1_000,
      modules: { workspace: false, screen: true, accessibility: false, camera: false },
    });
    expect((command.modules as Record<string, unknown>).enhancedVision).toBe(false);
    expect((command.modules as Record<string, unknown>).legacyScreenAnalysis).toBe(true);
    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "ready",
      helperPid: 100,
    });
    harness.manager.stop();
  });

  it("persists frame cadence, force-refreshes it, and replays it after restart", async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.manager.setEnvironmentFrameInterval(5_000);
    harness.manager.start({
      workspace: false,
      screen: true,
      accessibility: false,
      camera: false,
    });
    const first = harness.children[0]!;
    first.send(event("hello", { capabilities: capabilities() }));
    let configure = commands(first).at(-1)!;
    expect(configure.environmentFrameIntervalMs).toBe(5_000);
    first.send(event(
      "status",
      { capabilities: capabilities() },
      configure.requestId as string,
    ));

    harness.manager.setEnvironmentFrameInterval(1_000, true);
    configure = commands(first).at(-1)!;
    expect(configure.environmentFrameIntervalMs).toBe(1_000);
    const commandCount = commands(first).length;
    harness.manager.setEnvironmentFrameInterval(1_000);
    expect(commands(first)).toHaveLength(commandCount);

    first.crash(1);
    await vi.advanceTimersByTimeAsync(500);
    const restarted = harness.children[1]!;
    restarted.send(event("hello", { capabilities: capabilities() }));
    expect(commands(restarted).at(-1)?.environmentFrameIntervalMs).toBe(1_000);
    harness.manager.stop();
  });

  it("rejects unsafe frame cadence values", () => {
    const harness = createHarness();
    for (const value of [249, 60_001, 1.5, Number.NaN]) {
      expect(() => harness.manager.setEnvironmentFrameInterval(value)).toThrow(
        "interval",
      );
    }
  });

  it("forwards an explicit legacy screen-analysis opt-out", () => {
    const harness = createHarness();
    harness.manager.start({
      workspace: false,
      screen: true,
      accessibility: false,
      camera: false,
      legacyScreenAnalysis: false,
    });
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));

    const command = commands(child)[0]!;
    expect(command.modules).toMatchObject({
      screen: true,
      legacyScreenAnalysis: false,
    });
    harness.manager.stop();
  });

  it("reconfigures capture exclusion without restarting the helper", () => {
    const harness = createHarness();
    const modules = {
      workspace: false, screen: true, accessibility: false, camera: false,
      legacyScreenAnalysis: false,
    };
    harness.manager.start(modules);
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const initial = commands(child).at(-1)!;
    expect((initial.modules as Record<string, unknown>).excludeHostApplicationFromCapture).toBe(false);
    child.send(event("status", { capabilities: capabilities() }, initial.requestId as string));

    harness.manager.configure({ ...modules, excludeHostApplicationFromCapture: true });
    expect(harness.children).toHaveLength(1);
    expect((commands(child).at(-1)!.modules as Record<string, unknown>).excludeHostApplicationFromCapture).toBe(true);
    harness.manager.stop();
  });

  it("defaults a legacy helper's missing computer-history capability without rejecting status", () => {
    const harness = createHarness();
    harness.manager.start({
      workspace: false,
      screen: false,
      accessibility: false,
      camera: false,
      computerHistory: true,
    });
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const configure = commands(child).at(-1)!;
    child.send(event(
      "status",
      { capabilities: capabilities() },
      configure.requestId as string,
    ));

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "degraded",
      capabilities: {
        computerHistory: {
          supported: false,
          requested: false,
          state: "disabled",
          permission: "not_determined",
        },
      },
    });
    harness.manager.stop();
  });

  it("accepts and clones a native computer-history capability report", () => {
    const harness = createHarness();
    harness.manager.start({
      workspace: false,
      screen: false,
      accessibility: false,
      camera: false,
      computerHistory: true,
    });
    const child = harness.children[0]!;
    const inputHealth = {
      accessibilityPermission: "granted",
      listenEventPermission: "granted",
      eventTapState: "running",
      mouse: { state: "running", receivedCount: 7, lastEventAt: 1_788_200_001 },
      keyboard: { state: "running", receivedCount: 11, lastEventAt: 1_788_200_002 },
      tapDisabledCount: 1,
    };
    const computerHistory = {
      ...capability("running", "granted", true),
      computerHistoryInput: inputHealth,
    };
    const currentCapabilities = {
      ...capabilities(),
      computerHistory,
    };
    child.send(event("hello", { capabilities: currentCapabilities }));
    const configure = commands(child).at(-1)!;
    child.send(event(
      "status",
      { capabilities: currentCapabilities },
      configure.requestId as string,
    ));

    const status = harness.manager.currentStatus();
    expect(status.phase).toBe("ready");
    expect(status.capabilities?.computerHistory).toEqual(computerHistory);
    const cloned = harness.manager.currentStatus();
    expect(cloned.capabilities?.computerHistory).not.toBe(
      status.capabilities?.computerHistory,
    );
    expect(cloned.capabilities?.computerHistory.computerHistoryInput).not.toBe(
      status.capabilities?.computerHistory.computerHistoryInput,
    );
    expect(cloned.capabilities?.computerHistory.computerHistoryInput?.mouse).not.toBe(
      status.capabilities?.computerHistory.computerHistoryInput?.mouse,
    );
    harness.manager.stop();
  });

  it("backs off twice then degrades perception without touching other runtimes", async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.manager.start();

    harness.children[0]!.crash(1);
    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "restarting",
      crashCount: 1,
    });
    expect(harness.children).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(500);
    expect(harness.children).toHaveLength(2);
    harness.children[1]!.crash(2);
    await vi.advanceTimersByTimeAsync(1_999);
    expect(harness.children).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(harness.children).toHaveLength(3);

    harness.children[2]!.crash(3);
    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "degraded",
      crashCount: 3,
      helperPid: null,
    });
    expect(harness.manager.currentStatus().detail).toContain("聊天、记忆和语音不受影响");

    await vi.advanceTimersByTimeAsync(60_000);
    expect(harness.children).toHaveLength(3);
  });

  it("reports a requested placeholder module as degraded without stopping the helper", () => {
    const harness = createHarness();
    harness.manager.start({ workspace: false, screen: true, accessibility: false, camera: false });
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const command = JSON.parse(child.stdinData().trim()) as { requestId: string };
    child.send(event("status", {
      capabilities: {
        ...capabilities(),
        screen: {
          supported: true,
          requested: true,
          state: "unavailable",
          permission: "not_determined",
          reason: "screen_capture_not_implemented",
        },
      },
    }, command.requestId));

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "degraded",
      helperPid: 100,
      capabilities: {
        screen: { state: "unavailable", reason: "screen_capture_not_implemented" },
      },
    });
    expect(child.killedWith).toEqual([]);
    harness.manager.stop();
  });

  it("does not restart after an intentional stop", async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.manager.start();
    const child = harness.children[0]!;

    const stopped = harness.manager.stop();
    child.crash(null, "SIGTERM");
    await stopped;
    await vi.runAllTimersAsync();

    expect(harness.children).toHaveLength(1);
    expect(harness.manager.currentStatus().phase).toBe("stopped");
    expect(harness.manager.currentStatus().modules).toEqual({
      workspace: false,
      screen: false,
      accessibility: false,
      camera: false,
      enhancedVision: false,
      computerHistory: false,
      legacyScreenAnalysis: false,
      excludeHostApplicationFromCapture: false,
    });
    expect(child.killedWith).toEqual([]);
  });

  it("turns a helper stdin EPIPE into an isolated restart", async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.manager.start({ workspace: false, screen: true, accessibility: false, camera: false });
    const child = harness.children[0]!;

    child.stdin.emit("error", new Error("write EPIPE"));

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "restarting",
      crashCount: 1,
      helperPid: null,
    });
    expect(child.killedWith).toEqual(["SIGTERM"]);
    await vi.advanceTimersByTimeAsync(500);
    expect(harness.children).toHaveLength(2);
  });

  it("restarts when the helper never sends hello", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ handshakeTimeoutMs: 100 });
    harness.manager.start();

    await vi.advanceTimersByTimeAsync(99);
    expect(harness.manager.currentStatus().phase).toBe("starting");
    await vi.advanceTimersByTimeAsync(1);

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "restarting",
      crashCount: 1,
    });
    expect(harness.children[0]!.killedWith).toEqual(["SIGTERM"]);
  });

  it("restarts when configure never receives its matching status", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ configureTimeoutMs: 100 });
    harness.manager.start({ workspace: false, screen: true, accessibility: false, camera: false });
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));

    await vi.advanceTimersByTimeAsync(100);

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "restarting",
      crashCount: 1,
    });
    expect(child.killedWith).toEqual(["SIGTERM"]);
  });

  it("renews liveness only after the matching ping acknowledgement", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ pingIntervalMs: 100, pingTimeoutMs: 40 });
    harness.manager.start();
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const configure = commands(child)[0]!;
    child.send(event("status", { capabilities: capabilities() }, configure.requestId as string));

    await vi.advanceTimersByTimeAsync(100);
    const firstPing = commands(child)[1]!;
    expect(firstPing.command).toBe("ping");

    await vi.advanceTimersByTimeAsync(39);
    child.send(event("ack", { command: "ping" }, firstPing.requestId as string));
    await vi.advanceTimersByTimeAsync(99);
    expect(commands(child)).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);

    expect(commands(child)[2]).toMatchObject({ command: "ping" });
    expect(harness.manager.currentStatus().phase).toBe("ready");
    harness.manager.stop();
  });

  it("restarts when a ping acknowledgement is missing or has the wrong request id", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ pingIntervalMs: 100, pingTimeoutMs: 40 });
    harness.manager.start();
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const configure = commands(child)[0]!;
    child.send(event("status", { capabilities: capabilities() }, configure.requestId as string));

    await vi.advanceTimersByTimeAsync(100);
    child.send(event("ack", { command: "ping" }, "wrong-request-id"));
    await vi.advanceTimersByTimeAsync(39);
    expect(harness.manager.currentStatus().phase).toBe("ready");
    await vi.advanceTimersByTimeAsync(1);

    expect(harness.manager.currentStatus()).toMatchObject({
      phase: "restarting",
      crashCount: 1,
      helperPid: null,
    });
    expect(child.killedWith).toEqual(["SIGTERM"]);
  });

  it("clears scheduled and in-flight liveness timers on stop", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ pingIntervalMs: 100, pingTimeoutMs: 40 });
    harness.manager.start();
    const child = harness.children[0]!;
    child.send(event("hello", { capabilities: capabilities() }));
    const configure = commands(child)[0]!;
    child.send(event("status", { capabilities: capabilities() }, configure.requestId as string));
    await vi.advanceTimersByTimeAsync(100);
    expect(commands(child)[1]).toMatchObject({ command: "ping" });

    const stopped = harness.manager.stop();
    // Only the bounded graceful-shutdown fallback remains; liveness is gone.
    expect(vi.getTimerCount()).toBe(1);
    child.crash(0);
    await stopped;
    expect(vi.getTimerCount()).toBe(0);
    await vi.runAllTimersAsync();

    expect(harness.children).toHaveLength(1);
    expect(harness.manager.currentStatus().phase).toBe("stopped");
    expect(child.killedWith).toEqual([]);
  });

  it("waits for native cleanup before restart and kills only after the shutdown deadline", async () => {
    vi.useFakeTimers();
    const harness = createHarness({ shutdownTimeoutMs: 100 });
    harness.manager.start();
    const first = harness.children[0]!;
    first.send(event("hello", { capabilities: capabilities() }));

    const stopped = harness.manager.stop();
    harness.manager.start();
    expect(commands(first).at(-1)).toMatchObject({ command: "shutdown" });
    expect(first.killedWith).toEqual([]);
    expect(harness.children).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(99);
    expect(first.killedWith).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    await stopped;

    expect(first.killedWith).toEqual(["SIGKILL"]);
    expect(harness.children).toHaveLength(2);
  });

  it("publishes only candidate metadata on its dedicated event", () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const genericEvents: unknown[] = [];
    const observations: unknown[] = [];
    const logs: unknown[] = [];
    const candidates: unknown[] = [];
    harness.manager.on("event", (value) => genericEvents.push(value));
    harness.manager.on("observation", (value) => observations.push(value));
    harness.manager.on("log", (value) => logs.push(value));
    harness.manager.on("vision-candidate-available", (value) => candidates.push(value));

    const metadata = visionMetadata();
    child.send(event("vision_candidate_available", metadata));

    expect(candidates).toEqual([metadata]);
    expect(genericEvents).toEqual([]);
    expect(observations).toEqual([]);
    expect(logs).toEqual([]);
    const configure = commands(child)[0]!;
    expect(configure).toMatchObject({
      command: "configure",
      modules: { enhancedVision: true },
    });
    harness.manager.stop();
  });

  it("claims exactly one candidate without exposing its JPEG generically", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const genericEvents: unknown[] = [];
    const observations: unknown[] = [];
    const logs: unknown[] = [];
    harness.manager.on("event", (value) => genericEvents.push(value));
    harness.manager.on("observation", (value) => observations.push(value));
    harness.manager.on("log", (value) => logs.push(value));
    const metadata = visionMetadata();
    child.send(event("vision_candidate_available", metadata));

    const pending = harness.manager.claimVisionCandidate(metadata.candidateId);
    const claim = commands(child).at(-1)!;
    expect(claim).toEqual({
      protocolVersion: 1,
      requestId: expect.any(String),
      command: "claim_vision_candidate",
      candidateId: metadata.candidateId,
    });
    const jpegBase64 = jpegCandidateBase64();
    child.send(event("vision_candidate", {
      ...metadata,
      mimeType: "image/jpeg",
      jpegBase64,
    }, claim.requestId as string));

    await expect(pending).resolves.toEqual({
      ...metadata,
      mimeType: "image/jpeg",
      jpegBase64,
    });
    expect(genericEvents).toEqual([]);
    expect(observations).toEqual([]);
    expect(logs).toEqual([]);
    harness.manager.stop();
  });

  it("drops a candidate response with the wrong request id", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const genericEvents: unknown[] = [];
    harness.manager.on("event", (value) => genericEvents.push(value));
    const metadata = visionMetadata();
    child.send(event("vision_candidate_available", metadata));
    const pending = harness.manager.claimVisionCandidate(metadata.candidateId);
    const claim = commands(child).at(-1)!;
    let settled = false;
    void pending.then(() => { settled = true; }, () => { settled = true; });

    child.send(event("vision_candidate", visionPayload(metadata), "wrong-request-id"));
    await Promise.resolve();
    expect(settled).toBe(false);
    expect(genericEvents).toEqual([]);

    child.send(event(
      "vision_candidate",
      visionPayload(metadata),
      claim.requestId as string,
    ));
    await expect(pending).resolves.toMatchObject({ candidateId: metadata.candidateId });
    harness.manager.stop();
  });

  it("rejects a pending claim when a newer candidate supersedes it", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const first = visionMetadata();
    child.send(event("vision_candidate_available", first));
    const pending = harness.manager.claimVisionCandidate(first.candidateId);
    const oldClaim = commands(child).at(-1)!;
    const second = visionMetadata({
      candidateId: CANDIDATE_ID_TWO,
      revision: first.revision + 1,
      observedAt: first.observedAt + 0.5,
      expiresAt: first.observedAt + 5.5,
      windowKey: WINDOW_ID_TWO,
    });

    child.send(event("vision_candidate_available", second));

    await expect(pending).rejects.toMatchObject({ code: "stale_candidate" });
    child.send(event(
      "vision_candidate",
      visionPayload(first),
      oldClaim.requestId as string,
    ));
    const nextPending = harness.manager.claimVisionCandidate(second.candidateId);
    const nextClaim = commands(child).at(-1)!;
    child.send(event(
      "vision_candidate",
      visionPayload(second),
      nextClaim.requestId as string,
    ));
    await expect(nextPending).resolves.toMatchObject({ candidateId: second.candidateId });
    harness.manager.stop();
  });

  it("times out and clears the single pending claim", async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    const child = readyForVision(harness);
    const first = visionMetadata();
    child.send(event("vision_candidate_available", first));
    const pending = harness.manager.claimVisionCandidate(first.candidateId, 100);
    const rejected = expect(pending).rejects.toMatchObject({ code: "timeout" });

    await vi.advanceTimersByTimeAsync(100);
    await rejected;

    const second = visionMetadata({
      candidateId: CANDIDATE_ID_TWO,
      revision: first.revision + 1,
      observedAt: first.observedAt + 0.1,
      expiresAt: first.observedAt + 5.1,
      windowKey: WINDOW_ID_TWO,
    });
    child.send(event("vision_candidate_available", second));
    const nextPending = harness.manager.claimVisionCandidate(second.candidateId, 100);
    child.send(event(
      "vision_candidate_unavailable",
      { code: "unavailable" },
      commands(child).at(-1)!.requestId as string,
    ));
    await expect(nextPending).rejects.toMatchObject({ code: "unavailable" });
    harness.manager.stop();
  });

  it("releases a pending claim immediately when its caller aborts", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const genericEvents: unknown[] = [];
    harness.manager.on("event", (value) => genericEvents.push(value));
    const first = visionMetadata();
    child.send(event("vision_candidate_available", first));
    const controller = new AbortController();
    const pending = harness.manager.claimVisionCandidate(
      first.candidateId,
      undefined,
      controller.signal,
    );
    const oldClaim = commands(child).at(-1)!;

    controller.abort("context-changed");
    await expect(pending).rejects.toMatchObject({ code: "aborted" });
    child.send(event(
      "vision_candidate",
      visionPayload(first),
      oldClaim.requestId as string,
    ));

    const second = visionMetadata({
      candidateId: CANDIDATE_ID_TWO,
      revision: first.revision + 1,
      observedAt: first.observedAt + 0.1,
      expiresAt: first.observedAt + 5.1,
      windowKey: WINDOW_ID_TWO,
    });
    child.send(event("vision_candidate_available", second));
    const next = harness.manager.claimVisionCandidate(second.candidateId);
    const nextClaim = commands(child).at(-1)!;
    child.send(event(
      "vision_candidate",
      visionPayload(second),
      nextClaim.requestId as string,
    ));
    await expect(next).resolves.toMatchObject({ candidateId: second.candidateId });
    expect(genericEvents).toEqual([]);
    harness.manager.stop();
  });

  it("rejects and clears a claim on stop or helper restart", async () => {
    const stoppedHarness = createHarness();
    const stoppedChild = readyForVision(stoppedHarness);
    const stoppedMetadata = visionMetadata();
    stoppedChild.send(event("vision_candidate_available", stoppedMetadata));
    const stoppedClaim = stoppedHarness.manager.claimVisionCandidate(
      stoppedMetadata.candidateId,
    );
    stoppedHarness.manager.stop();
    await expect(stoppedClaim).rejects.toMatchObject({ code: "helper_stopped" });

    const restartedHarness = createHarness();
    const restartedChild = readyForVision(restartedHarness);
    const restartedMetadata = visionMetadata();
    restartedChild.send(event("vision_candidate_available", restartedMetadata));
    const restartedClaim = restartedHarness.manager.claimVisionCandidate(
      restartedMetadata.candidateId,
    );
    restartedChild.crash(1);
    await expect(restartedClaim).rejects.toMatchObject({ code: "helper_restarted" });
    restartedHarness.manager.stop();
  });

  it("rejects oversized or malformed image payloads without a generic leak", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const genericEvents: unknown[] = [];
    const protocolErrors: Error[] = [];
    harness.manager.on("event", (value) => genericEvents.push(value));
    harness.manager.on("protocol-error", (error) => protocolErrors.push(error));
    const first = visionMetadata();
    child.send(event("vision_candidate_available", first));
    const oversized = harness.manager.claimVisionCandidate(first.candidateId);
    child.send(event("vision_candidate", {
      ...first,
      mimeType: "image/jpeg",
      jpegBase64: jpegCandidateBase64(MAX_VISION_CANDIDATE_JPEG_BYTES + 1),
    }, commands(child).at(-1)!.requestId as string));
    await expect(oversized).rejects.toMatchObject({ code: "invalid_payload" });

    const second = visionMetadata({
      candidateId: CANDIDATE_ID_TWO,
      revision: first.revision + 1,
      observedAt: first.observedAt + 0.1,
      expiresAt: first.observedAt + 5.1,
      windowKey: WINDOW_ID_TWO,
    });
    child.send(event("vision_candidate_available", second));
    const malformed = harness.manager.claimVisionCandidate(second.candidateId);
    child.send(event("vision_candidate", {
      ...second,
      mimeType: "image/png",
      jpegBase64: jpegCandidateBase64(),
    }, commands(child).at(-1)!.requestId as string));
    await expect(malformed).rejects.toMatchObject({ code: "invalid_payload" });

    expect(protocolErrors.map((error) => error.message)).toEqual([
      "Invalid vision candidate payload",
      "Invalid vision candidate payload",
    ]);
    expect(genericEvents).toEqual([]);
    harness.manager.stop();
  });

  it("turns candidate EPIPE and helper errors into fixed claim codes", async () => {
    const transportHarness = createHarness();
    const transportChild = readyForVision(transportHarness);
    const transportMetadata = visionMetadata();
    transportChild.send(event("vision_candidate_available", transportMetadata));
    const transportClaim = transportHarness.manager.claimVisionCandidate(
      transportMetadata.candidateId,
    );
    transportChild.stdin.emit("error", new Error("write EPIPE"));
    await expect(transportClaim).rejects.toMatchObject({ code: "transport_error" });
    expect(transportHarness.manager.currentStatus().phase).toBe("restarting");
    transportHarness.manager.stop();

    const helperHarness = createHarness();
    const helperChild = readyForVision(helperHarness);
    const helperMetadata = visionMetadata();
    helperChild.send(event("vision_candidate_available", helperMetadata));
    const helperClaim = helperHarness.manager.claimVisionCandidate(helperMetadata.candidateId);
    const helperRequest = commands(helperChild).at(-1)!;
    const genericEvents: unknown[] = [];
    const helperErrors: unknown[] = [];
    helperHarness.manager.on("event", (value) => genericEvents.push(value));
    helperHarness.manager.on("helper-error", (value) => helperErrors.push(value));
    helperChild.send(event("error", {
      code: "internal_image_failure",
      message: "must not be exposed",
      fatal: false,
    }, helperRequest.requestId as string));
    await expect(helperClaim).rejects.toMatchObject({ code: "helper_error" });
    expect(genericEvents).toEqual([]);
    expect(helperErrors).toEqual([]);
    helperHarness.manager.stop();
  });

  it("allows only one pending claim", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const metadata = visionMetadata();
    child.send(event("vision_candidate_available", metadata));
    const first = harness.manager.claimVisionCandidate(metadata.candidateId);
    await expect(harness.manager.claimVisionCandidate(metadata.candidateId)).rejects.toMatchObject({
      code: "busy",
    });
    child.send(event(
      "vision_candidate_unavailable",
      { code: "unavailable" },
      commands(child).at(-1)!.requestId as string,
    ));
    await expect(first).rejects.toMatchObject({ code: "unavailable" });
    harness.manager.stop();
  });

  it("clears revision fencing when enhanced vision is disabled and re-enabled", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const candidates: unknown[] = [];
    harness.manager.on("vision-candidate-available", (value) => candidates.push(value));
    const beforeDisable = visionMetadata({ revision: 20 });
    child.send(event("vision_candidate_available", beforeDisable));
    const pendingBeforeDisable = harness.manager.claimVisionCandidate(
      beforeDisable.candidateId,
    );

    harness.manager.configure(visionModules({ enhancedVision: false }));
    await expect(pendingBeforeDisable).rejects.toMatchObject({ code: "stale_candidate" });
    acknowledgeLatestConfigure(harness.manager, child);
    await expect(
      harness.manager.claimVisionCandidate(beforeDisable.candidateId),
    ).rejects.toMatchObject({ code: "helper_unavailable" });
    harness.manager.configure(visionModules({ enhancedVision: true }));
    acknowledgeLatestConfigure(harness.manager, child);

    const afterEnable = visionMetadata({
      candidateId: CANDIDATE_ID_TWO,
      revision: 1,
      observedAt: beforeDisable.observedAt + 0.2,
      expiresAt: beforeDisable.observedAt + 5.2,
      windowKey: WINDOW_ID_TWO,
    });
    child.send(event("vision_candidate_available", afterEnable));

    expect(candidates).toEqual([beforeDisable, afterEnable]);
    const pending = harness.manager.claimVisionCandidate(afterEnable.candidateId);
    const claim = commands(child).at(-1)!;
    child.send(event("vision_candidate", visionPayload(afterEnable), claim.requestId as string));
    await expect(pending).resolves.toMatchObject({ revision: 1 });
    harness.manager.stop();
  });

  it("publishes metadata then claims one environment frame without generic pixel events", async () => {
    const harness = createHarness();
    const child = readyForVision(harness);
    const frames: unknown[] = [];
    const genericEvents: unknown[] = [];
    harness.manager.on("environment-frame-available", (value) => frames.push(value));
    harness.manager.on("event", (value) => genericEvents.push(value));
    const metadata = environmentFrameMetadata();

    child.send(event("environment_frame_available", metadata));
    expect(frames).toEqual([metadata]);
    const pending = harness.manager.claimEnvironmentFrame(metadata.frameId);
    const claim = commands(child).at(-1)!;
    expect(claim).toEqual({
      protocolVersion: 1,
      requestId: expect.any(String),
      command: "claim_environment_frame",
      frameId: metadata.frameId,
    });
    child.send(event(
      "environment_frame",
      environmentFramePayload(metadata),
      claim.requestId as string,
    ));

    await expect(pending).resolves.toEqual(environmentFramePayload(metadata));
    expect(genericEvents).toEqual([]);
    harness.manager.stop();
  });

  it("keeps valid observations flowing when one protocol line is malformed", () => {
    const harness = createHarness();
    const observations: unknown[] = [];
    const protocolErrors: Error[] = [];
    harness.manager.on("observation", (value) => observations.push(value));
    harness.manager.on("protocol-error", (error) => protocolErrors.push(error));
    harness.manager.start();
    const child = harness.children[0]!;

    child.stdout.write("not-json\n");
    child.send(event("observation", { revision: 1, kind: "foreground_app_changed" }));

    expect(protocolErrors).toHaveLength(1);
    expect(observations).toEqual([{ revision: 1, kind: "foreground_app_changed" }]);
  });
});

function createHarness(
  options: Partial<ConstructorParameters<typeof PerceptionManager>[0]> = {},
): {
  manager: PerceptionManager;
  children: FakePerceptionProcess[];
} {
  const children: FakePerceptionProcess[] = [];
  const spawnProcess: PerceptionProcessSpawner = () => {
    const child = new FakePerceptionProcess(100 + children.length);
    children.push(child);
    return child;
  };
  return {
    manager: new PerceptionManager({
      helperPath: "/tmp/fish-memory-perception-helper",
      spawnProcess,
      ...options,
    }),
    children,
  };
}

class FakePerceptionProcess extends EventEmitter implements PerceptionHelperProcess {
  readonly stdin = new PassThrough();
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly killedWith: NodeJS.Signals[] = [];
  private written = "";

  constructor(readonly pid: number) {
    super();
    this.stdin.on("data", (chunk) => (this.written += chunk.toString()));
  }

  stdinData(): string {
    return this.written;
  }

  send(value: Record<string, unknown>): void {
    this.stdout.write(`${JSON.stringify(value)}\n`);
  }

  crash(code: number | null, signal: NodeJS.Signals | null = null): void {
    this.emit("exit", code, signal);
  }

  kill(signal: NodeJS.Signals = "SIGTERM"): boolean {
    this.killedWith.push(signal);
    return true;
  }
}

function event(
  name: string,
  body: unknown,
  requestId?: string,
): Record<string, unknown> {
  return {
    protocolVersion: 1,
    sequence: 1,
    sentAt: Date.now() / 1_000,
    event: name,
    ...(requestId ? { requestId } : {}),
    body,
  };
}

function interactionEvent(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id: "33333333-3333-4333-8333-333333333333",
    timestamp: 1_788_200_000,
    kind: "mouse.click",
    app: { bundleIdentifier: "com.example.editor", name: "Editor" },
    window: { title: "Document" },
    mouse: {
      button: "left",
      clickCount: 1,
      location: { x: 10, y: 20 },
      target: { role: "button", title: "Save" },
    },
    ...overrides,
  };
}

function commands(child: FakePerceptionProcess): Array<Record<string, unknown>> {
  return child.stdinData()
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function capabilities() {
  return {
    workspace: capability("running", "not_required", true),
    screen: capability("disabled", "not_determined", false),
    accessibility: capability("disabled", "not_determined", false),
    camera: capability("disabled", "not_determined", false),
  };
}

function capability(state: string, permission: string, requested: boolean) {
  return { supported: true, requested, state, permission };
}

const CANDIDATE_ID_ONE = "11111111-1111-4111-8111-111111111111";
const CANDIDATE_ID_TWO = "22222222-2222-4222-8222-222222222222";
const WINDOW_ID_ONE = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA";
const WINDOW_ID_TWO = "BBBBBBBB-BBBB-4BBB-8BBB-BBBBBBBBBBBB";

function readyForVision(
  harness: ReturnType<typeof createHarness>,
): FakePerceptionProcess {
  harness.manager.start(visionModules());
  const child = harness.children[0]!;
  const currentCapabilities = visionCapabilities();
  child.send(event("hello", { capabilities: currentCapabilities }));
  acknowledgeLatestConfigure(harness.manager, child, currentCapabilities);
  return child;
}

function acknowledgeLatestConfigure(
  manager: PerceptionManager,
  child: FakePerceptionProcess,
  currentCapabilities = visionCapabilities(),
): void {
  const configure = commands(child).at(-1)!;
  expect(configure.command).toBe("configure");
  child.send(event(
    "status",
    { capabilities: currentCapabilities },
    configure.requestId as string,
  ));
  expect(manager.currentStatus().phase).toBe("ready");
}

function visionModules(
  overrides: Partial<{
    workspace: boolean;
    screen: boolean;
    accessibility: boolean;
    camera: boolean;
    enhancedVision: boolean;
  }> = {},
) {
  return {
    workspace: true,
    screen: true,
    accessibility: true,
    camera: false,
    enhancedVision: true,
    ...overrides,
  };
}

function visionCapabilities() {
  return {
    workspace: capability("running", "not_required", true),
    screen: capability("running", "granted", true),
    accessibility: capability("running", "granted", true),
    camera: capability("disabled", "not_determined", false),
  };
}

function visionMetadata(
  overrides: Partial<{
    candidateId: string;
    revision: number;
    observedAt: number;
    expiresAt: number;
    displayID: number;
    appBundleId: string;
    appName: string;
    windowKey: string;
  }> = {},
) {
  const observedAt = Date.now() / 1_000;
  return {
    candidateId: CANDIDATE_ID_ONE,
    revision: 7,
    observedAt,
    expiresAt: observedAt + 5,
    displayID: 1,
    appBundleId: "com.example.editor",
    appName: "Editor",
    windowKey: WINDOW_ID_ONE,
    ...overrides,
  };
}

function visionPayload(metadata = visionMetadata()) {
  return {
    ...metadata,
    mimeType: "image/jpeg",
    jpegBase64: jpegCandidateBase64(),
  };
}

function environmentFrameMetadata() {
  const observedAt = Date.now() / 1_000;
  return {
    frameId: CANDIDATE_ID_ONE,
    revision: 1,
    observedAt,
    expiresAt: observedAt + 3,
    displayID: 1,
    pixelWidth: 960,
    pixelHeight: 540,
    byteLength: 64,
  };
}

function environmentFramePayload(metadata = environmentFrameMetadata()) {
  return {
    ...metadata,
    mimeType: "image/jpeg",
    jpegBase64: jpegCandidateBase64(metadata.byteLength),
  };
}

function jpegCandidateBase64(byteLength = 64): string {
  if (byteLength < 5) throw new Error("test JPEG must have at least five bytes");
  const bytes = Buffer.alloc(byteLength);
  bytes[0] = 0xff;
  bytes[1] = 0xd8;
  bytes[2] = 0xff;
  bytes[byteLength - 2] = 0xff;
  bytes[byteLength - 1] = 0xd9;
  return bytes.toString("base64");
}
