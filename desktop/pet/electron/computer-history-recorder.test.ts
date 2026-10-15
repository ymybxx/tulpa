import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  ComputerHistoryRecorder,
  MAX_COMPUTER_HISTORY_SEGMENT_BYTES,
  type ComputerHistoryCheckpointRequest,
  type ComputerHistorySummarizeRequest,
} from "./computer-history-recorder";
import { COMPUTER_HISTORY_SELF_OBSERVATION_MARKER } from "./computer-history-self-observation";
import type { ComputerHistoryInteractionEvent } from "./perception-manager";

const FISH_ID = "00000000-0000-0000-0000-000000000001";
const SEGMENT_ID_ONE = "00000000-0000-0000-0000-000000000101";
const SEGMENT_ID_TWO = "00000000-0000-0000-0000-000000000102";
const BASE_MS = Date.parse("2026-09-02T10:00:00.000Z");
const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) =>
    rm(directory, { recursive: true, force: true })));
});

describe("ComputerHistoryRecorder", () => {
  it("delivers context events immediately and resumes from the durable offset without summaries", async () => {
    const root = await temporaryRoot();
    const delivered: unknown[][] = [];
    const urls: string[] = [];
    const create = () => new ComputerHistoryRecorder({
      rootPath: root, deliveryMode: "context", now: () => BASE_MS + 60_000,
      connection: async () => ({ apiUrl: "http://127.0.0.1:8123", apiKey: "local-key" }),
      fetcher: async (input, init) => {
        const url = String(input); urls.push(url);
        if (url.endsWith("/v1/fishes/assistant")) return Response.json({ fishId: FISH_ID });
        expect(url).toContain("/history/events");
        delivered.push((JSON.parse(init!.body as string) as { events: unknown[] }).events);
        return Response.json({ accepted: delivered.at(-1)!.length });
      },
    });
    let recorder = create();
    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 1_000));
    await waitFor(async () => {
      const [name] = await readdir(root);
      return name !== undefined && await readFile(path.join(root, name, "context-offset.json"), "utf8").then(() => true).catch(() => false);
    });
    await recorder.stop();
    expect(delivered.flat()).toHaveLength(1);
    recorder = create();
    await recorder.setEnabled(true);
    await recorder.record(event(2, BASE_MS + 2_000));
    await waitFor(async () => delivered.flat().length === 2);
    await recorder.stop();
    expect(urls.some((url) => url.includes("summarize") || url.includes("checkpoint"))).toBe(false);
  });

  it("rejects timestamps beyond year 9999 without creating an orphan segment", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    await recorder.record({ ...event(1, BASE_MS), timestamp: 253_402_300_800 });

    await waitFor(() => readdir(root).then((entries) => entries.length === 0));
    await recorder.stop();
  });

  it("accepts the first helper event while asynchronous recovery is still queued", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 5 * 60_000,
      fetcher: summaryFetcher([]),
    });

    const ready = recorder.setEnabled(true);
    const firstEvent = recorder.record(event(1, BASE_MS + 60_000));
    await Promise.all([ready, firstEvent]);

    const [segment] = await readdir(root);
    const eventsJsonl = await readFile(path.join(root, segment!, "events.jsonl"), "utf8");
    expect(eventsJsonl).toContain(event(1, 0).id);
    await recorder.stop();
  });

  it("refreshes an active segment only after Ask and observes the one-minute cooldown", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 60_000;
    const requests: ComputerHistoryCheckpointRequest[] = [];
    let assistantLookups = 0;
    let checkpoint: Record<string, unknown> | null = null;
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          assistantLookups += 1;
          return Response.json({ fishId: FISH_ID });
        }
        if (url.endsWith("/checkpoint") && init?.method !== "POST") {
          return checkpoint ? Response.json(checkpoint) : new Response(null, { status: 404 });
        }
        if (init?.method !== "POST") return new Response(null, { status: 404 });
        const request = JSON.parse(String(init.body)) as ComputerHistoryCheckpointRequest;
        requests.push(request);
        checkpoint = {
          schemaVersion: 1,
          segmentId: SEGMENT_ID_ONE,
          startedAt: request.startedAt,
          endedAt: request.endedAt,
          revision: request.baseRevision + 1,
          throughEventCount: request.throughEventCount,
          throughByteOffset: request.throughByteOffset,
          throughSuppressedEventCount: request.throughSuppressedEventCount,
          deltaChainDigest: request.deltaDigest,
          lastSuccessfulSummaryAt: new Date(now).toISOString(),
          status: "active",
          summary: "Summary",
          applications: [],
          finalizedAt: null,
          observationId: null,
          finalEventDigest: null,
        };
        return Response.json(checkpoint);
      },
    });
    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));
    await settleRecorder();
    expect(requests).toEqual([]);

    await recorder.requestRefresh(FISH_ID);
    await waitFor(() => Promise.resolve(requests.length === 1));
    expect(requests[0]).toMatchObject({
      fromEventCount: 0,
      throughEventCount: 1,
      finalize: false,
    });

    await recorder.record(event(2, BASE_MS + 61_000));
    await recorder.requestRefresh(FISH_ID);
    await settleRecorder();
    expect(requests).toHaveLength(1);

    now += 60_000;
    await recorder.requestRefresh("00000000-0000-0000-0000-000000000999");
    await settleRecorder();
    expect(requests).toHaveLength(1);

    await recorder.requestRefresh(FISH_ID);
    await waitFor(() => Promise.resolve(requests.length === 2));
    expect(requests[1]).toMatchObject({
      baseRevision: 1,
      fromEventCount: 1,
      throughEventCount: 2,
      finalize: false,
    });
    expect(assistantLookups).toBe(0);
    await recorder.stop();
  });

  it("honors the server cooldown after recovering a checkpoint ahead of local metadata", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 2 * 60_000;
    let serverCheckpoint: Record<string, unknown> | null = null;
    let posts = 0;
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (url.endsWith("/checkpoint") && init?.method !== "POST") {
          return serverCheckpoint
            ? Response.json(serverCheckpoint)
            : new Response(null, { status: 404 });
        }
        if (init?.method === "POST") posts += 1;
        return new Response(null, { status: 503 });
      },
    });
    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));
    await recorder.record(event(2, BASE_MS + 61_000));

    const [directory] = await readdir(root);
    const metadata = JSON.parse(
      await readFile(path.join(root, directory!, "metadata.json"), "utf8"),
    ) as { segmentId: string; startedAt: string; endedAt: string };
    const lines = (await readFile(path.join(root, directory!, "events.jsonl"), "utf8"))
      .trimEnd()
      .split("\n");
    const firstLine = `${lines[0]}\n`;
    serverCheckpoint = {
      schemaVersion: 1,
      segmentId: metadata.segmentId,
      startedAt: metadata.startedAt,
      endedAt: metadata.endedAt,
      revision: 1,
      throughEventCount: 1,
      throughByteOffset: Buffer.byteLength(firstLine, "utf8"),
      throughSuppressedEventCount: 0,
      deltaChainDigest: createHash("sha256").update(firstLine, "utf8").digest("hex"),
      lastSuccessfulSummaryAt: "2026-09-02T10:02:00.123456Z",
      status: "active",
      summary: "Recovered summary",
      applications: ["TextEdit"],
      finalizedAt: null,
      observationId: null,
      finalEventDigest: null,
    };

    await recorder.requestRefresh();
    await settleRecorder();

    expect(posts).toBe(0);
    await recorder.stop();
  });

  it("writes UTC JSONL in append order and waits for the five-second grace", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 10 * 60_000 + 4_000;
    const requests: ComputerHistorySummarizeRequest[] = [];
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: summaryFetcher(requests),
    });

    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));
    await recorder.record(event(2, BASE_MS + 60_000));
    await recorder.record(event(1, BASE_MS + 60_000));
    await recorder.recordSuppressed(3, (BASE_MS + 60_000) / 1_000);
    expect(requests).toEqual([]);

    now += 1_000;
    await recorder.setEnabled(true);
    await waitFor(() => Promise.resolve(requests.length === 1));

    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      schemaVersion: 1,
      segmentId: SEGMENT_ID_ONE,
      startedAt: "2026-09-02T10:00:00.000Z",
      endedAt: "2026-09-02T10:10:00.000Z",
      eventCount: 2,
      suppressedEventCount: 3,
    });
    const lines = requests[0]!.eventsJsonl.trimEnd().split("\n").map((line) =>
      JSON.parse(line) as Record<string, unknown>);
    expect(lines.map((line) => [line.ordinal, line.id, line.timestamp])).toEqual([
      [1, event(1, 0).id, "2026-09-02T10:01:00.000Z"],
      [2, event(2, 0).id, "2026-09-02T10:01:00.000Z"],
    ]);
    await waitFor(() => readdir(root).then((entries) => entries.length === 0));
    await recorder.stop();
  });

  it("never appends late events after a segment's grace boundary", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input) => input.toString().endsWith("/v1/fishes/assistant")
        ? Response.json({ fishId: FISH_ID })
        : new Response(null, { status: 503 }),
    });
    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));

    now = BASE_MS + 10 * 60_000 + 5_000;
    await recorder.record(event(2, BASE_MS + 2 * 60_000));
    await recorder.recordSuppressed(3, (BASE_MS + 2 * 60_000) / 1_000);

    const lines = await persistedLines(root);
    expect(lines.map((line) => line.id)).toEqual([event(1, 0).id]);
    const [directory] = await readdir(root);
    const metadata = JSON.parse(
      await readFile(path.join(root, directory!, "metadata.json"), "utf8"),
    ) as { suppressedEventCount: number };
    expect(metadata.suppressedEventCount).toBe(0);
    await recorder.stop();
  });

  it("redacts a real nested Fish event envelope and later selections in the same viewer", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const requests: ComputerHistorySummarizeRequest[] = [];
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: summaryFetcher(requests),
    });
    await recorder.setEnabled(true);

    const nested = storedEnvelope({
      ordinal: 38,
      windowTitle: "events.jsonl",
      axText: "1. text area — a previous captured interface",
    });
    await recorder.record(textEditWindowEvent(
      31,
      BASE_MS + 60_000,
      `1. document — events.jsonl | ${nested}`,
    ));
    await recorder.record(textEditSelectionEvent(
      32,
      BASE_MS + 61_000,
      nested,
    ));

    const lines = await persistedLines(root);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({
      ordinal: 1,
      kind: "window.changed",
      timestamp: "2026-09-02T10:01:00.000Z",
      app: { bundleIdentifier: "com.apple.TextEdit", name: "TextEdit" },
      window: { title: "events.jsonl" },
      ax: { mode: "fullTree", text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER },
    });
    expect(lines[1]).toMatchObject({
      ordinal: 2,
      kind: "selection.changed",
      selection: {
        text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
        truncated: false,
      },
    });
    expect((lines[1]!.selection as Record<string, unknown>).selectedRange).toBeUndefined();
    expect(JSON.stringify(lines)).not.toContain("a previous captured interface");

    now = BASE_MS + 10 * 60_000 + 5_000;
    await recorder.setEnabled(true);
    await settleRecorder();
    await waitFor(() => Promise.resolve(requests.length === 1));
    await settleRecorder();
    expect(requests).toHaveLength(1);
    expect(requests[0]!.eventDigest).toBe(
      createHash("sha256").update(requests[0]!.eventsJsonl, "utf8").digest("hex"),
    );
    expect(requests[0]!.eventsJsonl).not.toContain("a previous captured interface");
    await recorder.stop();
  });

  it("keeps self-observation protection across a UTC segment boundary", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 10 * 60_000 + 2_000,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    await recorder.record(textEditWindowEvent(
      33,
      BASE_MS + 10 * 60_000 - 1_000,
      storedEnvelope({ ordinal: 1, windowTitle: "events.jsonl" }),
    ));
    await recorder.record(textEditSelectionEvent(
      34,
      BASE_MS + 10 * 60_000 + 1_000,
      "schemaVersion",
    ));

    const directories = (await readdir(root)).sort();
    expect(directories).toEqual([
      "2026-09-02T10-00-00Z",
      "2026-09-02T10-10-00Z",
    ]);
    const second = JSON.parse(
      (await readFile(path.join(root, directories[1]!, "events.jsonl"), "utf8")).trim(),
    ) as Record<string, unknown>;
    expect(second.selection).toMatchObject({
      text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
    });
    await recorder.stop();
  });

  it("restores a recent self-observation context after recorder restart", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const first = recorderFor(root, { now: () => now, fetcher: summaryFetcher([]) });
    await first.setEnabled(true);
    await first.record(textEditWindowEvent(
      35,
      BASE_MS + 60_000,
      storedEnvelope({ ordinal: 4, windowTitle: "events.jsonl" }),
    ));
    await first.stop();

    now += 1_000;
    const recovered = recorderFor(root, { now: () => now, fetcher: summaryFetcher([]) });
    await recovered.setEnabled(true);
    await recovered.record(textEditSelectionEvent(
      36,
      BASE_MS + 62_000,
      "ordinal",
    ));

    const lines = await persistedLines(root);
    expect(lines).toHaveLength(2);
    expect(lines[1]!.selection).toMatchObject({
      text: COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
    });
    await recovered.stop();
  });

  it("exits the protected context when the user switches windows", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 5 * 60_000,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    await recorder.record(textEditWindowEvent(
      37,
      BASE_MS + 60_000,
      storedEnvelope({ ordinal: 5, windowTitle: "events.jsonl" }),
    ));
    await recorder.record({
      ...textEditWindowEvent(38, BASE_MS + 61_000, "1. text area — Meeting notes"),
      window: { title: "notes.txt" },
    });
    await recorder.record({
      ...textEditSelectionEvent(39, BASE_MS + 62_000, "ordinary selected notes"),
      window: { title: "notes.txt" },
    });

    const lines = await persistedLines(root);
    expect((lines[1]!.ax as Record<string, unknown>).text).toBe("1. text area — Meeting notes");
    expect((lines[2]!.selection as Record<string, unknown>).text).toBe(
      "ordinary selected notes",
    );
    await recorder.stop();
  });

  it("does not redact an unrelated code project file named events.jsonl", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 5 * 60_000,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    const fixture = storedEnvelope({
      ordinal: 12,
      windowTitle: "fixture.jsonl",
      axText: "unit-test fixture",
    });
    const source = `const fixture = ${JSON.stringify(fixture)};`;
    await recorder.record({
      id: "00000000-0000-0000-0000-000000000040",
      timestamp: (BASE_MS + 60_000) / 1_000,
      kind: "window.changed",
      app: { bundleIdentifier: "com.microsoft.VSCode", name: "Visual Studio Code" },
      window: { title: "events.jsonl" },
      ax: { mode: "fullTree", text: source, truncated: false },
    });

    const [line] = await persistedLines(root);
    expect((line!.ax as Record<string, unknown>).text).toBe(source);
    await recorder.stop();
  });

  it("recognizes Finder preview only when it exposes this recorder's managed path", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 5 * 60_000,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    const nested = storedEnvelope({ ordinal: 41, windowTitle: "events.jsonl" });
    await recorder.record({
      id: "00000000-0000-0000-0000-000000000041",
      timestamp: (BASE_MS + 60_000) / 1_000,
      kind: "window.changed",
      app: { bundleIdentifier: "com.apple.finder", name: "Finder" },
      window: { title: "2026-09-02T10-00-00Z" },
      ax: {
        mode: "fullTree",
        text: `1. preview — file://${root}/2026-09-02T10-00-00Z/events.jsonl | ${nested}`,
        truncated: false,
      },
    });

    const [line] = await persistedLines(root);
    expect((line!.ax as Record<string, unknown>).text).toBe(
      COMPUTER_HISTORY_SELF_OBSERVATION_MARKER,
    );
    await recorder.stop();
  });

  it("keeps ordinary JSON visible in TextEdit even when its filename is events.jsonl", async () => {
    const root = await temporaryRoot();
    const recorder = recorderFor(root, {
      now: () => BASE_MS + 5 * 60_000,
      fetcher: summaryFetcher([]),
    });
    await recorder.setEnabled(true);

    const ordinary = "{\"schemaVersion\":1,\"ordinal\":7,\"message\":\"application event\"}";
    await recorder.record(textEditWindowEvent(42, BASE_MS + 60_000, ordinary));

    const [line] = await persistedLines(root);
    expect((line!.ax as Record<string, unknown>).text).toBe(ordinary);
    await recorder.stop();
  });

  it("deletes a segment containing only session.started without calling the API", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const calls: string[] = [];
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input) => {
        calls.push(input.toString());
        throw new Error("must not fetch");
      },
    });

    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 1_000, "session.started"));
    await recorder.recordSuppressed(5, (BASE_MS + 2_000) / 1_000);
    expect(await readdir(root)).toHaveLength(1);
    now = BASE_MS + 10 * 60_000 + 5_000;
    await recorder.setEnabled(true);
    await settleRecorder();

    expect(calls).toEqual([]);
    expect(await readdir(root)).toEqual([]);
    await recorder.stop();
  });

  it("caps raw JSONL at 512 KiB and reports suppressed events", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const requests: ComputerHistorySummarizeRequest[] = [];
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: summaryFetcher(requests),
    });
    await recorder.setEnabled(true);
    for (let index = 1; index <= 9; index += 1) {
      await recorder.record(axEvent(index, BASE_MS + index * 1_000));
    }

    now = BASE_MS + 10 * 60_000 + 5_000;
    await recorder.setEnabled(true);
    await waitFor(() => Promise.resolve(requests.length === 1));

    expect(requests).toHaveLength(1);
    expect(Buffer.byteLength(requests[0]!.eventsJsonl, "utf8")).toBeLessThanOrEqual(
      MAX_COMPUTER_HISTORY_SEGMENT_BYTES,
    );
    expect(requests[0]!.eventCount).toBeLessThan(9);
    expect(requests[0]!.suppressedEventCount).toBeGreaterThan(0);
    await recorder.stop();
  });

  it("backs off failed segments without blocking later summaries or resetting their retry", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const posts: string[] = [];
    const finalized = new Map<string, ComputerHistorySummarizeRequest>();
    let firstFailuresRemaining = 2;
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (init?.method !== "POST") {
          if (url.endsWith("/checkpoint")) return new Response(null, { status: 404 });
          const segmentId = url.match(/segments\/([0-9a-f-]+)$/u)?.[1];
          const finalizedRequest = segmentId ? finalized.get(segmentId) : undefined;
          return finalizedRequest ? confirmation(finalizedRequest) : new Response(null, { status: 404 });
        }
        const request = JSON.parse(String(init.body)) as ComputerHistoryCheckpointRequest;
        const segmentId = url.match(/segments\/([0-9a-f-]+)\/checkpoint$/u)?.[1];
        if (!segmentId) throw new Error("missing checkpoint segment id");
        posts.push(segmentId);
        if (segmentId === SEGMENT_ID_ONE && firstFailuresRemaining > 0) {
          firstFailuresRemaining -= 1;
          return new Response(null, { status: 503 });
        }
        const legacy: ComputerHistorySummarizeRequest = {
          schemaVersion: 1,
          segmentId,
          startedAt: request.startedAt,
          endedAt: request.endedAt,
          eventCount: request.throughEventCount,
          suppressedEventCount: request.throughSuppressedEventCount,
          eventDigest: request.finalEventDigest ?? request.deltaDigest,
          eventsJsonl: request.deltaEventsJsonl,
        };
        finalized.set(segmentId, legacy);
        return checkpointResponse(request, segmentId);
      },
    });

    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));
    await recorder.record(event(2, BASE_MS + 11 * 60_000));
    now = BASE_MS + 20 * 60_000 + 5_000;
    await recorder.setEnabled(true);
    await settleRecorder();
    await waitFor(() => Promise.resolve(posts.length === 2));
    expect(posts).toEqual([SEGMENT_ID_ONE, SEGMENT_ID_TWO]);
    await waitFor(() => readdir(root).then((entries) => entries.length === 1));
    const remaining = (await readdir(root))[0]!;
    expect(JSON.parse(await readFile(path.join(root, remaining, "metadata.json"), "utf8")))
      .toMatchObject({ segmentId: SEGMENT_ID_ONE, summaryStatus: "pending" });
    expect(await readFile(path.join(root, remaining, "events.jsonl"), "utf8"))
      .toContain(event(1, BASE_MS + 60_000).id);

    now += 4_999;
    await recorder.setEnabled(true);
    await settleRecorder();
    expect(posts).toHaveLength(2);

    now += 1;
    await recorder.setEnabled(true);
    await settleRecorder();
    await waitFor(() => Promise.resolve(posts.length === 3));
    expect(posts).toEqual([SEGMENT_ID_ONE, SEGMENT_ID_TWO, SEGMENT_ID_ONE]);

    now += 29_999;
    await recorder.setEnabled(true);
    await settleRecorder();
    expect(posts).toHaveLength(3);

    now += 1;
    await recorder.setEnabled(true);
    await waitFor(() => Promise.resolve(posts.length === 4));
    expect(posts).toEqual([SEGMENT_ID_ONE, SEGMENT_ID_TWO, SEGMENT_ID_ONE, SEGMENT_ID_ONE]);
    await waitFor(() => readdir(root).then((entries) => entries.length === 0));
    await recorder.stop();
  });

  it("recovers an uncertain success through GET and removes .deleting directories", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const first = recorderFor(root, { now: () => now, fetcher: summaryFetcher([]) });
    await first.setEnabled(true);
    await first.record(event(1, BASE_MS + 60_000));
    await first.stop();

    await mkdir(path.join(root, `old.deleting-${SEGMENT_ID_TWO}`), { recursive: true });
    const segmentDirectory = (await readdir(root)).find((name) => !name.includes(".deleting-"))!;
    const metadata = JSON.parse(
      await readFile(path.join(root, segmentDirectory, "metadata.json"), "utf8"),
    ) as { segmentId: string };
    const eventsJsonl = await readFile(path.join(root, segmentDirectory, "events.jsonl"), "utf8");
    const digest = createHash("sha256").update(eventsJsonl, "utf8").digest("hex");
    const posts: string[] = [];
    const acknowledgements: string[] = [];
    now = BASE_MS + 10 * 60_000 + 5_000;
    const recovered = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (init?.method === "DELETE") {
          acknowledgements.push(url);
          return new Response(null, { status: 204 });
        }
        if (init?.method === "POST") posts.push(url);
        if (url.endsWith("/checkpoint")) {
          return Response.json({
            schemaVersion: 1,
            segmentId: metadata.segmentId,
            startedAt: "2026-09-02T10:00:00.000Z",
            endedAt: "2026-09-02T10:10:00.000Z",
            revision: 1,
            throughEventCount: 1,
            throughByteOffset: Buffer.byteLength(eventsJsonl, "utf8"),
            throughSuppressedEventCount: 0,
            deltaChainDigest: digest,
            lastSuccessfulSummaryAt: "2026-09-02T10:10:00.000Z",
            status: "finalized",
            summary: "Summary",
            applications: [],
            finalizedAt: "2026-09-02T10:10:00.123456Z",
            observationId: "00000000-0000-0000-0000-000000000201",
            finalEventDigest: digest,
          });
        }
        return confirmation({
          schemaVersion: 1,
          segmentId: metadata.segmentId,
          startedAt: "2026-09-02T10:00:00.000Z",
          endedAt: "2026-09-02T10:10:00.000Z",
          eventCount: 1,
          suppressedEventCount: 0,
          eventDigest: digest,
          eventsJsonl,
        });
      },
    });

    await recovered.setEnabled(true);
    await waitFor(() => readdir(root).then((entries) => entries.length === 0));

    expect(posts).toEqual([]);
    expect(acknowledgements).toHaveLength(1);
    const acknowledgement = new URL(acknowledgements[0]!);
    expect(acknowledgement.searchParams.get("observationId")).toBe(
      "00000000-0000-0000-0000-000000000201",
    );
    expect(acknowledgement.searchParams.get("finalEventDigest")).toBe(digest);
    expect(await readdir(root)).toEqual([]);
    await recovered.stop();
  });

  it("recovers from an acknowledged checkpoint using the final Observation", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const first = recorderFor(root, { now: () => now, fetcher: summaryFetcher([]) });
    await first.setEnabled(true);
    await first.record(event(1, BASE_MS + 60_000));
    await first.stop();

    const [segmentDirectory] = await readdir(root);
    const metadata = JSON.parse(
      await readFile(path.join(root, segmentDirectory!, "metadata.json"), "utf8"),
    ) as { segmentId: string };
    const eventsJsonl = await readFile(
      path.join(root, segmentDirectory!, "events.jsonl"),
      "utf8",
    );
    const digest = createHash("sha256").update(eventsJsonl, "utf8").digest("hex");
    let posts = 0;
    now = BASE_MS + 10 * 60_000 + 5_000;
    const recovered = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (init?.method === "POST") posts += 1;
        if (url.includes("/checkpoint")) return new Response(null, { status: 404 });
        return confirmation({
          schemaVersion: 1,
          segmentId: metadata.segmentId,
          startedAt: "2026-09-02T10:00:00.000Z",
          endedAt: "2026-09-02T10:10:00.000Z",
          eventCount: 1,
          suppressedEventCount: 0,
          eventDigest: digest,
          eventsJsonl,
        });
      },
    });

    await recovered.setEnabled(true);
    await waitFor(() => readdir(root).then((entries) => entries.length === 0));

    expect(posts).toBe(0);
    await recovered.stop();
  });

  it("keeps a segment when the returned digest does not confirm the request", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const recorder = recorderFor(root, {
      now: () => now,
      fetcher: async (input, init) => {
        if (input.toString().endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (init?.method !== "POST") return new Response(null, { status: 404 });
        const request = JSON.parse(String(init.body)) as ComputerHistorySummarizeRequest;
        return Response.json({
          startedAt: request.startedAt,
          endedAt: request.endedAt,
          payload: {
            segmentId: request.segmentId,
            eventDigest: "0".repeat(64),
            eventCount: request.eventCount,
            suppressedEventCount: request.suppressedEventCount,
          },
        });
      },
    });

    await recorder.setEnabled(true);
    await recorder.record(event(1, BASE_MS + 60_000));
    now = BASE_MS + 10 * 60_000 + 5_000;
    await recorder.setEnabled(true);

    expect(await readdir(root)).toHaveLength(1);
    await recorder.stop();
  });

  it.each(["startedAt", "eventCount"] as const)(
    "keeps a segment when confirmation %s does not match",
    async (mismatch) => {
      const root = await temporaryRoot();
      let now = BASE_MS + 5 * 60_000;
      const recorder = recorderFor(root, {
        now: () => now,
        fetcher: async (input, init) => {
          if (input.toString().endsWith("/v1/fishes/assistant")) {
            return Response.json({ fishId: FISH_ID });
          }
          if (init?.method !== "POST") return new Response(null, { status: 404 });
          const request = JSON.parse(String(init.body)) as ComputerHistorySummarizeRequest;
          return Response.json({
            startedAt: mismatch === "startedAt"
              ? "2026-09-02T09:50:00Z"
              : request.startedAt,
            endedAt: request.endedAt,
            payload: {
              segmentId: request.segmentId,
              eventDigest: request.eventDigest,
              eventCount: mismatch === "eventCount"
                ? request.eventCount + 1
                : request.eventCount,
              suppressedEventCount: request.suppressedEventCount,
            },
          });
        },
      });

      await recorder.setEnabled(true);
      await recorder.record(event(1, BASE_MS + 60_000));
      now = BASE_MS + 10 * 60_000 + 5_000;
      await recorder.setEnabled(true);

      expect(await readdir(root)).toHaveLength(1);
      await recorder.stop();
    },
  );

  it("expires an unconfirmed pending segment after seven days without summarizing", async () => {
    const root = await temporaryRoot();
    let now = BASE_MS + 5 * 60_000;
    const first = recorderFor(root, { now: () => now, fetcher: summaryFetcher([]) });
    await first.setEnabled(true);
    await first.record(event(1, BASE_MS + 60_000));
    await first.stop();

    const calls: string[] = [];
    now = BASE_MS + 10 * 60_000 + 7 * 24 * 60 * 60_000;
    const recovered = recorderFor(root, {
      now: () => now,
      fetcher: async (input) => {
        calls.push(input.toString());
        throw new Error("expired segments must not be submitted");
      },
    });

    await recovered.setEnabled(false);

    expect(calls).toEqual([]);
    expect(await readdir(root)).toEqual([]);
    await recovered.stop();
  });
});

async function temporaryRoot(): Promise<string> {
  const root = path.join(
    await mkdtemp(path.join(os.tmpdir(), "fish-memory-computer-history-")),
    "segments",
  );
  temporaryDirectories.push(path.dirname(root));
  return root;
}

async function settleRecorder(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function waitFor(check: () => Promise<boolean>, attempts = 40): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await check()) return;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("recorder did not settle");
}

async function persistedLines(root: string): Promise<Record<string, unknown>[]> {
  const directories = (await readdir(root)).filter((name) => !name.includes(".deleting-"));
  const contents = await Promise.all(directories.sort().map((directory) =>
    readFile(path.join(root, directory, "events.jsonl"), "utf8")));
  return contents.join("").trimEnd().split("\n").filter(Boolean).map((line) =>
    JSON.parse(line) as Record<string, unknown>);
}

function storedEnvelope(options: {
  ordinal: number;
  windowTitle: string;
  axText?: string;
}): string {
  return JSON.stringify({
    schemaVersion: 1,
    ordinal: options.ordinal,
    id: `90000000-0000-0000-0000-${String(options.ordinal).padStart(12, "0")}`,
    timestamp: "2026-09-02T10:00:30.000Z",
    kind: "window.changed",
    app: { bundleIdentifier: "com.apple.TextEdit", name: "TextEdit" },
    window: { title: options.windowTitle },
    ax: {
      mode: "fullTree",
      text: options.axText ?? "1. document — captured content",
      truncated: false,
    },
  });
}

function textEditWindowEvent(
  index: number,
  timestampMs: number,
  text: string,
): ComputerHistoryInteractionEvent {
  return {
    id: `00000000-0000-0000-0000-${String(index).padStart(12, "0")}`,
    timestamp: timestampMs / 1_000,
    kind: "window.changed",
    app: { bundleIdentifier: "com.apple.TextEdit", name: "TextEdit" },
    window: { title: "events.jsonl" },
    ax: { mode: "fullTree", text, truncated: false },
  };
}

function textEditSelectionEvent(
  index: number,
  timestampMs: number,
  text: string,
): ComputerHistoryInteractionEvent {
  return {
    id: `00000000-0000-0000-0000-${String(index).padStart(12, "0")}`,
    timestamp: timestampMs / 1_000,
    kind: "selection.changed",
    app: { bundleIdentifier: "com.apple.TextEdit", name: "TextEdit" },
    window: { title: "events.jsonl" },
    selection: {
      text,
      truncated: false,
      selectedRange: { location: 1_024, length: text.length },
      target: { role: "text area", title: "events.jsonl" },
    },
  };
}

function recorderFor(
  rootPath: string,
  options: {
    now: () => number;
    fetcher: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
  },
): ComputerHistoryRecorder {
  let nextId = 0;
  return new ComputerHistoryRecorder({
    rootPath,
    connection: async () => ({ apiUrl: "http://127.0.0.1:8123", apiKey: "local-key" }),
    now: options.now,
    fetcher: options.fetcher,
    retryDelaysMs: [5_000, 30_000, 120_000, 600_000],
    idFactory: () => [SEGMENT_ID_ONE, SEGMENT_ID_TWO][nextId++]!,
  });
}

function summaryFetcher(
  requests: ComputerHistorySummarizeRequest[],
): (input: string | URL | Request, init?: RequestInit) => Promise<Response> {
  const finalized = new Map<string, ComputerHistorySummarizeRequest>();
  return async (input, init) => {
    const url = input.toString();
    if (url.endsWith("/v1/fishes/assistant")) return Response.json({ fishId: FISH_ID });
    if (init?.method !== "POST") {
      const segmentId = url.match(/segments\/([0-9a-f-]+)(?:\/checkpoint)?$/u)?.[1];
      if (url.endsWith("/checkpoint")) return new Response(null, { status: 404 });
      const request = segmentId ? finalized.get(segmentId) : undefined;
      if (request) return confirmation(request);
      return new Response(null, { status: 404 });
    }
    expect(url).toBe(
      `http://127.0.0.1:8123/internal/fishes/${FISH_ID}`
      + `/computer-history/segments/${SEGMENT_ID_ONE}/checkpoint`,
    );
    expect(new Headers(init.headers).get("authorization")).toBe("Bearer local-key");
    const checkpoint = JSON.parse(String(init.body)) as ComputerHistoryCheckpointRequest;
    const request: ComputerHistorySummarizeRequest = {
      schemaVersion: checkpoint.schemaVersion,
      segmentId: SEGMENT_ID_ONE,
      startedAt: checkpoint.startedAt,
      endedAt: checkpoint.endedAt,
      eventCount: checkpoint.throughEventCount,
      suppressedEventCount: checkpoint.throughSuppressedEventCount,
      eventDigest: checkpoint.finalEventDigest ?? checkpoint.deltaDigest,
      eventsJsonl: checkpoint.deltaEventsJsonl,
    };
    requests.push(request);
    finalized.set(request.segmentId, request);
    return checkpointResponse(checkpoint, request.segmentId);
  };
}

function checkpointResponse(
  request: ComputerHistoryCheckpointRequest,
  segmentId: string,
): Response {
  return Response.json({
    schemaVersion: 1,
    segmentId,
    startedAt: request.startedAt,
    endedAt: request.endedAt,
    revision: request.baseRevision + 1,
    throughEventCount: request.throughEventCount,
    throughByteOffset: request.throughByteOffset,
    throughSuppressedEventCount: request.throughSuppressedEventCount,
    deltaChainDigest: request.deltaDigest,
    lastSuccessfulSummaryAt: request.throughEventCount > request.fromEventCount
      ? request.endedAt
      : null,
    status: request.finalize ? "finalized" : "active",
    summary: "Summary",
    applications: [],
    finalizedAt: request.finalize ? request.endedAt : null,
    observationId: request.finalize ? "00000000-0000-0000-0000-000000000201" : null,
    finalEventDigest: request.finalize ? request.finalEventDigest : null,
  });
}

function confirmation(request: ComputerHistorySummarizeRequest): Response {
  return Response.json({
    id: "00000000-0000-0000-0000-000000000201",
    startedAt: request.startedAt,
    endedAt: request.endedAt,
    payload: {
      segmentId: request.segmentId,
      eventDigest: request.eventDigest,
      eventCount: request.eventCount,
      suppressedEventCount: request.suppressedEventCount,
    },
  });
}

function event(
  index: number,
  timestampMs: number,
  kind: ComputerHistoryInteractionEvent["kind"] = "mouse.click",
): ComputerHistoryInteractionEvent {
  return {
    id: `00000000-0000-0000-0000-${String(index).padStart(12, "0")}`,
    timestamp: timestampMs / 1_000,
    kind,
    app: { bundleIdentifier: "com.example.editor", name: "Editor" },
    window: { title: "Document" },
    ...(kind === "mouse.click"
      ? { mouse: { button: "left", clickCount: 1, location: { x: 10, y: 20 } } }
      : {}),
  };
}

function axEvent(index: number, timestampMs: number): ComputerHistoryInteractionEvent {
  return {
    id: `10000000-0000-0000-0000-${String(index).padStart(12, "0")}`,
    timestamp: timestampMs / 1_000,
    kind: "window.changed",
    app: { bundleIdentifier: "com.example.editor", name: "Editor" },
    window: { title: "Document" },
    ax: { mode: "fullTree", text: "x".repeat(64 * 1024), truncated: false },
  };
}
