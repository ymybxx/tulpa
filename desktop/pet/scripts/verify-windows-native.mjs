import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform !== "win32") throw new Error("This smoke test requires Windows");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const helper = process.argv[2] ?? path.join(root, "runtime/win32-x64/fish-memory-perception-helper.exe");
const require = createRequire(import.meta.url);
const {
  parsePerceptionEnvelope, parseEnvironmentFrameMetadata, parseEnvironmentFramePayload,
  parseComputerHistoryInteractionEvent, parseComputerHistorySuppressedEvents, parseSystemAudioWindow,
} = require(path.join(root, "dist-electron/electron/perception-manager.js"));
// The helper produces these fixtures with the same UUID, UTF-16/UTF-8 cleaning,
// frame/audio serializers and WIC encoder used at runtime, without device access.
const fixtureRun = spawnSync(helper, ["--protocol-fixtures"], { windowsHide: true, encoding: "utf8", timeout: 10_000, maxBuffer: 2_000_000 });
if (fixtureRun.error) throw fixtureRun.error;
assert.equal(fixtureRun.status, 0, fixtureRun.stderr);
const fixtures = fixtureRun.stdout.trim().split(/\r?\n/).map(parsePerceptionEnvelope);
const kinds = new Set(), audioStatuses = new Set();
for (const [index, fixture] of fixtures.entries()) {
  assert.equal(fixture.sequence, index + 1);
  switch (fixture.event) {
    case "environment_frame_available": parseEnvironmentFrameMetadata(fixture.body); break;
    case "environment_frame": parseEnvironmentFramePayload(fixture.body); break;
    case "interaction_event": {
      const event = parseComputerHistoryInteractionEvent(fixture.body);
      kinds.add(event.kind);
      if (event.keyboard?.text) assert(Buffer.byteLength(event.keyboard.text) <= 8192);
      if (event.selection?.text) assert(Buffer.byteLength(event.selection.text) <= 8192);
      break;
    }
    case "interaction_events_suppressed": parseComputerHistorySuppressedEvents(fixture.body); break;
    case "system_audio_window": {
      const window = parseSystemAudioWindow(fixture.body); audioStatuses.add(window.audioStatus);
      if (window.audioStatus === "unavailable") assert.equal(window.audioGapReason, "system_audio_disabled");
      if (window.audioStatus === "complete") assert.equal(window.audioSilence, true);
      break;
    }
    default: assert.fail(`Unexpected fixture event: ${fixture.event}`);
  }
}
assert.equal(kinds.size, 9);
assert.deepEqual(audioStatuses, new Set(["unavailable", "complete"]));
assert(fixtures.some(value => value.event === "environment_frame"));
const child = spawn(helper, ["--host-pid", String(process.pid)], { windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
const records = [], errors = [];
let pending = "";
child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
child.stdout.on("data", value => {
  pending += value;
  while (pending.includes("\n")) {
    const index = pending.indexOf("\n"); const line = pending.slice(0, index); pending = pending.slice(index + 1);
    try { records.push(parsePerceptionEnvelope(line)); } catch (error) { errors.push(String(error)); }
  }
});
child.stderr.on("data", () => {});
const exited = new Promise((resolve, reject) => { child.once("close", resolve); child.once("error", reject); });
const waitFor = async (predicate) => {
  const started = Date.now();
  while (!records.some(predicate)) {
    if (errors.length) throw new Error(errors.join("\n"));
    if (Date.now() - started > 10_000 || child.exitCode !== null) throw new Error("Helper protocol timeout");
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  return records.find(predicate);
};
const send = (command, fields = {}) => child.stdin.write(`${JSON.stringify({ protocolVersion: 1, requestId: command, command, ...fields })}\n`);
try {
  const hello = await waitFor(record => record.event === "hello");
  assert.equal(hello.protocolVersion, 1);
  assert.equal(hello.body.capabilities.camera.supported, false);
  assert.equal(hello.body.capabilities.screen.requested, false);
  send("configure", { modules: { workspace: false, screen: false, accessibility: false, camera: false, computerHistory: false, systemAudio: false, legacyScreenAnalysis: false }, environmentFrameIntervalMs: 1000 });
  const configured = await waitFor(record => record.event === "status" && record.requestId === "configure");
  assert.equal(configured.body.capabilities.screen.state, "disabled");
  send("claim_environment_frame", { frameId: "missing" });
  await waitFor(record => record.event === "environment_frame_unavailable" && record.requestId === "claim_environment_frame");
  send("ping"); await waitFor(record => record.event === "ack" && record.requestId === "ping");
  child.stdin.write('{"protocolVersion":999,"requestId":"invalid","command":"ping"}\n');
  await waitFor(record => record.event === "error" && record.requestId === "invalid");
  send("shutdown");
  assert.equal(await exited, 0);
  for (let i = 1; i < records.length; ++i) assert(records[i].sequence > records[i - 1].sequence);
  assert.equal(errors.length, 0);
  console.log(JSON.stringify({ ok: true, records: records.length, productionParserFixtures: fixtures.length, captureRequested: false }));
} finally { if (child.exitCode === null) child.kill(); }
