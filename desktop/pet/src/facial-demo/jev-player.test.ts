import { describe, expect, it, vi } from "vitest";
import type { DemoEvent } from "../shared/facial-demo";
import { JevTestPlayer, type JevTestStart } from "./jev-player";
import { buildBindings } from "./parameter-bindings";
import { CLOSED_MOUTH, type LipShape } from "../renderer/pet-renderer";

const bindings = buildBindings([{ id: "ParamEyeLOpen", name: "眼睛", minimum: 0, maximum: 1, defaultValue: 1 }]).active;
const face = (value: number) => ({ ParamEyeLOpen: value });
function fixture(resume: () => Promise<void> = async () => {}) {
  let receive: (event: DemoEvent) => void = () => {};
  let callback: FrameRequestCallback | null = null;
  let now = 0;
  const nodes: Array<{ start: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }> = [];
  const audio = {
    currentTime: 0, destination: {}, resume: vi.fn(resume), close: vi.fn(async () => {}),
    createGain: () => ({ gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() }),
    createBuffer: (_channels: number, samples: number, rate: number) => ({ duration: samples / rate, getChannelData: () => new Float32Array(samples) }),
    createBufferSource: () => {
      const node = { buffer: null, connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn(), onended: null };
      nodes.push(node); return node;
    },
  };
  const transport = {
    start: vi.fn(async (_request: JevTestStart) => {}), stop: vi.fn(async (_id: string) => {}),
    onEvent: (handler: (event: DemoEvent) => void) => { receive = handler; return vi.fn(); },
  };
  const preview = { beginFacialSpeech: vi.fn(), endFacialTest: vi.fn(), applyFacialTest: vi.fn(), setFacialTestLip: vi.fn(), facialTestSnapshot: () => ({ values: face(.75) }) };
  // The played audio's lip sync: the test hands shapes in as the worklet would.
  let heard: (shape: LipShape) => void = () => {};
  const lips = { stop: vi.fn() };
  const view = { status: vi.fn(), sentence: vi.fn(), values: vi.fn() };
  const player = new JevTestPlayer(transport, view, {
    createAudio: () => audio as unknown as AudioContext,
    followLips: (_audio, _input, onShape) => { heard = onShape; return lips; },
    requestFrame: next => { callback = next; return 1; }, cancelFrame: () => { callback = null; }, now: () => now,
  });
  return { player, transport, preview, view, audio, nodes, lips, hear: (shape: LipShape) => heard(shape),
    start: () => player.start("selected-preview", "第一句。第二句。", bindings, preview),
    emit: (event: DemoEvent) => receive({ id: transport.start.mock.calls.at(-1)?.[0].id, ...event }),
    tick: (seconds: number, milliseconds = seconds * 1000) => { audio.currentTime = seconds; now = milliseconds; const next = callback; callback = null; next?.(milliseconds); },
  };
}
const speech = (segmentIndex: number) => ({ type: "audio", segmentIndex, sampleRate: 1, pcm: "AAA=" });
const expression = (segmentIndex: number, value: number) => ({ type: "face", mode: "jev", segmentIndex, targets: face(value) });

describe("selected-model JEV speech player", () => {
  it("requests only enabled parameters, applies local gain, and moves the mouth with TA's lip sync", async () => {
    const f = fixture();
    const all = buildBindings([
      { id: "ParamEyeLOpen", name: "眼", minimum: 0, maximum: 2, defaultValue: 1 },
      { id: "ParamCheek", name: "脸红", minimum: 0, maximum: 1, defaultValue: 0 },
    ], null, [], {}, { ParamEyeLOpen: { enabled: true, multiplier: 2 }, ParamCheek: { enabled: false, multiplier: 1 } }).all;
    await f.player.start("a", "测试。", all, f.preview);
    expect(f.transport.start.mock.calls[0]![0].parameters.map(value => value.id)).toEqual(["ParamEyeLOpen"]);
    // How far the mouth opens is the lip sync's alone.
    expect(f.preview.beginFacialSpeech).toHaveBeenCalledWith(expect.objectContaining({ mouth: null }));
    f.emit(expression(0, .75)); f.emit(speech(0)); f.tick(.1);
    expect(f.preview.applyFacialTest).toHaveBeenLastCalledWith({ ParamEyeLOpen: .5 });
    const said = { volume: .6, a: 0, e: 0, i: .6, o: 0, u: 0 };
    f.hear(said);
    expect(f.preview.setFacialTestLip).toHaveBeenLastCalledWith(said);
    f.player.stop();
    expect(f.lips.stop).toHaveBeenCalledOnce();
    expect(f.preview.setFacialTestLip).toHaveBeenLastCalledWith(CLOSED_MOUTH);
    f.hear(said);
    expect(f.preview.setFacialTestLip).toHaveBeenLastCalledWith(CLOSED_MOUTH);
  });

  it("makes no request when every generation control is off", async () => {
    const f = fixture();
    await expect(f.player.start("a", "测试。", bindings.map(value => ({ ...value, enabled: false })), f.preview)).rejects.toThrow("至少开启");
    expect(f.transport.start).not.toHaveBeenCalled();
    expect(f.preview.beginFacialSpeech).not.toHaveBeenCalled();
  });
  it("sends only the selected model's actual standard range and no reasoning mode", async () => {
    const f = fixture(); await f.start();
    expect(f.transport.start).toHaveBeenCalledWith(expect.objectContaining({ modelId: "selected-preview", parameters: [expect.objectContaining({ id: "ParamEyeLOpen", minimum: 0, maximum: 1, defaultValue: 1, mode: "score" })] }));
    expect(f.transport.start.mock.calls[0]![0]).not.toHaveProperty("mode");
    expect(f.transport.start.mock.calls[0]![0]).not.toHaveProperty("currentFace");
    f.tick(.3);
    expect(f.view.values).toHaveBeenLastCalledWith(face(.75));
    f.player.stop();
  });
  it("holds voice until its JEV target is ready, then applies it on the audio clock", async () => {
    const f = fixture(); await f.start();
    f.emit({ type: "started", segments: ["第一句。", "第二句。"] });
    f.emit(speech(0)); expect(f.nodes).toHaveLength(0);
    f.emit({ ...expression(0, .3), mode: "reasoning" }); expect(f.nodes).toHaveLength(0);
    f.emit(expression(0, .2)); expect(f.nodes).toHaveLength(1);
    expect(f.preview.applyFacialTest).not.toHaveBeenCalled();
    f.tick(.1);
    expect(f.preview.applyFacialTest).toHaveBeenLastCalledWith(face(.2));
    expect(f.view.sentence).toHaveBeenLastCalledWith("第一句。");
    f.player.stop();
  });
  it("never applies a future sentence's expression or a stale run's result", async () => {
    const f = fixture(); await f.start();
    const oldId = f.transport.start.mock.calls[0]![0].id;
    f.emit(expression(1, .1)); f.emit(speech(0)); expect(f.nodes).toHaveLength(0);
    f.emit(expression(0, .6)); f.tick(.1);
    expect(f.preview.applyFacialTest).toHaveBeenLastCalledWith(face(.6));
    f.player.stop(); f.preview.applyFacialTest.mockClear(); await f.start();
    f.emit({ ...expression(0, .9), id: oldId }); f.emit(speech(0));
    expect(f.nodes).toHaveLength(1);
    expect(f.preview.applyFacialTest).not.toHaveBeenCalled();
    f.player.stop();
  });
  it("starts its 5 second timeout only after earlier speech can finish", async () => {
    const f = fixture(); await f.start();
    f.emit(expression(0, .5)); f.emit(speech(0)); f.emit(speech(1));
    f.tick(.5, 10_000); expect(f.player.running).toBe(true);
    f.tick(1.1, 11_000); expect(f.player.running).toBe(true);
    f.tick(1.1, 16_001); expect(f.player.running).toBe(false);
    expect(f.view.status).toHaveBeenLastCalledWith(expect.stringContaining("超过 5 秒"), false);
    expect(f.nodes[0]!.stop).toHaveBeenCalled();
  });
  it("cancels audio, releases the preview and cancels the backend on stop", async () => {
    const f = fixture(); await f.start();
    f.emit(expression(0, .5)); f.emit(speech(0));
    const id = f.transport.start.mock.calls[0]![0].id;
    f.player.stop(); f.emit(expression(0, .1));
    expect(f.nodes[0]!.stop).toHaveBeenCalledOnce();
    expect(f.audio.close).toHaveBeenCalledOnce();
    expect(f.preview.endFacialTest).toHaveBeenCalledOnce();
    expect(f.transport.stop).toHaveBeenCalledWith(id);
    expect(f.preview.applyFacialTest).not.toHaveBeenCalled();
  });
  it("does not resurrect after stop while AudioContext resume is pending", async () => {
    let resume!: () => void;
    const f = fixture(() => new Promise<void>(resolve => { resume = resolve; }));
    const starting = f.start(); f.player.stop(); resume(); await starting;
    expect(f.transport.start).not.toHaveBeenCalled();
    expect(f.preview.beginFacialSpeech).not.toHaveBeenCalled();
    expect(f.audio.close).toHaveBeenCalled();
  });
  it("finishes only after the last audio span and restores preview controls", async () => {
    const f = fixture(); await f.start();
    f.emit(expression(0, .4)); f.emit(speech(0)); f.emit({ type: "audio_end", segmentIndex: 0 });
    f.emit({ type: "done", status: "completed" });
    f.tick(.5); expect(f.player.running).toBe(true);
    f.tick(1.5); expect(f.player.running).toBe(false);
    expect(f.preview.endFacialTest).toHaveBeenCalledOnce();
    expect(f.view.status).toHaveBeenLastCalledWith(expect.stringContaining("朗读完成"), false);
  });
});
