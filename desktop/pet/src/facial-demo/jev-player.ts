import type { DemoEvent, Face } from "../shared/facial-demo";
import type { FacialTestApi, FacialTestStartRequest } from "../shared/facial-test";
import { FaceTimeline } from "./timeline";
import { SpeechGate } from "./speech-gate";
import { conflictParts, fromNativeTargets, overlayBindings, planDecisions, toNativeTargets, type DecisionPlan, type ParameterBinding } from "./parameter-bindings";
import { facialConflictPlan, resolveFacialConflicts, type FacialConflictPlan, type ParameterEffects } from "../shared/facial-conflicts";
import type { FacialOverlayBindings } from "../shared/facial-bindings";
import { followPlayedAudio } from "../renderer/audio-lipsync";
import { CLOSED_MOUTH, type LipShape } from "../renderer/pet-renderer";

export interface JevTestPreview {
  beginFacialSpeech(bindings: FacialOverlayBindings): void;
  applyFacialTest(targets: Face): void;
  /** The speaking mouth, from the lip sync TA speaks with on the desktop. */
  setFacialTestLip(value: LipShape): void;
  facialTestSnapshot(): { values: Face };
  endFacialTest(): void;
}
export type JevTestStart = FacialTestStartRequest;
export type JevTestTransport = Pick<FacialTestApi, "start" | "stop" | "onEvent">;
interface PlayerCallbacks {
  status(message: string, running: boolean): void;
  sentence(text: string): void;
  values(face: Face): void;
}
interface PlayerRuntime {
  createAudio(): AudioContext;
  followLips(audio: AudioContext, input: AudioNode, onShape: (value: LipShape) => void): { stop(): void };
  requestFrame(callback: FrameRequestCallback): number;
  cancelFrame(id: number): void;
  now(): number;
}
const browserRuntime: PlayerRuntime = {
  createAudio: () => new AudioContext(),
  followLips: followPlayedAudio,
  requestFrame: callback => requestAnimationFrame(callback),
  cancelFrame: id => cancelAnimationFrame(id), now: () => performance.now(),
};

/** One selected-model preview, one audio clock, and only JEV decisions. */
export class JevTestPlayer {
  private readonly unsubscribe: () => void;
  private id: string | null = null;
  private preview: JevTestPreview | null = null;
  private bindings: readonly ParameterBinding[] = [];
  private plan: DecisionPlan | null = null;
  private conflicts: FacialConflictPlan = { exclusive: [], holds: {} };
  private audio: AudioContext | null = null;
  /** Every scheduled chunk plays through here, and the mouth follows it. */
  private output: GainNode | null = null;
  private lips: { stop(): void } | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private spans: Array<{ segment: number; start: number; end: number }> = [];
  private gate = new SpeechGate<DemoEvent>();
  private timeline = new FaceTimeline();
  private segments: string[] = [];
  private audioFinished = new Set<number>();
  private failedSegments = new Set<number>();
  private providerDone = false;
  private nextAudioTime = 0;
  private activeSegment = -1;
  private waitingSince: number | null = null;
  private displayAt = 0;
  private frame: number | null = null;

  constructor(
    private readonly transport: JevTestTransport,
    private readonly callbacks: PlayerCallbacks,
    private readonly runtime: PlayerRuntime = browserRuntime,
  ) {
    this.unsubscribe = transport.onEvent(event => this.handle(event));
  }

  get running(): boolean { return this.id !== null; }

  async start(modelId: string, text: string, bindings: readonly ParameterBinding[], preview: JevTestPreview, effects?: ParameterEffects): Promise<void> {
    if (this.running) throw new Error("请先停止当前测试");
    const generated = bindings.filter(binding => binding.controllable && binding.enabled);
    if (!text.trim() || !generated.length) throw new Error("请输入文字，并至少开启一个可用参数的自动生成");
    const id = crypto.randomUUID();
    this.id = id;
    this.bindings = bindings;
    const plan = planDecisions(generated);
    this.plan = plan;
    this.conflicts = facialConflictPlan(conflictParts(generated), effects);
    this.preview = preview;
    this.providerDone = false;
    this.nextAudioTime = 0;
    this.activeSegment = -1;
    this.displayAt = 0;
    this.waitingSince = null;
    this.spans = []; this.segments = [];
    this.timeline = new FaceTimeline(); this.gate.clear();
    this.failedSegments.clear(); this.audioFinished.clear();
    this.callbacks.status("正在生成语音与表情…", true);
    try {
      const audio = this.runtime.createAudio();
      this.audio = audio;
      await audio.resume();
      if (this.id !== id) return;
      this.output = audio.createGain();
      this.output.connect(audio.destination);
      this.lips = this.runtime.followLips(audio, this.output, (shape) => {
        if (this.id === id) this.preview?.setFacialTestLip(shape);
      });
      preview.beginFacialSpeech(overlayBindings(bindings));
      this.frame = this.runtime.requestFrame(() => this.tick());
      await this.transport.start({ id, modelId, text,
        parameters: plan.parameters });
    } catch (error) {
      if (this.id === id) this.stop(`测试失败：${message(error)}`);
    }
  }

  stop(status = "已停止"): void {
    const id = this.id;
    this.id = null; // Invalidate first: close/stop may synchronously emit old events.
    if (this.frame !== null) this.runtime.cancelFrame(this.frame);
    this.frame = null;
    this.gate.clear(); this.waitingSince = null;
    for (const source of this.sources) { try { source.stop(); } catch {} source.disconnect(); }
    this.sources.clear(); this.spans = [];
    this.lips?.stop(); this.lips = null;
    this.output?.disconnect(); this.output = null;
    const audio = this.audio; this.audio = null;
    if (audio) void audio.close().catch(() => {});
    this.preview?.setFacialTestLip(CLOSED_MOUTH);
    this.preview?.endFacialTest(); this.preview = null;
    if (id) void this.transport.stop(id).catch(() => {});
    this.callbacks.status(status, false);
  }

  destroy(): void { this.stop(); this.unsubscribe(); }

  private apply(targets: Face): void {
    this.preview?.applyFacialTest(toNativeTargets(targets, this.bindings));
    this.callbacks.values(targets);
  }

  private actualFace(): Face {
    const values = this.preview?.facialTestSnapshot().values ?? {};
    return fromNativeTargets(values, this.bindings);
  }

  private handle(event: DemoEvent): void {
    if (!this.id || event.id !== this.id) return;
    const segment = Number(event.segmentIndex);
    try {
      if (event.type === "started") this.segments = event.segments as string[];
      if (event.type === "segment") this.segments[segment] = String(event.text);
      if (event.type === "audio") { this.gate.push(segment, event); this.flush(); }
      if (event.type === "face" && event.mode === "jev") {
        const answers = event.targets as Face;
        const expanded = answers && typeof answers === "object" ? this.plan?.expand(answers) : null;
        const targets = expanded && resolveFacialConflicts(expanded, conflictParts(this.bindings), this.conflicts);
        if (!targets) {
          this.failedSegments.add(segment); return;
        }
        const current = this.timeline.receive(segment, targets);
        if (current) this.apply(current);
        this.gate.ready(segment); this.flush();
      }
      if (event.type === "audio_end") this.audioFinished.add(segment);
      if (event.type === "error") {
        if (event.stage === "jev") this.failedSegments.add(segment);
        else this.stop(`测试失败：${String(event.message ?? "语音生成失败")}`);
      }
      if (event.type === "fatal") this.stop(`测试失败：${String(event.message ?? "服务已断开")}`);
      if (event.type === "done") {
        this.providerDone = true;
        if (event.status === "failed" || event.status === "stopped") this.stop(`测试${event.status === "failed" ? "失败" : "已停止"}`);
      }
    } catch (error) { this.stop(`播放失败：${message(error)}`); }
  }

  private flush(): void {
    const audio = this.audio;
    if (!audio || !this.output) return;
    for (const { segment, chunk } of this.gate.drain()) {
      const binary = atob(String(chunk.pcm));
      const view = new DataView(Uint8Array.from(binary, character => character.charCodeAt(0)).buffer);
      const sampleRate = Number(chunk.sampleRate);
      if (!view.byteLength || view.byteLength % 2 || !Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error("无效的语音数据");
      const buffer = audio.createBuffer(1, view.byteLength / 2, sampleRate);
      const samples = buffer.getChannelData(0);
      for (let index = 0; index < samples.length; index++) samples[index] = view.getInt16(index * 2, true) / 32768;
      const source = audio.createBufferSource(); source.buffer = buffer; source.connect(this.output);
      const start = Math.max(this.nextAudioTime, audio.currentTime + 0.03);
      const end = start + buffer.duration;
      this.spans.push({ segment, start, end }); this.nextAudioTime = end;
      this.sources.add(source);
      source.onended = () => { source.disconnect(); this.sources.delete(source); };
      source.start(start); this.waitingSince = null;
    }
  }

  private tick(): void {
    this.frame = null;
    const audio = this.audio;
    if (!this.id || !audio) return;
    const now = this.runtime.now();
    if (this.gate.hasPending() && audio.currentTime >= this.nextAudioTime) {
      this.waitingSince ??= now;
      const pending = this.gate.pendingSegments()[0]!;
      if (this.failedSegments.has(pending) || this.providerDone) {
        this.stop("这一句没拿到有效的表情，朗读已停止。"); return;
      }
      if (now - this.waitingSince >= 5000) {
        this.stop("等待表情超过 5 秒，朗读已停止，请重试。"); return;
      }
    }
    const span = this.spans.find(item => item.start <= audio.currentTime && audio.currentTime < item.end);
    if (!span && this.activeSegment >= 0 && this.audioFinished.has(this.activeSegment)
      && this.spans.filter(item => item.segment === this.activeSegment).every(item => item.end <= audio.currentTime)) {
      this.timeline.finish(this.activeSegment); this.activeSegment = -1;
    }
    if (span && span.segment !== this.activeSegment) {
      this.activeSegment = span.segment;
      this.callbacks.sentence(this.segments[span.segment] ?? "");
      const targets = this.timeline.enter(span.segment);
      if (targets) this.apply(targets);
      this.callbacks.status("正在朗读 · 表情按句同步", true);
    }
    if (now - this.displayAt >= 200) {
      this.displayAt = now;
      const face = this.actualFace(); this.callbacks.values(face);
    }
    if (this.providerDone && !this.gate.hasPending() && (!this.spans.length || audio.currentTime > this.nextAudioTime + 0.45)) {
      this.stop("朗读完成，可以输入其他文字或切换模型继续测试。"); return;
    }
    this.frame = this.runtime.requestFrame(() => this.tick());
  }
}
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
