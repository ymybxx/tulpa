import type { PetAudioFrame } from "../shared/contracts";
import { CLOSED_MOUTH, type LipShape } from "./pet-renderer";

interface LipSyncResult {
  generation: string | null;
  shape: LipShape;
}

export class RealtimeLipSync {
  private context: AudioContext | null = null;
  private worklet: AudioWorkletNode | null = null;
  private sampleRate = 16_000;
  private generation: string | null = null;
  private initializePromise: Promise<void> | null = null;
  private initializationFailed = false;
  private readonly pendingFrames: PetAudioFrame[] = [];

  constructor(private readonly onShape: (value: LipShape) => void) {}

  async initialize(): Promise<void> {
    if (this.initializePromise) return this.initializePromise;
    this.initializePromise = this.createWorklet().catch((error) => {
      this.initializationFailed = true;
      throw error;
    });
    return this.initializePromise;
  }

  start(generation: string, sampleRate: number): void {
    this.generation = generation;
    this.sampleRate = sampleRate;
    this.worklet?.port.postMessage({ type: "reset", generation });
    void this.initialize().then(() => this.context?.resume()).catch(() => undefined);
  }

  push(frame: PetAudioFrame): void {
    if (frame.generation !== this.generation) return;
    if (!this.worklet) {
      if (this.initializationFailed) {
        this.onShape(fallbackShape(frame.pcm));
        return;
      }
      this.pendingFrames.push(frame);
      // The audio is already playing elsewhere. Retain only a short onset so
      // late worklet startup never produces visibly delayed lip movement.
      if (this.pendingFrames.length > 2) this.pendingFrames.shift();
      void this.initialize().catch(() => undefined);
      return;
    }
    this.worklet.port.postMessage(
      {
        type: "pcm",
        generation: frame.generation,
        sampleRate: this.sampleRate,
        pcm: frame.pcm,
      },
      [frame.pcm],
    );
  }

  pause(generation?: string | null): void {
    if (generation && generation !== this.generation) return;
    this.pendingFrames.length = 0;
    this.worklet?.port.postMessage({
      type: "reset",
      generation: this.generation,
    });
    this.onShape(CLOSED_MOUTH);
  }

  stop(generation?: string | null): void {
    if (generation && generation !== this.generation) return;
    this.generation = null;
    this.pendingFrames.length = 0;
    this.worklet?.port.postMessage({ type: "reset", generation: null });
    this.onShape(CLOSED_MOUTH);
  }

  async close(): Promise<void> {
    this.stop();
    this.worklet?.disconnect();
    this.worklet = null;
    const context = this.context;
    this.context = null;
    if (context) await context.close();
  }

  private async createWorklet(): Promise<void> {
    const context = new AudioContext({ latencyHint: "interactive" });
    await context.audioWorklet.addModule("./lipsync-worklet.js");
    const worklet = new AudioWorkletNode(context, "fish-memory-lipsync", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const muted = context.createGain();
    muted.gain.value = 0;
    worklet.connect(muted).connect(context.destination);
    worklet.port.onmessage = (event: MessageEvent<LipSyncResult>) => {
      const shape = lipShapeForGeneration(this.generation, event.data);
      if (shape) this.onShape(shape);
    };
    this.context = context;
    this.worklet = worklet;
    worklet.port.postMessage({ type: "reset", generation: this.generation });
    for (const frame of this.pendingFrames.splice(0)) {
      if (frame.generation !== this.generation) continue;
      worklet.port.postMessage({
        type: "pcm",
        generation: frame.generation,
        sampleRate: this.sampleRate,
        pcm: frame.pcm,
      }, [frame.pcm]);
    }
    await context.resume();
  }
}

export function lipShapeForGeneration(
  currentGeneration: string | null,
  result: LipSyncResult,
): LipShape | null {
  return currentGeneration && result.generation === currentGeneration
    ? result.shape
    : null;
}

function fallbackShape(pcm: ArrayBuffer): LipShape {
  const samples = new Int16Array(pcm);
  if (!samples.length) return CLOSED_MOUTH;
  let squareSum = 0;
  for (const value of samples) {
    const normalized = value / 32768;
    squareSum += normalized * normalized;
  }
  const volume = Math.max(0, Math.min(1, (Math.sqrt(squareSum / samples.length) - 0.008) * 8.5));
  return { volume, a: volume, e: 0, i: 0, o: 0, u: 0 };
}

export function floatSamplesToPcm16(samples: Float32Array): ArrayBuffer {
  const result = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index] ?? 0));
    result[index] = value < 0
      ? Math.round(value * 32768)
      : Math.round(value * 32767);
  }
  return result.buffer;
}

/**
 * The mouth for audio this page plays: taps `input` as it is heard and runs
 * the lip sync TA speaks with, so a preview's mouth matches the desktop's.
 */
export function followPlayedAudio(
  audio: AudioContext,
  input: AudioNode,
  onShape: (value: LipShape) => void,
): { stop(): void } {
  const lipSync = new RealtimeLipSync(onShape);
  const generation = `played-audio-${Date.now()}`;
  let samplePosition = 0;
  lipSync.start(generation, audio.sampleRate);
  const tap = audio.createScriptProcessor(1024, 1, 1);
  const muted = audio.createGain();
  muted.gain.value = 0;
  tap.onaudioprocess = (event) => {
    const pcm = floatSamplesToPcm16(event.inputBuffer.getChannelData(0));
    lipSync.push({ generation, samplePosition, pcm });
    samplePosition += pcm.byteLength / 2;
  };
  input.connect(tap);
  tap.connect(muted).connect(audio.destination);
  return {
    stop: () => {
      tap.onaudioprocess = null;
      try { input.disconnect(tap); } catch { /* Already disconnected with its context. */ }
      tap.disconnect();
      muted.disconnect();
      void lipSync.close().catch(() => undefined);
    },
  };
}
