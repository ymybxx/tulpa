import type { LipShape } from "../renderer/pet-renderer";
import { floatSamplesToPcm16, RealtimeLipSync } from "../renderer/audio-lipsync";

export class MicrophoneLipSyncTest {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private lipSync: RealtimeLipSync | null = null;
  private samplePosition = 0;
  private version = 0;

  constructor(private readonly onShape: (value: LipShape) => void) {}

  async start(): Promise<void> {
    if (this.context) return;
    const version = ++this.version;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    if (version !== this.version) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    const context = new AudioContext({ latencyHint: "interactive" });
    const lipSync = new RealtimeLipSync(this.onShape);
    try {
      await lipSync.initialize();
      if (version !== this.version) {
        stream.getTracks().forEach(track => track.stop());
        await lipSync.close().catch(() => undefined);
        await context.close().catch(() => undefined);
        return;
      }
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(2048, 1, 1);
      const sink = context.createGain();
      sink.gain.value = 0;
      const generation = `model-lipsync-test-${Date.now()}`;
      lipSync.start(generation, context.sampleRate);
      processor.onaudioprocess = (event) => {
        const pcm = floatSamplesToPcm16(event.inputBuffer.getChannelData(0));
        const sampleCount = pcm.byteLength / 2;
        lipSync.push({ generation, samplePosition: this.samplePosition, pcm });
        this.samplePosition += sampleCount;
      };
      source.connect(processor);
      processor.connect(sink).connect(context.destination);
      this.context = context;
      this.stream = stream;
      this.source = source;
      this.processor = processor;
      this.sink = sink;
      this.lipSync = lipSync;
      this.samplePosition = 0;
      await context.resume();
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      await lipSync.close().catch(() => undefined);
      await context.close().catch(() => undefined);
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.version += 1;
    const context = this.context;
    const lipSync = this.lipSync;
    this.processor && (this.processor.onaudioprocess = null);
    this.source?.disconnect();
    this.processor?.disconnect();
    this.sink?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.context = null;
    this.stream = null;
    this.source = null;
    this.processor = null;
    this.sink = null;
    this.lipSync = null;
    this.samplePosition = 0;
    if (lipSync) await lipSync.close().catch(() => undefined);
    if (context) await context.close().catch(() => undefined);
  }
}

export { floatSamplesToPcm16 } from "../renderer/audio-lipsync";
