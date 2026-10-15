import { readFile } from "node:fs/promises";
import vm from "node:vm";

import { describe, expect, it } from "vitest";

type Shape = Record<"volume" | "a" | "e" | "i" | "o" | "u", number> & { pitch?: number };
type Processor = {
  onMessage(message: unknown): void;
  port: { messages: Array<{ shape: Shape }> };
};

async function processor(): Promise<Processor> {
  const source = await readFile(new URL("../../public-runtime/lipsync-worklet.js", import.meta.url), "utf8");
  let Processor!: new () => Processor;
  class AudioWorkletProcessor {
    port = {
      messages: [] as Array<{ shape: Shape }>,
      postMessage: (value: { shape: Shape }) => this.port.messages.push({ shape: { ...value.shape } }),
      onmessage: null,
    };
  }
  vm.runInNewContext(source, {
    AudioWorkletProcessor,
    registerProcessor: (_name: string, value: new () => Processor) => { Processor = value; },
    Float32Array, Float64Array, Int16Array, ArrayBuffer, Math, Object, Infinity,
  });
  const instance = new Processor();
  instance.onMessage({ type: "reset", generation: "g" });
  return instance;
}

/** A vowel from a pulse train through formant resonators (source-filter model). */
function vowel(rate: number, pitch: number, formants: number[], seconds = 0.4): Int16Array {
  const length = Math.round(rate * seconds);
  let signal = Float64Array.from({ length }, (_value, index) =>
    index % Math.round(rate / pitch) === 0 ? 1 : 0);
  for (const [frequency, bandwidth] of formants.map((value, index) => [value, [80, 100, 150][index]!])) {
    const radius = Math.exp(-Math.PI * bandwidth! / rate);
    const c1 = 2 * radius * Math.cos(2 * Math.PI * frequency! / rate);
    const c2 = -radius * radius;
    const output = new Float64Array(length);
    for (let index = 0; index < length; index += 1) {
      output[index] = signal[index]! + c1 * (output[index - 1] ?? 0) + c2 * (output[index - 2] ?? 0);
    }
    signal = output;
  }
  const peak = Math.max(...signal.map(Math.abs));
  return Int16Array.from(signal, (value) => Math.round((value / peak) * 12_000));
}

function hiss(rate: number, seconds: number): Int16Array {
  let seed = 7;
  return Int16Array.from({ length: Math.round(rate * seconds) }, () => {
    seed = (seed * 1103515245 + 12345) % 2 ** 31;
    return Math.round((seed / 2 ** 31 - 0.5) * 12_000);
  });
}

function send(target: Processor, rate: number, samples: Int16Array): Shape {
  target.onMessage({ type: "pcm", generation: "g", sampleRate: rate, pcm: samples.slice().buffer });
  return target.port.messages.at(-1)!.shape;
}

function dominant(shape: Shape): string {
  return (["a", "e", "i", "o", "u"] as const).reduce((best, key) => (shape[key] > shape[best] ? key : best), "a");
}

describe("lipsync worklet", () => {
  it.each([
    [16_000, 120], [16_000, 240], [24_000, 120], [24_000, 240],
  ])("tells a, i and u apart by formants, not by pitch (%i Hz audio, %i Hz voice)", async (rate, pitch) => {
    for (const [expected, formants] of [["a", [850, 1300, 2800]], ["i", [300, 2400, 3100]], ["u", [350, 750, 2700]]] as const) {
      const target = await processor();
      const shape = send(target, rate, vowel(rate, pitch, [...formants]));
      expect(dominant(shape)).toBe(expected);
      expect((["a", "e", "i", "o", "u"] as const).filter((key) => shape[key] > 0.05)).toEqual([expected]);
    }
  });

  it("keeps the vowel's shape through a hissing consonant but nearly closes the mouth", async () => {
    const target = await processor();
    const voiced = send(target, 16_000, vowel(16_000, 200, [300, 2400, 3100]));
    const shape = send(target, 16_000, hiss(16_000, 0.15));
    expect(dominant(shape)).toBe("i");
    expect(shape.volume).toBeLessThan(voiced.volume * 0.3);
  });

  it.each([
    [16_000, 110], [16_000, 220], [16_000, 330], [24_000, 140], [24_000, 260],
  ])("hears the voice's pitch, not twice its period (%i Hz audio, %i Hz voice)", async (rate, pitch) => {
    for (const formants of [[850, 1300, 2800], [300, 2400, 3100]]) {
      const shape = send(await processor(), rate, vowel(rate, pitch, formants));
      expect(Math.abs(12 * Math.log2(shape.pitch! / pitch))).toBeLessThan(0.5);
    }
    expect(send(await processor(), rate, hiss(rate, 0.3)).pitch).toBe(0);
  });

  it("barely opens the mouth for a loud hiss on its own", async () => {
    const target = await processor();
    const shape = send(target, 16_000, hiss(16_000, 0.5));
    expect(shape.volume).toBeLessThan(0.15);
    expect(shape.a + shape.e + shape.i + shape.o + shape.u).toBeLessThan(0.05);
  });
});
