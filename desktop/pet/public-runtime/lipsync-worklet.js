// Mouth shapes from speech: loudness opens the mouth, the vowel shapes it.
// A vowel is told by its first two formants, the resonances of the vocal
// tract, which move with the mouth: F1 rises as the mouth opens (i/u low,
// a high) and F2 rises as the tongue moves forward and falls as the lips
// round (i high, u/o low). Both are read from an all-pole (LPC) fit of each
// short frame, independent of the voice's pitch.

const FRAME_SECONDS = 0.03;
const HOP_SECONDS = 0.015;
// Formants that tell vowels apart sit below about 3.5 kHz.
const ANALYSIS_RATE = 11_000;
// A voiced frame repeats itself one pitch period later; hiss does not.
const VOICED_PERIODICITY = 0.35;
// Through a voiceless consonant (s, sh, x, f, h) the teeth nearly meet.
const VOICELESS_OPENING = 0.15;
// Attack and release of the emitted shape, in seconds.
const ATTACK = 0.07;
const RELEASE = 0.18;

// Typical Mandarin formants (Hz), between adult male and female values.
// Sounds that share a mouth shape share a viseme: 饿 and 耶 are both e, the
// apical i of 资/知 is i, and 鱼's rounded ü is u.
const VOWELS = [
  ["a", 900, 1350],
  ["o", 580, 950],
  ["e", 560, 1300],
  ["e", 540, 2050],
  ["i", 310, 2450],
  ["i", 380, 1650],
  ["u", 360, 760],
  ["u", 320, 1950],
];

class FishMemoryLipSyncProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceRate = 16000;
    this.generation = null;
    this.window = new Float32Array(Math.round(this.sourceRate * FRAME_SECONDS));
    this.windowLength = 0;
    this.smoothed = { volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 };
    this.pitch = 0;
    this.dominantVowel = null;
    this.candidateVowel = null;
    this.candidateFrames = 0;
    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  onMessage(message) {
    if (message?.type === "reset") {
      this.generation = typeof message.generation === "string"
        ? message.generation
        : null;
      this.windowLength = 0;
      this.smoothed = { volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 };
      this.pitch = 0;
      this.dominantVowel = null;
      this.candidateVowel = null;
      this.candidateFrames = 0;
      this.emitShape();
      return;
    }
    if (
      message?.type === "pcm" &&
      message.generation === this.generation &&
      message.pcm instanceof ArrayBuffer
    ) {
      this.sourceRate = Number(message.sampleRate) || 16000;
      const wanted = Math.max(160, Math.round(this.sourceRate * FRAME_SECONDS));
      if (wanted !== this.window.length) {
        this.window = new Float32Array(wanted);
        this.windowLength = 0;
      }
      const hop = Math.round(this.sourceRate * HOP_SECONDS);
      for (const sample of new Int16Array(message.pcm)) {
        this.window[this.windowLength++] = sample / 32768;
        if (this.windowLength === this.window.length) {
          this.analyse();
          // Frames overlap by half: keep the newer half for the next one.
          this.window.copyWithin(0, hop);
          this.windowLength = this.window.length - hop;
        }
      }
    }
  }

  process(_inputs, outputs) {
    for (const output of outputs) {
      for (const channel of output) channel.fill(0);
    }
    return true;
  }

  analyse() {
    let squareSum = 0;
    let crossings = 0;
    for (let index = 0; index < this.window.length; index += 1) {
      const value = this.window[index];
      squareSum += value * value;
      if (index && (value >= 0) !== (this.window[index - 1] >= 0)) crossings += 1;
    }
    const rms = Math.sqrt(squareSum / this.window.length);
    const loudness = clamp((rms - 0.008) * 8.5);
    const factor = Math.max(1, Math.round(this.sourceRate / ANALYSIS_RATE));
    const samples = decimate(this.window, factor);
    // Hissing consonants (s, sh, x) are loud noise with no pitch and no
    // formants: the mouth keeps its vowel's shape but stays nearly closed.
    const repeat = loudness >= 0.025 && crossings / (this.window.length / this.sourceRate) < 3000
      ? periodicity(samples, this.sourceRate / factor)
      : { strength: 0, pitch: 0 };
    const voiced = repeat.strength >= VOICED_PERIODICITY;
    // The voice's pitch rides along for the face: intonation moves the head and brows.
    this.pitch = voiced ? repeat.pitch : 0;
    const volume = voiced ? loudness : loudness * VOICELESS_OPENING;
    const heard = voiced ? classify(formants(samples, this.sourceRate / factor)) : null;
    if (volume < 0.025) {
      this.dominantVowel = null;
      this.candidateVowel = null;
      this.candidateFrames = 0;
    } else if (!heard) {
      // A frame without a clear reading (a consonant, a glide) neither
      // confirms nor interrupts the vowel being heard.
    } else if (heard === this.dominantVowel) {
      this.candidateVowel = null;
      this.candidateFrames = 0;
    } else if (heard === this.candidateVowel) {
      this.candidateFrames += 1;
      if (this.candidateFrames >= 2) {
        this.dominantVowel = heard;
        this.candidateVowel = null;
        this.candidateFrames = 0;
      }
    } else {
      this.candidateVowel = heard;
      this.candidateFrames = 1;
    }
    // A direct AEIOU rig expects one dominant viseme, not five small values at
    // once. Cross-fading is provided by the attack/release smoothing below.
    const target = { volume, a: 0, e: 0, i: 0, o: 0, u: 0 };
    if (this.dominantVowel) target[this.dominantVowel] = volume;
    // The mouth closes slowly as the voice fades, but snaps nearly shut for a
    // consonant; one vowel hands over to the next at the opening speed, or
    // short syllables never show.
    const closing = target.volume < this.smoothed.volume;
    for (const key of Object.keys(target)) {
      const quick = target[key] > this.smoothed[key] || !voiced || (key !== "volume" && !closing);
      const tau = quick ? ATTACK : RELEASE;
      this.smoothed[key] += (target[key] - this.smoothed[key]) * (1 - Math.exp(-HOP_SECONDS / tau));
    }
    this.emitShape();
  }

  emitShape() {
    this.port.postMessage({ generation: this.generation, shape: { ...this.smoothed, pitch: this.pitch } });
  }
}

/**
 * How strongly the frame repeats at a voice's pitch (70–400 Hz), 0..1, and
 * that pitch in Hz. A voice also repeats at twice its period, nearly as well,
 * so the shortest period that repeats almost as strongly is the pitch.
 */
function periodicity(samples, rate) {
  let mean = 0;
  for (const value of samples) mean += value;
  mean /= samples.length;
  let best = 0;
  const shortest = Math.floor(rate / 400);
  const longest = Math.min(Math.ceil(rate / 70), Math.floor(samples.length / 2));
  const scores = new Float64Array(Math.max(0, longest - shortest + 1));
  for (let lag = shortest; lag <= longest; lag += 1) {
    let product = 0;
    let head = 0;
    let tail = 0;
    for (let index = 0; index + lag < samples.length; index += 1) {
      const a = samples[index] - mean;
      const b = samples[index + lag] - mean;
      product += a * b;
      head += a * a;
      tail += b * b;
    }
    const score = head > 0 && tail > 0 ? product / Math.sqrt(head * tail) : 0;
    scores[lag - shortest] = score;
    best = Math.max(best, score);
  }
  for (let index = 1; index < scores.length; index += 1) {
    const score = scores[index];
    // A local peak close to the best.
    if (score >= best * 0.9 && score >= scores[index - 1] && score >= (scores[index + 1] ?? -Infinity)) {
      return { strength: best, pitch: rate / (shortest + index) };
    }
  }
  return { strength: best, pitch: 0 };
}

/** [F1, F2] in Hz from low-passed samples at analysisRate, or null when unclear. */
function formants(decimated, analysisRate) {
  const samples = Float64Array.from(decimated);
  const length = samples.length;
  for (let index = length - 1; index > 0; index -= 1) samples[index] -= 0.97 * samples[index - 1];
  for (let index = 0; index < length; index += 1) {
    samples[index] *= 0.54 - 0.46 * Math.cos((2 * Math.PI * index) / (length - 1));
  }
  const order = 2 + Math.round(analysisRate / 1000);
  const coefficients = lpc(samples, order);
  if (!coefficients) return null;
  const peaks = [];
  const [real, imaginary] = polynomialRoots(coefficients);
  for (let index = 0; index < real.length; index += 1) {
    if (imaginary[index] <= 0) continue;
    const frequency = (Math.atan2(imaginary[index], real[index]) * analysisRate) / (2 * Math.PI);
    const radius = Math.hypot(real[index], imaginary[index]);
    const bandwidth = (-Math.log(radius) * analysisRate) / Math.PI;
    if (frequency > 180 && frequency < analysisRate / 2 - 200 && bandwidth < 600) peaks.push(frequency);
  }
  peaks.sort((left, right) => left - right);
  const first = peaks.find((frequency) => frequency >= 200 && frequency <= 1150);
  if (first === undefined) return null;
  const second = peaks.find((frequency) => frequency > first + 150 && frequency <= 3500);
  return second === undefined ? null : [first, second];
}

/** The nearest vowel's viseme, or null when the pair is far from every vowel. */
function classify(pair) {
  if (!pair) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const [viseme, first, second] of VOWELS) {
    const distance = Math.log(pair[0] / first) ** 2 + Math.log(pair[1] / second) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = viseme;
    }
  }
  return bestDistance <= 0.36 ? best : null;
}

/** Low-pass (windowed sinc) and keep every factor-th sample. */
function decimate(frame, factor) {
  if (factor === 1) return Float64Array.from(frame);
  const half = 2 * factor;
  const taps = [];
  let total = 0;
  for (let offset = -half; offset <= half; offset += 1) {
    const x = offset / factor;
    const sinc = offset === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const tap = sinc * (0.54 + 0.46 * Math.cos((Math.PI * offset) / (half + 1)));
    taps.push(tap);
    total += tap;
  }
  const output = new Float64Array(Math.floor(frame.length / factor));
  for (let index = 0; index < output.length; index += 1) {
    let sum = 0;
    for (let tap = 0; tap < taps.length; tap += 1) {
      const source = index * factor + tap - half;
      if (source >= 0 && source < frame.length) sum += frame[source] * taps[tap];
    }
    output[index] = sum / total;
  }
  return output;
}

/** Autocorrelation LPC (Levinson-Durbin): [1, a1 … ap], or null for silence. */
function lpc(samples, order) {
  const correlation = new Float64Array(order + 1);
  for (let lag = 0; lag <= order; lag += 1) {
    let sum = 0;
    for (let index = lag; index < samples.length; index += 1) sum += samples[index] * samples[index - lag];
    correlation[lag] = sum;
  }
  if (correlation[0] <= 1e-12) return null;
  correlation[0] *= 1 + 1e-6;
  const coefficients = new Float64Array(order + 1);
  coefficients[0] = 1;
  let error = correlation[0];
  for (let step = 1; step <= order; step += 1) {
    let sum = correlation[step];
    for (let index = 1; index < step; index += 1) sum += coefficients[index] * correlation[step - index];
    const reflection = -sum / error;
    const previous = coefficients.slice();
    for (let index = 1; index < step; index += 1) coefficients[index] = previous[index] + reflection * previous[step - index];
    coefficients[step] = reflection;
    error *= 1 - reflection * reflection;
    if (error <= 0) return null;
  }
  return coefficients;
}

/** Roots of z^n + c1 z^(n-1) + … + cn (Durand-Kerner), as [real, imaginary]. */
function polynomialRoots(coefficients) {
  const degree = coefficients.length - 1;
  const real = new Float64Array(degree);
  const imaginary = new Float64Array(degree);
  for (let index = 0; index < degree; index += 1) {
    const angle = (2 * Math.PI * index) / degree + 0.4;
    real[index] = 0.9 * Math.cos(angle);
    imaginary[index] = 0.9 * Math.sin(angle);
  }
  for (let iteration = 0; iteration < 100; iteration += 1) {
    let largest = 0;
    for (let index = 0; index < degree; index += 1) {
      const zr = real[index];
      const zi = imaginary[index];
      let valueR = 1;
      let valueI = 0;
      for (let term = 1; term <= degree; term += 1) {
        const nextR = valueR * zr - valueI * zi + coefficients[term];
        valueI = valueR * zi + valueI * zr;
        valueR = nextR;
      }
      let denominatorR = 1;
      let denominatorI = 0;
      for (let other = 0; other < degree; other += 1) {
        if (other === index) continue;
        const differenceR = zr - real[other];
        const differenceI = zi - imaginary[other];
        const nextR = denominatorR * differenceR - denominatorI * differenceI;
        denominatorI = denominatorR * differenceI + denominatorI * differenceR;
        denominatorR = nextR;
      }
      const scale = denominatorR * denominatorR + denominatorI * denominatorI || 1e-30;
      const stepR = (valueR * denominatorR + valueI * denominatorI) / scale;
      const stepI = (valueI * denominatorR - valueR * denominatorI) / scale;
      real[index] = zr - stepR;
      imaginary[index] = zi - stepI;
      largest = Math.max(largest, Math.abs(stepR) + Math.abs(stepI));
    }
    if (largest < 1e-10) break;
  }
  return [real, imaginary];
}

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

registerProcessor("fish-memory-lipsync", FishMemoryLipSyncProcessor);
