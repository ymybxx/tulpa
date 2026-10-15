import type { NativeModelParameter } from "../shared/contracts";
import type { FacialMotionBindings, SpeechPart } from "../shared/facial-bindings";
import catalog from "../shared/facial-demo-parameters.json";
import { FacialTransition, transitionProfile, type MotionState, type TransitionProfile } from "./facial-transition";

export interface FacialDemoSnapshot {
  active: boolean;
  values: Record<string, number>;
  targets: Record<string, number>;
}

interface Channel {
  value: number;
  velocity: number;
  target: number;
  baseline: number;
  minimum: number;
  maximum: number;
}
interface ExpressionChannel {
  motion: FacialTransition;
  baseline: number;
  minimum: number;
  maximum: number;
  /** Sway or reach-and-return instead of holding the chosen step. */
  style?: "swing" | "pulse";
  /** How long a pulse stays at its step: slow parts (arms, body) linger longer. */
  hold: number;
  gesture: Gesture | null;
  /** The family that moves together (both eyes, both brows), for shared variation. */
  group: string;
  /** When it starts after a new step: with the face, after it (head), or after the head (body). */
  role: "face" | "head" | "body";
  /** The step asked for, before variation. */
  step: number;
  /** Where this channel moves once its turn comes. */
  goal: number;
  /** Seconds before `goal` starts, or null once it has. */
  pending: number | null;
}

/**
 * Spoken expressions move like a person rather than switching poses: the face
 * changes first, the head turns after it and the body follows the head. Jev
 * answers the same line the same way every time, so to keep one mood from
 * looking stamped each sentence's parts also reach a little further or
 * shorter and start a moment apart; a family of parts (both eyes, both brows)
 * shares one draw so it stays symmetric.
 */
export interface ExpressionVariation {
  /** Largest share a step may grow or shrink by, e.g. 0.2 for ±20%. */
  amplitude: number;
  /** Seconds after the face before the head and then the body start toward a new step. */
  onset: { head: number; body: number };
  /** Longest extra wait, in seconds, drawn per family on top of its onset. */
  onsetSpread: number;
  random?: () => number;
}

/**
 * Within a sentence the face follows the voice, as a speaker's does: louder
 * words carry the expression fully and quieter ones let it ease back a little,
 * the head and brows follow the intonation, and a pause lets the eyes blink.
 * All of it is read from the played audio, so it works the same for any model
 * and any line. Nods are not made here: whether to nod depends on what is
 * said, which the decision model judges.
 */
export interface SpeechDynamics {
  /** Share of each step kept on the quietest words, by how the part moves (face, head, body). */
  floor: { face: number; head: number; body: number };
  /** Seconds of quiet after speaking that count as a pause and bring a blink. */
  pause: number;
  /** Fewest seconds between two such blinks. */
  blinkGap: number;
  /**
   * Intonation: degrees the head lifts per semitone above the speaker's own
   * pitch (and drops below it), up to `headLimit`; brow lift per semitone of
   * rise, up to `browLimit`. Brows rise with the voice, they do not sink with it.
   */
  intonation: { head: number; headLimit: number; brow: number; browLimit: number };
  /**
   * Glancing away to think, as speakers do: the chance as a turn starts after
   * a silence and at a later pause, and the fewest seconds between two glances.
   */
  glance: { start: number; pause: number; gap: number };
  /** Where the speaker's own pitch (log2 Hz) is kept from one sentence to the next: TA keeps one voice. */
  memory?: { key: number | null };
  random?: () => number;
}

/** A look away from the listener and back. */
interface Glance { time: number; hold: number; x: number; y: number }

/** The voice as the face hears it: smoothed loudness and the moments it marks. */
interface Voice {
  level: number;
  smooth: number;
  /** Recent loudest, falling off slowly, so effort is relative to this speaker and line. */
  peak: number;
  /** How fully the expression shows now, 0..1. */
  effort: number;
  heard: boolean;
  /** Speaking since the last pause, so the next pause may blink. */
  spoke: boolean;
  quiet: number;
  sinceBlink: number;
  /** Seconds into the current blink, or null when none is under way. */
  blink: number | null;
  /** Hz while voiced, else 0. */
  pitch: number;
  /** The speaker's own pitch as log2 Hz, followed slowly; null until heard. */
  key: number | null;
  /** Semitones above (or below) the speaker's own pitch, smoothed; the head follows it more slowly than the brows. */
  tone: number;
  headTone: number;
  /** Seconds since the voice was last voiced, and voiced in all. */
  unvoiced: number;
  voicedTime: number;
  /** The speaker's pitch was already known from an earlier sentence. */
  known: boolean;
  glance: Glance | null;
  sinceGlance: number;
}

/** A speaker turns a new turn after this many seconds of silence. */
const TURN_SILENCE = 1;
/** How slowly the head follows the intonation. */
const HEAD_TONE_SECONDS = 0.3;
/** A glance away jumps out like a saccade and drifts back a little slower. */
const GLANCE_OUT = 0.07;
const GLANCE_BACK = 0.12;

const BLINK_CLOSE = 0.08;
/** Below this the voice counts as paused. */
const PAUSE_LEVEL = 0.15;
const BLINK_OPEN = 0.14;
const FACE_GROUPS = new Set(["eyes", "brows", "mouth", "cheek", "face"]);
const smoothstep = (t: number) => { const x = clamp(t, 0, 1); return x * x * (3 - 2 * x); };
/** Down and back up, 0 → 1 → 0, for a blink `t` seconds in. */
const dip = (t: number, down: number, up: number) => t < down ? smoothstep(t / down) : 1 - smoothstep((t - down) / up);
/** Out, held, and back: 0 → 1, staying at 1 for `hold` seconds, then → 0. */
const held = (t: number, out: number, hold: number, back: number) => t < out ? smoothstep(t / out) : 1 - smoothstep((t - out - hold) / back);

/** Live2D's standard body turns; a VTS author ties them to the head, and they follow it. */
const BODY_PARAMETERS = new Set(["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"]);
const HEAD_GROUPS = new Set(["head", "arms", "hands"]);
const BODY_GROUPS = new Set(["body", "overall"]);
/** A part's movement toward and around one chosen step, until it rests again. */
interface Gesture {
  goal: number;
  /** Where the part is heading now: the goal, or back toward rest. */
  heading: number;
  /** Seconds spent at `heading`. */
  dwell: number;
  swings: number;
  /** Done and going back to rest; once there, a new sentence may start again. */
  returning: boolean;
}

/** A reach holds at least this long before going back; slower parts hold longer. */
const PULSE_HOLD = 0.35;
/** Each swing turns back this close to rest, so the part keeps moving without dropping. */
const SWING_RETURN = 0.35;
/** Half-swings before the part settles back to rest: three waves. */
const SWING_COUNT = 6;

const MOUTH_OPEN = "ParamMouthOpenY";
const groups = new Map(catalog.map(parameter => [parameter.id, parameter.group]));
const clamp = (value: number, minimum: number, maximum: number) => Math.max(minimum, Math.min(maximum, value));

/** A swing reads as a wave, not a tremble, whatever the part: each stroke takes a visible moment. */
function gestureProfile(profile: TransitionProfile, motion?: "swing" | "pulse"): TransitionProfile {
  const stroke = motion === "swing" ? 0.3 : motion === "pulse" ? 0.2 : 0;
  return { ...profile, minimumDuration: Math.max(profile.minimumDuration, stroke) };
}

/** Keep the audio-driven mouth's fast, critically damped response independent of pose transitions. */
function advance(channel: Channel, delta: number, frequency: number): void {
  const offset = channel.value - channel.target;
  const impulse = (channel.velocity + frequency * offset) * delta;
  const decay = Math.exp(-frequency * delta);
  const value = channel.target + (offset + impulse) * decay;
  channel.velocity = (channel.velocity - frequency * impulse) * decay;
  channel.value = clamp(value, channel.minimum, channel.maximum);
  if (channel.value !== value) channel.velocity = 0;
}

/**
 * Opt-in demo targets over a captured editing pose. Only real Core parameter IDs
 * are admitted; no Cubism ID lookup can accidentally create virtual parameters.
 */
export class FacialParameterController {
  private readonly definitions = new Map<string, NativeModelParameter>();
  private readonly channels = new Map<string, ExpressionChannel>();
  private targets: Record<string, number> = {};
  private readonly speech: Channel = { value: 0, velocity: 0, target: 0, baseline: 0, minimum: 0, maximum: 1 };
  private speechEnabled = false;
  private readonly mouth: FacialMotionBindings["mouth"];
  private readonly closedEyes: ReadonlyMap<string, number>;
  private readonly voice: Voice = { level: 0, smooth: 0, peak: 0, effort: 1, heard: false, spoke: false, quiet: Infinity, sinceBlink: Infinity, blink: null,
    pitch: 0, key: null, tone: 0, headTone: 0, unvoiced: Infinity, voicedTime: 0, known: false, glance: null, sinceGlance: Infinity };
  /** How far each speaking part is moved this frame, in its own units. */
  private readonly offsets: Record<SpeechPart, number> = { headPitch: 0, eyeX: 0, eyeY: 0, brow: 0 };

  constructor(parameters: readonly NativeModelParameter[], private readonly baseline: Readonly<Record<string, number>>, mouth?: FacialMotionBindings["mouth"], private readonly transitions?: FacialMotionBindings["transitions"], private readonly variation?: ExpressionVariation,
    private readonly dynamics?: SpeechDynamics, eyes: FacialMotionBindings["eyes"] = []) {
    for (const parameter of parameters) this.definitions.set(parameter.id, parameter);
    this.mouth = mouth === undefined ? { id: MOUTH_OPEN, closed: 0, open: 1 } : mouth;
    this.closedEyes = new Map(dynamics ? eyes.map(eye => [eye.id, eye.closed]) : []);
    const remembered = dynamics?.memory?.key;
    if (remembered !== undefined && remembered !== null) { this.voice.key = remembered; this.voice.known = true; }
    // Intonation, glances and blinks move parts the expression may not, so those take part from the start.
    if (dynamics) {
      for (const [id, transition] of Object.entries(transitions ?? {})) if (transition.follow) this.channel(id);
      for (const id of this.closedEyes.keys()) this.channel(id);
    }
  }

  /** Each update is a complete target snapshot; removed channels return smoothly to the editing pose. */
  setTargets(targets: Readonly<Record<string, number>>): void {
    const accepted: Record<string, number> = {};
    for (const [id, value] of Object.entries(targets)) {
      if (!Number.isFinite(value)) continue;
      const channel = this.channel(id);
      if (!channel) continue;
      accepted[id] = clamp(value, channel.minimum, channel.maximum);
    }
    this.targets = accepted;
    const variation = this.variation;
    const random = variation?.random ?? Math.random;
    const draws = new Map<string, { scale: number; wait: number }>();
    for (const [id, channel] of this.channels) {
      const step = accepted[id] ?? channel.baseline;
      if (!variation) { this.start(channel, step); continue; }
      // An unchanged step keeps its own course: a repeated answer does not restart the part.
      if (step === channel.step) continue;
      channel.step = step;
      const draw = draws.get(channel.group) ?? { scale: 1 + (random() * 2 - 1) * variation.amplitude, wait: random() * variation.onsetSpread };
      draws.set(channel.group, draw);
      channel.goal = clamp(channel.baseline + (step - channel.baseline) * draw.scale, channel.minimum, channel.maximum);
      channel.pending = draw.wait + (channel.role === "face" ? 0 : variation.onset[channel.role]);
    }
  }

  private start(channel: ExpressionChannel, goal: number): void {
    if (!this.variation) channel.step = goal;
    channel.goal = goal;
    channel.pending = null;
    if (channel.style) this.gesture(channel, goal);
    else channel.motion.setTarget(goal);
  }

  setSpeechLevel(level: number): void {
    if (!Number.isFinite(level) || !this.mouth || !this.channel(this.mouth.id) || this.mouth.open === this.mouth.closed) return;
    this.speechEnabled = true;
    this.speech.target = clamp(level, 0, 1);
  }

  /** Loudness (0..1) and pitch (Hz, 0 when unvoiced) of the voice being played now. */
  hearVoice(level: number, pitch = 0): void {
    if (!this.dynamics) return;
    if (Number.isFinite(level)) this.voice.level = clamp(level, 0, 1);
    this.voice.pitch = Number.isFinite(pitch) && pitch > 0 ? pitch : 0;
  }

  update(deltaSeconds: number): Readonly<Record<string, number>> {
    const delta = Number.isFinite(deltaSeconds) ? Math.max(0, deltaSeconds) : 0;
    if (this.dynamics) this.listen(this.dynamics, delta);
    for (const channel of this.channels.values()) {
      if (channel.pending !== null) {
        channel.pending -= delta;
        if (channel.pending <= 0) this.start(channel, channel.goal);
      }
      if (channel.gesture) this.advanceGesture(channel, channel.gesture, delta);
      channel.motion.update(delta);
    }
    advance(this.speech, delta, 30);
    return this.values();
  }

  /** Every part has arrived at rest and the mouth has closed: nothing left to show. */
  settled(): boolean {
    for (const channel of this.channels.values()) {
      if (channel.pending !== null || channel.gesture) return false;
      if (this.voice.blink !== null || this.voice.glance !== null) return false;
      const { position, velocity } = channel.motion.snapshot(), span = channel.maximum - channel.minimum;
      if (Math.abs(position - channel.baseline) > span * 1e-3 || Math.abs(velocity) > span * 1e-2) return false;
    }
    return !this.speechEnabled || this.speech.value < 1e-3;
  }

  snapshot(): { values: Record<string, number>; targets: Record<string, number> } {
    const targets = { ...this.targets };
    const mouth = this.mouth && this.channels.get(this.mouth.id);
    if (this.speechEnabled && mouth && this.mouth) targets[this.mouth.id] = this.withSpeech(targets[this.mouth.id] ?? mouth.baseline, this.speech.target, mouth);
    return { values: this.values(), targets };
  }

  /** Local diagnostics for motion continuity; never sent to JEV. */
  kinematics(): Record<string, MotionState> {
    return Object.fromEntries([...this.channels].map(([id, channel]) => [id, channel.motion.snapshot()]));
  }

  private values(): Record<string, number> {
    const values: Record<string, number> = {};
    for (const [id, channel] of this.channels) values[id] = this.dynamics ? this.voiced(id, channel, this.dynamics) : channel.motion.value;
    const mouth = this.mouth && this.channels.get(this.mouth.id);
    if (this.speechEnabled && mouth && this.mouth) values[this.mouth.id] = this.withSpeech(mouth.motion.value, this.speech.value, mouth);
    return values;
  }

  private listen(dynamics: SpeechDynamics, delta: number): void {
    const voice = this.voice;
    const follow = (from: number, to: number, seconds: number) => from + (to - from) * (1 - Math.exp(-delta / seconds));
    voice.smooth = follow(voice.smooth, voice.level, voice.level > voice.smooth ? 0.08 : 0.3);
    voice.peak = Math.max(voice.smooth, voice.peak * Math.exp(-delta / 1.5));
    if (voice.smooth > 0.05) voice.heard = true;
    // Before the voice starts the chosen step shows in full; after, it rides on the effort.
    if (voice.heard) voice.effort = follow(voice.effort, clamp(voice.smooth / Math.max(voice.peak, 0.15), 0, 1), 0.25);
    voice.sinceBlink += delta;
    if (voice.blink !== null && (voice.blink += delta) >= BLINK_CLOSE + BLINK_OPEN) voice.blink = null;
    this.intone(dynamics, delta, follow);
    voice.sinceGlance += delta;
    if (voice.glance && (voice.glance.time += delta) >= GLANCE_OUT + voice.glance.hold + GLANCE_BACK) voice.glance = null;
    const away = voice.glance ? held(voice.glance.time, GLANCE_OUT, voice.glance.hold, GLANCE_BACK) : 0;
    this.offsets.eyeX = (voice.glance?.x ?? 0) * away;
    this.offsets.eyeY = (voice.glance?.y ?? 0) * away;
    // A head is heavy: it rides the tone's slow course, not every syllable.
    voice.headTone = follow(voice.headTone, voice.tone, HEAD_TONE_SECONDS);
    this.offsets.headPitch = clamp(voice.headTone * dynamics.intonation.head, -dynamics.intonation.headLimit, dynamics.intonation.headLimit);
    this.offsets.brow = clamp(voice.tone * dynamics.intonation.brow, 0, dynamics.intonation.browLimit);
    // A pause is heard as it starts, not once the smoothed effort has died away; the mouth's
    // own loudness takes a moment to fall, so a little left still counts as quiet.
    if (voice.level >= PAUSE_LEVEL) {
      if (voice.quiet >= TURN_SILENCE) this.maybeGlance(dynamics, dynamics.glance.start);
      voice.quiet = 0; if (voice.smooth > 0.2) voice.spoke = true; return;
    }
    voice.quiet += delta;
    if (!voice.spoke || voice.quiet < dynamics.pause) return;
    voice.spoke = false;
    if (voice.blink === null && voice.sinceBlink >= dynamics.blinkGap) { voice.blink = 0; voice.sinceBlink = 0; }
    this.maybeGlance(dynamics, dynamics.glance.pause);
  }

  /** Follows the voice's pitch against the speaker's own, so any voice, high or low, lifts the head the same. */
  private intone(dynamics: SpeechDynamics, delta: number, follow: (from: number, to: number, seconds: number) => number): void {
    const voice = this.voice;
    if (voice.pitch > 0 && voice.level >= PAUSE_LEVEL) {
      const tone = Math.log2(voice.pitch);
      // The speaker's own pitch is the running average: quick to form over the first words, then
      // steady. TA keeps one voice, so the last sentence's carries over to the next.
      voice.voicedTime += delta;
      voice.key = voice.key === null ? tone : follow(voice.key, tone, Math.min(2, 0.2 + voice.voicedTime));
      if (this.dynamics?.memory) this.dynamics.memory.key = voice.key;
      voice.unvoiced = 0;
      // Until the speaker's pitch is known, its tone counts for less; a misheard octave is no more than a big step.
      const known = voice.known ? 1 : Math.min(1, voice.voicedTime);
      voice.tone = follow(voice.tone, clamp(12 * (tone - voice.key), -8, 8) * known, 0.12);
      return;
    }
    // Through a consonant the head holds its tone; in a pause it settles back.
    voice.unvoiced += delta;
    if (voice.unvoiced > 0.15) voice.tone = follow(voice.tone, 0, 0.5);
  }

  private maybeGlance(dynamics: SpeechDynamics, chance: number): void {
    const voice = this.voice, random = dynamics.random ?? Math.random;
    if (voice.glance || voice.sinceGlance < dynamics.glance.gap || random() >= chance) return;
    // Aside, and more often up than down, as when looking for the words.
    const side = random() < 0.5 ? -1 : 1;
    voice.glance = { time: 0, hold: 0.5 + random() * 0.6, x: side * (0.3 + random() * 0.2), y: -0.15 + random() * 0.4 };
    voice.sinceGlance = 0;
  }

  private voiced(id: string, channel: ExpressionChannel, dynamics: SpeechDynamics): number {
    const voice = this.voice;
    const floor = channel.group === "gaze" ? 1 : channel.role !== "face" ? dynamics.floor[channel.role] : FACE_GROUPS.has(channel.group) ? dynamics.floor.face : 1;
    let value = channel.baseline + (channel.motion.value - channel.baseline) * (floor + (1 - floor) * voice.effort);
    const part = this.transitions?.[id]?.follow;
    if (part) { const offset = this.offsets[part.part]; value += offset * (offset >= 0 ? part.up : part.down); }
    const closed = this.closedEyes.get(id);
    if (closed !== undefined && voice.blink !== null) value += (closed - value) * dip(voice.blink, BLINK_CLOSE, BLINK_OPEN);
    return clamp(value, channel.minimum, channel.maximum);
  }

  private withSpeech(value: number, level: number, channel: ExpressionChannel): number {
    const { closed, open } = this.mouth!;
    const expression = clamp((value - closed) / (open - closed), 0, 1);
    return clamp(closed + (open - closed) * Math.max(expression, level), channel.minimum, channel.maximum);
  }

  /** A new step starts a gesture; the same step while one is under way lets it finish. */
  private gesture(channel: ExpressionChannel, goal: number): void {
    const still = Math.abs(goal - channel.baseline) <= (channel.maximum - channel.minimum) * 0.02;
    if (still) { channel.gesture = null; channel.motion.setTarget(channel.baseline); return; }
    if (channel.gesture?.goal === goal) return;
    channel.gesture = { goal, heading: goal, dwell: 0, swings: 0, returning: false };
    channel.motion.setTarget(goal);
  }

  private advanceGesture(channel: ExpressionChannel, gesture: Gesture, delta: number): void {
    const amplitude = gesture.goal - channel.baseline;
    const arrived = Math.abs(channel.motion.value - gesture.heading) <= Math.abs(amplitude) * 0.08;
    gesture.dwell = arrived ? gesture.dwell + delta : 0;
    if (!arrived) return;
    if (gesture.returning) { channel.gesture = null; return; }
    const rest = () => { gesture.returning = true; gesture.heading = channel.baseline; channel.motion.setTarget(channel.baseline); };
    if (channel.style === "pulse") {
      if (gesture.dwell >= channel.hold) rest();
      return;
    }
    gesture.dwell = 0;
    if (++gesture.swings >= SWING_COUNT) { rest(); return; }
    gesture.heading = gesture.heading === gesture.goal ? channel.baseline + amplitude * SWING_RETURN : gesture.goal;
    channel.motion.setTarget(gesture.heading);
  }

  private channel(id: string): ExpressionChannel | null {
    const existing = this.channels.get(id);
    if (existing) return existing;
    const definition = this.definitions.get(id);
    if (!definition) return null;
    const baseline = clamp(this.baseline[id] ?? definition.defaultValue, definition.minimum, definition.maximum);
    const transition = this.transitions?.[id];
    const group = transition?.group ?? groups.get(id) ?? "face";
    const role = BODY_PARAMETERS.has(id) || BODY_GROUPS.has(group) ? "body" : HEAD_GROUPS.has(group) ? "head" : "face";
    const profile = transitionProfile(BODY_PARAMETERS.has(id) ? "body" : group);
    const channel: ExpressionChannel = { baseline, minimum: definition.minimum, maximum: definition.maximum,
      style: transition?.motion, gesture: null, hold: Math.max(PULSE_HOLD, profile.minimumDuration * 2.5),
      group, role, step: baseline, goal: baseline, pending: null,
      motion: new FacialTransition(baseline, definition.minimum, definition.maximum,
        gestureProfile(profile, transition?.motion),
        transition ? transition.maximum - transition.minimum : definition.maximum - definition.minimum) };
    this.channels.set(id, channel);
    return channel;
  }
}
