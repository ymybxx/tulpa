import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";
import type { NativeModelParameter } from "../shared/contracts";
import type { FacialOverlayBindings } from "../shared/facial-bindings";
import { FacialParameterController, type ExpressionVariation, type SpeechDynamics } from "./facial-parameter-controller";

const FADE_SECONDS = 0.3;
/** Longest a release waits for the parts to walk home before fading what is left. */
const RETURN_LIMIT_SECONDS = 4;
/**
 * The head starts 0.15 s after the face, the body 0.3 s; each family a little
 * later still at random, and reaching up to 20% further or shorter.
 */
export const SPEECH_VARIATION: ExpressionVariation = { amplitude: 0.2, onset: { head: 0.15, body: 0.3 }, onsetSpread: 0.08 };
/**
 * Quiet words keep 60% of the face's step and 85% of the head's and body's.
 * A pause at a comma or full stop (the voice quiet about 0.2 s) brings a
 * blink, at most one every 1.5 s. The head lifts 0.5° per semitone the voice rises above its
 * own pitch and drops as it falls, within 3°; brows lift with a rising voice.
 * A turn after a silence starts with a glance away 60% of the time, a later
 * pause 25%, glances at least 2 s apart.
 */
export const SPEECH_DYNAMICS: SpeechDynamics = { floor: { face: 0.6, head: 0.85, body: 0.85 }, pause: 0.18, blinkGap: 1.5,
  intonation: { head: 0.5, headLimit: 3, brow: 0.04, browLimit: 0.25 }, glance: { start: 0.6, pause: 0.25, gap: 2 } };
/** How spoken expressions move: varied per sentence and following the voice within it. */
export const SPEECH_MOTION = { variation: SPEECH_VARIATION, dynamics: SPEECH_DYNAMICS };
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/** A transient expression layer in the SDK pipeline; it never saves a motion baseline. */
export class CubismFacialOverlay extends ICubismUpdater {
  readonly controller: FacialParameterController;
  private readonly channels: Array<{ id: string; index: number; neutral: number; min: number; max: number }>;
  private readonly eyes: ReadonlyMap<string, number>;
  private readonly gaze: ReadonlySet<string>;
  private values: Readonly<Record<string, number>> = {};
  private applied = new Map<number, number>();
  private weight = 1;
  private transition = { from: 1, to: 1, elapsed: FADE_SECONDS };
  /** Seconds since release, while the parts walk back to rest before the layer fades out. */
  private returning: number | null = null;

  constructor(parameters: readonly NativeModelParameter[], readonly bindings: FacialOverlayBindings, speech: { variation: ExpressionVariation; dynamics: SpeechDynamics } | null = SPEECH_MOTION) {
    // After blink, expressions, look, sway and defaults; before physics consumes inputs.
    super(CubismUpdateOrder.CubismUpdateOrder_Physics - 0.5);
    this.channels = parameters.flatMap((parameter, index) => {
      const neutral = bindings.neutralValues[parameter.id];
      return Number.isFinite(neutral) ? [{ id: parameter.id, index, neutral: neutral!, min: parameter.minimum, max: parameter.maximum }] : [];
    });
    const enabled = new Set(this.channels.map(channel => channel.id));
    // Spoken expressions vary a little each sentence; previews and the editor stay exact.
    this.controller = new FacialParameterController(parameters.filter(parameter => enabled.has(parameter.id)), bindings.neutralValues, bindings.mouth, bindings.transitions,
      speech?.variation, speech?.dynamics, bindings.eyes);
    this.eyes = new Map(bindings.eyes.map(eye => [eye.id, eye.closed]));
    this.gaze = new Set(bindings.gazeParameterIds);
  }

  /** At the end of speech every part walks back at its own pace — face first, body last — then the layer goes. */
  release(): void {
    if (this.returning !== null || this.transition.to === 0) return;
    this.returning = 0;
    this.controller.setTargets({});
    this.controller.setSpeechLevel(0);
  }
  resume(): void { this.returning = null; this.transition = { from: this.weight, to: 1, elapsed: 0 }; }
  get finished(): boolean { return this.transition.to === 0 && this.weight === 0; }

  onLateUpdate(model: CubismModel, delta: number): void {
    this.values = this.controller.update(delta);
    if (this.returning !== null) {
      this.returning += Math.max(0, delta);
      // Once home there is nothing left to fade; a part held off its rest fades out instead.
      if (this.controller.settled() || this.returning >= RETURN_LIMIT_SECONDS) {
        this.returning = null;
        this.transition = { from: this.weight, to: 0, elapsed: this.controller.settled() ? FADE_SECONDS : 0 };
      }
    }
    this.transition.elapsed = Math.min(FADE_SECONDS, this.transition.elapsed + Math.max(0, delta));
    const t = this.transition.elapsed / FADE_SECONDS;
    this.weight = this.transition.from + (this.transition.to - this.transition.from) * t * t * (3 - 2 * t);
    this.applied.clear();
    for (const channel of this.channels) {
      if (this.gaze.has(channel.id)) continue;
      const value = this.compose(channel, model.getParameterValueByIndex(channel.index));
      if (value === null) continue;
      model.setParameterValueByIndex(channel.index, value);
      this.applied.set(channel.index, value);
    }
  }

  /** Recompose parameters written by lip-sync after physics, instead of restoring
   * their pre-audio values. Demo speech and untouched rig channels keep the
   * existing cached pose; expression offsets must never be added twice. */
  finishFrame(model: CubismModel, lateParameters?: ReadonlyMap<string, { value: number; neutral: number }>): void {
    for (const channel of this.channels) {
      const late = lateParameters?.get(channel.id);
      const base = model.getParameterValueByIndex(channel.index);
      const value = late
        ? this.compose(channel, late.value, late.neutral, base)
        : this.gaze.has(channel.id) ? this.compose(channel, base)
        : this.applied.get(channel.index);
      if (value !== null && value !== undefined) model.setParameterValueByIndex(channel.index, value);
    }
  }

  private compose(channel: { id: string; neutral: number; min: number; max: number }, base: number, neutral = channel.neutral, fadeBase = base): number | null {
    const target = this.values[channel.id];
    if (!Number.isFinite(target) || this.weight === 0) return null;
    let value = base + (target! - neutral);
    const closed = this.eyes.get(channel.id);
    if (closed !== undefined && Math.abs(channel.neutral - closed) > 1e-6) {
      value = closed + (base - closed) * (target! - closed) / (channel.neutral - closed);
    }
    const mouth = this.bindings.mouth;
    if (mouth?.id === channel.id && mouth.open !== mouth.closed) {
      const span = mouth.open - mouth.closed;
      value = mouth.closed + span * Math.max((base - mouth.closed) / span, (target! - mouth.closed) / span);
    }
    return clamp(fadeBase + (value - fadeBase) * this.weight, channel.min, channel.max);
  }
}
