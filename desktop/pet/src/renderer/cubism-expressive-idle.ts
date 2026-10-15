import type { CubismModel } from "@cubism/model/cubismmodel";
import type { CubismPhysics } from "@cubism/physics/cubismphysics";
import { ICubismUpdater } from "@cubism/motion/icubismupdater";
import type { ExpressiveIdleSettings, IdleChannel, NaturalIdleSettings } from "../shared/contracts";
import { expressiveIdleDefaults, normalizeNaturalIdleSettings } from "../shared/natural-idle-settings";
import { ExpressiveIdleMotion, type ExpressivePose } from "./expressive-idle-motion";
import { centeredValue, IdlePhysicsOwnership, resolveIdleRig, type IdleBinding, type IdleRig } from "./idle-parameter-binding";
import { UNKNOWN_IDLE_OWNERSHIP } from "./idle-parameter-ownership";
class IdleStage extends ICubismUpdater {
  constructor(order: number, private readonly run: (delta: number) => void) { super(order); }
  onLateUpdate(_model: CubismModel, delta: number): void { this.run(delta); }
}
/** Rad/s of the head's turn toward the pointer: most of the way in about 0.4 s. */
const POINTER_HEAD_RATE = 10;
interface Handoff {
  owner: boolean;
  elapsed: number;
  from: number;
}
/** Independent pose/eye clocks with parameter ownership at the authored/physics boundaries. */
export class CubismExpressiveIdle {
  readonly stages: readonly ICubismUpdater[];
  private readonly motion: ExpressiveIdleMotion;
  private readonly physicsOwnership: IdlePhysicsOwnership;
  private readonly authored: Float32Array;
  private readonly last: Float32Array;
  private readonly beforeBlink: Float32Array;
  private readonly gains = new Map<number, number>();
  private readonly handoffs = new Map<number, Handoff>();
  private rig: IdleRig;
  private settings: NaturalIdleSettings = normalizeNaturalIdleSettings(undefined);
  private expressive = expressiveIdleDefaults("natural");
  private motions: ReadonlySet<string> = new Set();
  private expressions: ReadonlySet<string> = new Set();
  private defaults: ReadonlySet<string> = new Set();
  private gain = 0;
  private bodyIntensity = 1;
  private pose: Readonly<ExpressivePose> | null = null;
  private pointer = { x: 0, y: 0, remaining: 0, weight: 0 };
  /**
   * The head's share of a look at the pointer. The eyes jump straight to it;
   * the head follows within about half a second, and as it turns the eyes ease
   * back by the head's turn — eyes first, head after, as a person looks.
   */
  private readonly pointerHead = { x: 0, y: 0, vx: 0, vy: 0 };
  constructor(private readonly model: CubismModel, private readonly physics: CubismPhysics | null, random?: () => number) {
    this.motion = new ExpressiveIdleMotion(random);
    this.physicsOwnership = new IdlePhysicsOwnership(physics);
    this.rig = resolveIdleRig(model, physics, this.expressive);
    this.authored = new Float32Array(model.getParameterCount());
    this.last = new Float32Array(model.getModel().parameters.values);
    this.beforeBlink = new Float32Array(model.getParameterCount());
    this.stages = [new IdleStage(190, () => { if (this.isActive())
        this.beforeBlink.set(model.getModel().parameters.values); }),
      new IdleStage(250, () => this.restoreDrivenEyes()), new IdleStage(350, () => this.capture()),
      new IdleStage(550.5, dt => this.beforePhysics(dt)), new IdleStage(650, dt => this.afterPhysics(dt))];
  }
  configure(value: NaturalIdleSettings | undefined): void {
    const settings = normalizeNaturalIdleSettings(value);
    this.settings = settings;
    if (settings.expressive) {
      this.expressive = settings.expressive;
      this.rig = resolveIdleRig(this.model, this.physics, this.expressive);
      this.motion.configure(settings);
    }
  }
  setOwnership(motions: ReadonlySet<string>, expressions: ReadonlySet<string>, defaults: ReadonlySet<string>): void {
    this.motions = motions;
    this.expressions = expressions;
    this.defaults = defaults;
  }
  setGaze(x: number, y: number, source: "pointer" | "ambient"): void {
    if (source === "ambient")
      return;
    this.pointer.x = Math.max(-1, Math.min(1, x));
    this.pointer.y = Math.max(-1, Math.min(1, y));
    this.pointer.remaining = 1.8;
  }
  isActive(): boolean { return (this.settings.enabled && !!this.settings.expressive) || this.gain > 0.0001; }
  private restoreDrivenEyes(): void {
    if (!this.isActive())
      return;
    // Remove the SDK blink only from our mapped eyes, before expressions.
    // Unmapped eyes keep their original SDK behavior.
    for (const item of [this.rig.channels.eyeLOpen, this.rig.channels.eyeROpen]) {
      if (!item)
        continue;
      const current = this.model.getParameterValueByIndex(item.index);
      this.model.setParameterValueByIndex(item.index, current + (this.beforeBlink[item.index]! - current) * this.gain);
    }
  }
  release(): void { this.physicsOwnership.restore(); }
  diagnostics(): object {
    return { enabled: this.settings.enabled, expressive: !!this.settings.expressive, gain: this.gain,
      bindingIds: Object.fromEntries(Object.entries(this.rig.channels).map(([key, value]) => [key, value.id])), ...this.motion.diagnostics() };
  }
  private capture(): void {
    if (this.isActive())
      this.authored.set(this.model.getModel().parameters.values);
  }
  private has(set: ReadonlySet<string>, id: string): boolean { return set.has(id) || set.has(UNKNOWN_IDLE_OWNERSHIP); }
  private owned(id: string): boolean { return this.has(this.motions, id) || this.has(this.expressions, id) || this.has(this.defaults, id); }
  private channelOwned(key: IdleChannel, item: IdleBinding): boolean {
    const visible = key === "headX" ? this.rig.visibleX : key === "headY" ? this.rig.visibleY : key === "headZ" ? this.rig.visibleZ : undefined;
    return this.owned(item.id) || !!(visible && this.owned(visible.id));
  }
  private beforePhysics(dt: number): void {
    if (!this.isActive())
      return;
    this.physicsOwnership.restore();
    const enabled = this.settings.enabled && !!this.settings.expressive;
    this.gain += ((enabled ? 1 : 0) - this.gain) * -Math.expm1(-dt / 0.18);
    if (!enabled && this.gain < 0.0001) {
      this.gain = 0;
      this.pose = null;
      return;
    }
    this.pose = this.motion.update(dt);
    this.bodyIntensity += (this.expressive.bodyIntensity - this.bodyIntensity) * -Math.expm1(-dt / 0.25);
    this.pointer.remaining = Math.max(0, this.pointer.remaining - dt);
    this.pointer.weight += ((this.pointer.remaining > 0 ? 1 : 0) - this.pointer.weight) * -Math.expm1(-dt / 0.18);
    this.followPointer(dt);
    for (const ids of [this.motions, this.expressions, this.defaults])
      for (const id of ids)
        this.physicsOwnership.reserve(id, this.gain);
    const p = this.pose;
    const heads = { headX: p.x + this.pointerHead.x * 8 * this.pointer.weight, headY: p.y + this.pointerHead.y * 5 * this.pointer.weight, headZ: p.z };
    for (const key of ["headX", "headY", "headZ"] as const) {
      const item = this.rig.channels[key];
      if (item)
        this.apply(item, centeredValue(item, heads[key] / 30), this.channelOwned(key, item), dt);
    }
    const bodies = { bodyX: (p.x * 0.3 * this.rig.directions.x + p.yawBias) * this.bodyIntensity,
      bodyY: (p.y * 0.22 * this.rig.directions.y + p.posture) * this.bodyIntensity,
      bodyZ: (p.z * 0.3 * this.rig.directions.z + p.lean) * this.bodyIntensity };
    for (const key of ["bodyX", "bodyY", "bodyZ"] as const) {
      const item = this.rig.channels[key];
      if (!item)
        continue;
      const target = centeredValue(item, bodies[key] / 10);
      const previous = this.bodyValues.get(item.index) ?? target;
      const next = previous + (target - previous) * -Math.expm1(-dt / this.expressive.followDelay);
      this.bodyValues.set(item.index, next);
      this.apply(item, next, this.owned(item.id), dt);
      this.physicsOwnership.reserve(item.id, this.gain);
    }
    const blink = p.blink;
    for (const key of ["eyeLOpen", "eyeROpen"] as const) {
      const item = this.rig.channels[key];
      if (!item)
        continue;
      const base = item.base > item.min ? item.base : item.max;
      const open = this.has(this.expressions, item.id) ? this.authored[item.index]! :
        item.min + (base - item.min) * p.open;
      const value = item.min + (open - item.min) * blink;
      // Expressions set the eye pose, while independent blinks can still multiply it.
      this.apply(item, value, this.has(this.motions, item.id) || this.has(this.defaults, item.id), dt, false);
    }
    for (const item of [this.rig.browL, this.rig.browR])
      if (item)
        this.apply(item, centeredValue(item, p.brow), this.owned(item.id), dt);
    if (this.rig.breath) {
      const item = this.rig.breath;
      this.apply(item, item.min + (item.max - item.min) * p.breath, this.owned(item.id), dt);
    }
  }
  private readonly bodyValues = new Map<number, number>();
  /** A critically damped turn toward the pointer: it starts gently, after the eyes, and does not overshoot. */
  private followPointer(dt: number): void {
    const head = this.pointerHead, stiffness = POINTER_HEAD_RATE * POINTER_HEAD_RATE;
    for (let left = dt; left > 0; left -= 1 / 120) {
      const step = Math.min(left, 1 / 120);
      head.vx += (stiffness * (this.pointer.x - head.x) - 2 * POINTER_HEAD_RATE * head.vx) * step;
      head.vy += (stiffness * (this.pointer.y - head.y) - 2 * POINTER_HEAD_RATE * head.vy) * step;
      head.x += head.vx * step; head.y += head.vy * step;
    }
  }
  private afterPhysics(dt: number): void {
    if (!this.pose || this.gain === 0)
      return;
    const p = this.pose;
    // A directly authored visible head output also wins over the rig's output.
    for (const [item, input] of [[this.rig.visibleX, this.rig.channels.headX], [this.rig.visibleY, this.rig.channels.headY], [this.rig.visibleZ, this.rig.channels.headZ]]) {
      if (item && item.index !== input?.index) {
        const owner = this.owned(item.id);
        const current = this.model.getParameterValueByIndex(item.index);
        this.writeHandoff(item, owner ? current + (this.authored[item.index]! - current) * this.gain : current, owner, dt);
      }
    }
    const normalized = (item: IdleBinding | undefined) => !item ? 0 :
      (this.model.getParameterValueByIndex(item.index) - item.base) / Math.max(0.0001, item.max - item.base, item.base - item.min);
    const aimX = p.aimX * (1 - this.pointer.weight) + this.pointer.x * 0.48 * this.pointer.weight;
    const aimY = p.aimY * (1 - this.pointer.weight) + this.pointer.y * 0.32 * this.pointer.weight;
    // Leave room for autonomous glances as well as head compensation. A low
    // output cap makes larger planned targets visually indistinguishable.
    const gaze = { eyeX: 0.9 * Math.tanh((aimX - normalized(this.rig.visibleX) * 0.62) / 0.9),
      eyeY: 0.9 * Math.tanh((aimY - normalized(this.rig.visibleY) * 0.34) / 0.9) };
    for (const key of ["eyeX", "eyeY"] as const) {
      const item = this.rig.channels[key];
      if (item)
        this.apply(item, centeredValue(item, gaze[key]), this.owned(item.id), dt);
    }
    this.last.set(this.model.getModel().parameters.values);
  }
  private apply(item: IdleBinding, target: number, owner: boolean, dt: number, handoff = true): void {
    const wanted = owner ? 0 : this.gain;
    const previous = this.gains.get(item.index) ?? (owner ? 0 : this.gain);
    const weight = previous + (wanted - previous) * -Math.expm1(-dt / 0.12);
    this.gains.set(item.index, weight);
    const current = this.model.getParameterValueByIndex(item.index);
    const base = owner ? current + (this.authored[item.index]! - current) * this.gain : current;
    const value = base + (target - base) * weight;
    if (handoff)
      this.writeHandoff(item, value, owner, dt);
    else
      this.model.setParameterValueByIndex(item.index, value);
  }
  private writeHandoff(item: IdleBinding, value: number, owner: boolean, dt: number): void {
    let state = this.handoffs.get(item.index);
    if (!state) {
      state = { owner, elapsed: 1, from: value };
      this.handoffs.set(item.index, state);
    }
    if (state.owner !== owner) {
      state.owner = owner;
      state.elapsed = 0;
      state.from = this.last[item.index]!;
    }
    state.elapsed += dt;
    const t = Math.min(1, state.elapsed / 0.18), blend = t * t * (3 - 2 * t);
    this.model.setParameterValueByIndex(item.index, state.from + (value - state.from) * blend);
  }
}
