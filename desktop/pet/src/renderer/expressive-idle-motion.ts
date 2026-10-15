import type { ExpressiveIdleSettings, NaturalIdleSettings } from "../shared/contracts";
import { expressiveIdleDefaults, normalizeNaturalIdleSettings } from "../shared/natural-idle-settings";
import { IdleGazeMotion } from "./idle-gaze-motion";
type Axis = "x" | "y" | "z";
export interface ExpressivePose {
  x: number;
  y: number;
  z: number;
  lean: number;
  posture: number;
  yawBias: number;
  open: number;
  brow: number;
  aimX: number;
  aimY: number;
  blink: number;
  breath: number;
}
export interface IdlePoseEvent {
  time: number;
  kind: string;
  large: boolean;
  axis?: Axis;
  side?: number;
}
type PoseKey = "x" | "y" | "z" | "lean" | "posture" | "yawBias";
interface Knot extends Record<PoseKey, number> {
  time: number;
  pause: boolean;
}
const KEYS: readonly PoseKey[] = ["x", "y", "z", "lean", "posture", "yawBias"];
const styles = [
  { name: "attentive", gain: 1 },
  { name: "curious", gain: 1.07 },
  { name: "relaxed", gain: 0.94 },
  { name: "focused", gain: 0.96 },
];
const families: readonly {
  name: string;
  axis?: Axis;
  side?: number;
  x: number[];
  y: number[];
  z: number[];
}[] = [
  { name: "center", x: [-0.22, 0.22], y: [-0.38, 0.38], z: [-0.3, 0.3] },
  { name: "tilt-left", axis: "z", side: -1, x: [-0.4, 0.23], y: [-0.46, 0.46], z: [-1, -0.59] },
  { name: "tilt-right", axis: "z", side: 1, x: [-0.23, 0.4], y: [-0.46, 0.46], z: [0.59, 1] },
  { name: "look-left", axis: "x", side: -1, x: [-1, -0.55], y: [-0.46, 0.69], z: [0, 0.47] },
  { name: "look-right", axis: "x", side: 1, x: [0.55, 1], y: [-0.46, 0.69], z: [-0.47, 0] },
  { name: "dip", axis: "y", side: -1, x: [-0.41, 0.41], y: [-1, -0.54], z: [-0.35, 0.35] },
  { name: "lift", axis: "y", side: 1, x: [-0.36, 0.36], y: [0.54, 0.92], z: [-0.35, 0.35] },
];
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const ease = (x: number) => { const t = clamp(x, 0, 1); return t * t * t * (10 + t * (-15 + t * 6)); };
const between = (random: () => number, a: number, b: number) => a + (b - a) * random();
const choose = <T>(random: () => number, values: readonly T[]): T => values[Math.min(values.length - 1, Math.floor(random() * values.length))]!;
/** Independent streams prevent a pose configuration change from rescheduling blinks. */
function stream(random: () => number): () => number {
  let state = Math.floor(random() * 0xffffffff) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function tangent(a: Knot | undefined, b: Knot, c: Knot | undefined, key: PoseKey): number {
  if (!a || !c)
    return 0;
  const before = b.time - a.time, after = c.time - b.time;
  const left = (b[key] - a[key]) / before, right = (c[key] - b[key]) / after;
  if (left * right <= 0)
    return 0;
  const w1 = 2 * after + before, w2 = after + 2 * before;
  return (w1 + w2) / (w1 / left + w2 / right);
}
/** Bounded rolling spline: no prerecorded clip, time wrap, or ever-growing event history. */
export class ExpressiveIdleMotion {
  private settings: ExpressiveIdleSettings = expressiveIdleDefaults("natural");
  private readonly poseRandom: () => number;
  private readonly gaze: IdleGazeMotion;
  private readonly blinkRandom: () => number;
  private readonly knots: Knot[] = [{ time: 0, x: -3.5, y: -2, z: -3, lean: 0, posture: 0, yawBias: 0, pause: false }];
  private readonly recent: string[] = [];
  private readonly recentStyles: string[] = [];
  private readonly largeDirections: Partial<Record<Axis, {
    time: number;
    side: number;
  }>> = {};
  private style = styles[0]!;
  private nextStyle = 5;
  private time = 0;
  private realTime = 0;
  private speed = 1;
  private intensity = 1;
  private blinkAt = 3;
  private blink = { at: -10, close: 0.075, hold: 0.025, open: 0.18 };
  private doubleBlink = false;
  private readonly output: ExpressivePose = { x: 0, y: 0, z: 0, lean: 0, posture: 0, yawBias: 0, open: 1, brow: 0, aimX: 0, aimY: 0.035, blink: 1, breath: 0.5 };
  constructor(random: () => number = Math.random, private readonly onTarget?: (event: IdlePoseEvent) => void) {
    this.poseRandom = stream(random);
    this.gaze = new IdleGazeMotion(stream(random));
    this.blinkRandom = stream(random);
    this.blinkAt = between(this.blinkRandom, 2.1, 5);
  }
  configure(value: NaturalIdleSettings): void {
    this.config = normalizeNaturalIdleSettings(value);
    this.settings = this.config.expressive ?? expressiveIdleDefaults("natural");
    this.gaze.configure(this.settings.eyeActivity);
  }
  private config: NaturalIdleSettings = { enabled: false, intensity: 1, speed: 1 };
  update(delta: number): Readonly<ExpressivePose> {
    const dt = Number.isFinite(delta) ? clamp(delta, 0, 0.1) : 0;
    this.realTime += dt;
    this.speed += (this.config.speed - this.speed) * -Math.expm1(-dt / 0.35);
    this.intensity += (this.config.intensity - this.intensity) * -Math.expm1(-dt / 0.25);
    this.time += dt * this.speed;
    while (this.knots.length < 3 || this.knots[this.knots.length - 2]!.time <= this.time)
      this.append();
    while (this.knots.length > 3 && this.knots[2]!.time <= this.time)
      this.knots.shift();
    const i = this.knots[1]!.time <= this.time ? 1 : 0;
    const a = this.knots[i]!, b = this.knots[i + 1]!, span = b.time - a.time;
    const t = clamp((this.time - a.time) / span, 0, 1), t2 = t * t, t3 = t2 * t;
    for (const key of KEYS) {
      this.output[key] = (2 * t3 - 3 * t2 + 1) * a[key] + (t3 - 2 * t2 + t) * span * tangent(this.knots[i - 1], a, b, key)
        + (-2 * t3 + 3 * t2) * b[key] + (t3 - t2) * span * tangent(a, b, this.knots[i + 2], key);
    }
    for (const axis of ["x", "y", "z"] as const)
      this.output[axis] = clamp(this.output[axis] * this.intensity, -28, 28);
    const eyes = this.gaze.update(this.realTime);
    this.output.aimX = eyes.aimX; this.output.aimY = eyes.aimY;
    this.output.open = eyes.open; this.output.brow = eyes.brow;
    this.updateBlink();
    this.output.breath = 0.5 + Math.sin(this.realTime * 1.35) * 0.1;
    return this.output;
  }
  diagnostics(): {
    time: number;
    realTime: number;
    retainedKnots: number;
  } {
    return { time: this.time, realTime: this.realTime, retainedKnots: this.knots.length };
  }
  private append(): void {
    const last = this.knots.at(-1)!, r = this.poseRandom, settings = this.settings;
    if (last.time > 0 && !last.pause && r() < settings.pauseProbability) {
      this.knots.push({ ...last, time: last.time + between(r, 0.12, 0.3), pause: true });
      return;
    }
    if (last.time >= this.nextStyle) {
      this.style = choose(r, styles.filter(s => !this.recentStyles.includes(s.name)));
      this.recentStyles.push(this.style.name);
      if (this.recentStyles.length > 2)
        this.recentStyles.shift();
      this.nextStyle = last.time + between(r, 4, 7);
    }
    const large = r() < settings.largeMotionProbability;
    const candidates = families.filter(f => !this.recent.includes(f.name) && (!large || !f.axis ||
      !this.largeDirections[f.axis] || this.largeDirections[f.axis]!.side === f.side ||
      // Speed is bounded at 1.5, so this also preserves the minimum wall-time interval.
      last.time - this.largeDirections[f.axis]!.time >= settings.reverseInterval * 1.5));
    const family = choose(r, candidates.length ? candidates : families.filter(f => !this.recent.includes(f.name)));
    this.recent.push(family.name);
    if (this.recent.length > 2)
      this.recent.shift();
    const gain = this.style.gain * between(r, 0.94, 1.05) * (large ? 1 : 0.58);
    const point: Knot = {
      time: 0, pause: false,
      x: between(r, family.x[0]!, family.x[1]!) * settings.yawRange * gain,
      y: between(r, family.y[0]!, family.y[1]!) * settings.pitchRange * gain,
      z: between(r, family.z[0]!, family.z[1]!) * settings.rollRange * gain,
      lean: between(r, -0.7, 0.7), posture: between(r, -0.4, 0.9), yawBias: between(r, -0.65, 0.65),
    };
    const distance = Math.max(...(["x", "y", "z"] as const).map(key => Math.abs(point[key] - last[key])));
    const pace = settings.preset === "active" ? 1 : settings.preset === "gentle" ? 1.65 : 1.25;
    point.time = last.time + Math.max(between(r, 0.58, 1.02) * pace, distance / 42);
    if (large && family.axis)
      this.largeDirections[family.axis] = { time: point.time, side: family.side! };
    this.knots.push(point);
    this.onTarget?.({ time: point.time, kind: family.name, large, axis: family.axis, side: family.side });
  }
  private updateBlink(): void {
    const r = this.blinkRandom;
    while (this.realTime >= this.blinkAt) {
      this.blink = { at: this.blinkAt, close: between(r, 0.065, 0.085), hold: between(r, 0.018, 0.035), open: between(r, 0.15, 0.215) };
      const again = !this.doubleBlink && r() < 0.1;
      this.blinkAt += this.blink.close + this.blink.hold + this.blink.open + (again ? between(r, 0.18, 0.26) : between(r, 2.8, 6.9));
      this.doubleBlink = again;
    }
    const b = this.blink, t = this.realTime - b.at;
    this.output.blink = t < b.close ? 1 - ease(t / b.close) : t < b.close + b.hold ? 0 : ease((t - b.close - b.hold) / b.open);
    if (!this.settings.blink)
      this.output.blink = 1;
  }
}
