/**
 * C2 quintic trajectory replacement: splice from the current position, velocity
 * and acceleration, then arrive at rest. Limits are fractions of a part's range.
 * References (algorithm only; no external code/runtime dependency):
 * https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html
 * https://github.com/pantor/ruckig (bounded velocity, acceleration and jerk).
 * This is a bounded animation spline, not Ruckig's time-optimal solver.
 */
export interface TransitionProfile {
  minimumDuration: number;
  velocity: number;
  acceleration: number;
  jerk: number;
}
export interface MotionState { position: number; velocity: number; acceleration: number }

const FACE = { minimumDuration: 0.14, velocity: 4, acceleration: 40, jerk: 700 };
// A person turns their head unhurriedly while talking: about 0.8 s for a
// quarter of its reach (15° of ±30°), about 1.1 s for half. The body is heavier still.
const PROFILES: Readonly<Record<string, TransitionProfile>> = {
  head: { minimumDuration: 0.4, velocity: 0.9, acceleration: 2.3, jerk: 30 },
  body: { minimumDuration: 0.5, velocity: 0.8, acceleration: 1.8, jerk: 22 },
  overall: { minimumDuration: 0.5, velocity: 0.8, acceleration: 1.8, jerk: 22 },
  arms: { minimumDuration: 0.28, velocity: 1.5, acceleration: 7, jerk: 90 },
  hands: { minimumDuration: 0.20, velocity: 2.5, acceleration: 20, jerk: 300 },
  eyes: { minimumDuration: 0.10, velocity: 6, acceleration: 70, jerk: 1800 },
  gaze: { minimumDuration: 0.08, velocity: 9, acceleration: 120, jerk: 3000 },
  brows: FACE,
  mouth: FACE,
  cheek: { minimumDuration: 0.35, velocity: 1.5, acceleration: 10, jerk: 90 },
};
export const transitionProfile = (group: string): TransitionProfile => PROFILES[group] ?? FACE;
const clamp = (x: number, low: number, high: number) => Math.max(low, Math.min(high, x));
const evaluate = (coefficients: readonly number[], x: number): number => coefficients.reduceRight((sum, value) => sum * x + value, 0);
const derivative = (coefficients: readonly number[]): number[] => coefficients.slice(1).map((value, i) => value * (i + 1));

/** All real roots in (0,1), isolated by derivative extrema; degrees are at most 4. */
function roots(coefficients: readonly number[]): number[] {
  const c = [...coefficients];
  while (c.length > 1 && Math.abs(c.at(-1)!) < 1e-12) c.pop();
  if (c.length < 2) return [];
  if (c.length === 2) { const x = -c[0]! / c[1]!; return x > 0 && x < 1 ? [x] : []; }
  const knots = [0, ...roots(derivative(c)), 1], found: number[] = [];
  for (let i = 0; i < knots.length - 1; i++) {
    let a = knots[i]!, b = knots[i + 1]!, fa = evaluate(c, a);
    if (a > 0 && Math.abs(fa) < 1e-10) found.push(a);
    if (fa * evaluate(c, b) >= 0) continue;
    for (let j = 0; j < 36; j++) {
      const middle = (a + b) / 2, fm = evaluate(c, middle);
      if (fa * fm <= 0) b = middle;
      else { a = middle; fa = fm; }
    }
    found.push((a + b) / 2);
  }
  return found;
}

interface Trajectory { coefficients: number[]; duration: number; elapsed: number; target: number }
function spline(state: MotionState, target: number, duration: number): Trajectory {
  const d = target - state.position, v = state.velocity * duration, a = state.acceleration * duration * duration;
  return { duration, elapsed: 0, target, coefficients: [state.position, v, a / 2,
    10 * d - 6 * v - 1.5 * a, -15 * d + 8 * v + 1.5 * a, 6 * d - 3 * v - 0.5 * a] };
}
function bounded(trajectory: Trajectory, limits: TransitionProfile): boolean {
  let c = trajectory.coefficients;
  const ranges = [[0, 1], [-limits.velocity, limits.velocity], [-limits.acceleration, limits.acceleration], [-limits.jerk, limits.jerk]];
  for (const [min, max] of ranges) {
    const d = derivative(c);
    for (const x of [0, ...roots(d), 1]) {
      const value = evaluate(c, x);
      if (value < min! - 1e-8 || value > max! + 1e-8) return false;
    }
    c = d.map(value => value / trajectory.duration);
  }
  return true;
}

function plan(state: MotionState, target: number, limits: TransitionProfile): Trajectory | null {
  const distance = Math.abs(target - state.position);
  // Exact rest-to-rest quintic peak derivatives provide the first time estimate.
  let duration = Math.max(limits.minimumDuration, 1.875 * distance / limits.velocity,
    Math.sqrt(5.774 * distance / limits.acceleration), Math.cbrt(60 * distance / limits.jerk));
  for (let attempt = 0; attempt < 24; attempt++, duration *= 1.12) {
    const candidate = spline(state, target, duration);
    if (bounded(candidate, limits)) return candidate;
  }
  return null;
}

/** Native ranges may be reversed in a manual binding; only the span sets limits. */
export class FacialTransition {
  private state: MotionState;
  private desired: number;
  private trajectory: Trajectory | null = null;
  private readonly span: number;
  private readonly limits: TransitionProfile;

  constructor(value: number, private readonly minimum: number, private readonly maximum: number,
    profile: TransitionProfile, workingSpan = maximum - minimum) {
    this.span = maximum - minimum;
    this.desired = this.span > 0 ? clamp((value - minimum) / this.span, 0, 1) : 0;
    this.state = { position: this.desired, velocity: 0, acceleration: 0 };
    const ratio = this.span > 0 && Number.isFinite(workingSpan) && workingSpan > 0 ? Math.min(1, workingSpan / this.span) : 1;
    this.limits = { minimumDuration: profile.minimumDuration, velocity: profile.velocity * ratio,
      acceleration: profile.acceleration * ratio, jerk: profile.jerk * ratio };
  }

  get value(): number { return this.minimum + this.state.position * this.span; }
  snapshot(): MotionState {
    return { position: this.value, velocity: this.state.velocity * this.span, acceleration: this.state.acceleration * this.span };
  }
  setTarget(value: number): void {
    if (!Number.isFinite(value) || this.span <= 0) return;
    const next = clamp((value - this.minimum) / this.span, 0, 1);
    if (Math.abs(next - this.desired) < 1e-12) return; // Repeated sentences must not restart a move.
    this.desired = next;
    const replacement = plan(this.state, next, this.limits);
    // Near a model limit, reversing instantly may have no safe quintic solution.
    // Keep the already validated braking path, then use only the latest target.
    if (replacement) this.trajectory = replacement;
  }
  update(delta: number): number {
    if (!Number.isFinite(delta) || delta <= 0) return this.value;
    let remaining = delta;
    while (remaining > 0) {
      if (!this.trajectory) {
        if (Math.abs(this.state.position - this.desired) < 1e-12) break;
        this.trajectory = plan(this.state, this.desired, this.limits);
        if (!this.trajectory) break;
      }
      const t = this.trajectory, step = Math.min(remaining, t.duration - t.elapsed);
      t.elapsed += step; remaining -= step;
      if (t.elapsed >= t.duration) {
        this.state = { position: t.target, velocity: 0, acceleration: 0 };
        this.trajectory = null;
      } else {
        const u = t.elapsed / t.duration, velocity = derivative(t.coefficients), acceleration = derivative(velocity);
        this.state = { position: clamp(evaluate(t.coefficients, u), 0, 1),
          velocity: evaluate(velocity, u) / t.duration, acceleration: evaluate(acceleration, u) / (t.duration * t.duration) };
      }
    }
    return this.value;
  }
}
