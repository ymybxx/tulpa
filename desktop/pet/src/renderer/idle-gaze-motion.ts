const clamp = (x: number, min: number, max: number) => Math.max(min, Math.min(max, x));
const ease = (x: number) => { const t = clamp(x, 0, 1); return t * t * t * (10 + t * (-15 + t * 6)); };
type Phase = "fixation" | "glance" | "return";

/** Short glances return to a fixation; activity changes gaze timing and travel, not eyelid depth.
 * Uses the existing frame clock and only samples new targets at phase boundaries.
 */
export class IdleGazeMotion {
  private activity = 1;
  private time = 0;
  private phase: Phase = "fixation";
  private start = 0;
  private end = 0;
  private move = 0.25;
  private fromX = 0;
  private fromY = 0.035;
  private x = 0;
  private y = 0.035;
  private anchorX = 0;
  private anchorY = 0.035;
  private lastGlance = -10;
  private lastGlanceDirection = -1;
  private readonly glanceDirections = [0, 1, 2, 3];
  private remainingDirections = 0;
  private cueDuration = 0;
  private readonly output = { aimX: 0, aimY: 0.035, open: 1, brow: 0 };

  constructor(private readonly random: () => number) {}

  configure(value: number): void {
    const next = Number.isFinite(value) ? clamp(value, 0, 2) : 1;
    if (next === this.activity) return;
    if (next === 0 || this.activity === 0) {
      // Live changes settle from the displayed position, never jump to a target.
      this.fromX = this.output.aimX; this.fromY = this.output.aimY;
      this.x = 0; this.y = 0.035;
      this.start = this.time; this.move = 0.3;
      this.end = next === 0 ? Infinity : this.time + this.move;
      this.phase = "fixation"; this.cueDuration = 0;
    } else if (this.phase !== "glance") {
      const holdStart = Math.max(this.time, this.start + this.move);
      const remaining = Math.max(0, this.end - holdStart);
      if (remaining > 0) this.end = holdStart + remaining * this.meanHold(next) / this.meanHold(this.activity);
    }
    this.activity = next;
  }

  update(time: number): Readonly<typeof this.output> {
    this.time = time;
    while (time >= this.end) this.next(this.end);
    const elapsed = time - this.start, t = ease(elapsed / this.move);
    this.output.aimX = this.fromX + (this.x - this.fromX) * t;
    this.output.aimY = this.fromY + (this.y - this.fromY) * t;
    const c = this.cueDuration > 0 ? clamp(elapsed / this.cueDuration, 0, 1) : 0;
    const cue = 16 * c * c * (1 - c) * (1 - c);
    // A brief, tiny alert cue. Ordinary openness is always the model's baseline.
    this.output.open = 1 + 0.012 * cue;
    this.output.brow = 0.02 * cue;
    return this.output;
  }

  private between(min: number, max: number): number { return min + (max - min) * this.random(); }
  private meanHold(activity: number): number { return activity <= 1 ? 6 - 3 * activity : 4.25 - 1.25 * activity; }
  private hold(): number {
    const a = this.activity;
    return a <= 1 ? this.between(4 - 2 * a, 8 - 4 * a) : this.between(3 - a, 5.5 - 1.5 * a);
  }
  private nextGlanceDirection(): number {
    // Draw without replacement so a random run cannot omit up/down indefinitely.
    if (this.remainingDirections === 0) {
      for (let i = 0; i < 4; i++) this.glanceDirections[i] = i;
      this.remainingDirections = 4;
    }
    let index = Math.floor(this.random() * this.remainingDirections);
    if (this.remainingDirections === 4 && this.glanceDirections[index] === this.lastGlanceDirection)
      index = (index + 1 + Math.floor(this.random() * 3)) % 4;
    const direction = this.glanceDirections[index]!;
    this.glanceDirections[index] = this.glanceDirections[--this.remainingDirections]!;
    this.lastGlanceDirection = direction;
    return direction;
  }
  private next(start: number): void {
    this.fromX = this.x; this.fromY = this.y;
    let hold: number;
    if (this.phase === "glance") {
      this.phase = "return";
      this.x = this.anchorX; this.y = this.anchorY;
      this.move = this.between(0.26, 0.42);
      hold = this.hold();
    } else if (start - this.lastGlance >= 2.5 && this.random() < 0.06 + 0.12 * this.activity) {
      this.phase = "glance";
      this.anchorX = this.x; this.anchorY = this.y;
      this.lastGlance = start;
      const direction = this.nextGlanceDirection();
      if (direction < 2) {
        this.x = (direction === 0 ? -1 : 1) * this.between(0.24 + 0.2 * this.activity, 0.4 + 0.25 * this.activity);
        this.y = this.between(-0.1, 0.16);
      } else {
        this.x = this.between(-0.2, 0.2);
        this.y = direction === 2 ? -this.between(0.22 + 0.12 * this.activity, 0.34 + 0.2 * this.activity) :
          this.between(0.35 + 0.18 * this.activity, 0.5 + 0.24 * this.activity);
      }
      this.move = this.between(0.24, 0.38);
      hold = this.between(0.4 + 0.15 * this.activity, 0.7 + 0.3 * this.activity);
    } else {
      this.phase = "fixation";
      const radius = 0.07 + 0.115 * this.activity;
      // Keep small shifts distinct from glances, but visible at normal pet sizes.
      this.x = this.between(-radius, radius);
      this.y = 0.035 + this.between(-radius * 0.65, radius * 0.65);
      this.move = this.between(0.22, 0.38);
      hold = this.hold();
    }
    this.start = start;
    this.end = start + this.move + hold;
    this.cueDuration = this.phase === "glance" ? Math.min(0.65, this.move + hold) : 0;
  }
}
