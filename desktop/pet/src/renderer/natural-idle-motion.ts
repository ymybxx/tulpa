import type { NaturalIdleSettings } from "../shared/contracts";
import { normalizeNaturalIdleSettings } from "../shared/natural-idle-settings";

export const NATURAL_IDLE_INPUT_PARAMETERS = ["ParamAngleX", "ParamAngleY", "ParamAngleZ"] as const;

export interface IdleHeadPose {
  x: number;
  y: number;
  z: number;
}

const neutral = (): IdleHeadPose => ({ x: 0, y: 0, z: 0 });

/**
 * Irregular pose changes, with correlated yaw/tilt and small settling pauses.
 * Quintic interpolation has zero endpoint velocity and acceleration, so choosing
 * the next random target never introduces a per-frame noise impulse into physics.
 */
export class NaturalIdleMotion {
  private settings: NaturalIdleSettings = normalizeNaturalIdleSettings(undefined);
  private from = neutral();
  private target = neutral();
  private elapsed = 0;
  private duration = 0;
  private hold = 0;
  private gain = 0;
  private speed = 1;
  private readonly output = neutral();

  constructor(private readonly random: () => number = Math.random) {}

  configure(value: NaturalIdleSettings | undefined): void {
    this.settings = normalizeNaturalIdleSettings(value);
  }

  update(deltaSeconds: number, available: boolean): Readonly<IdleHeadPose> {
    const dt = Number.isFinite(deltaSeconds) ? Math.min(0.1, Math.max(0, deltaSeconds)) : 0;
    const wantedGain = this.settings.enabled && available ? this.settings.intensity : 0;
    this.gain += (wantedGain - this.gain) * -Math.expm1(-dt / 0.24);
    if (wantedGain === 0 && this.gain < 0.0001) {
      this.gain = 0;
      this.output.x = this.output.y = this.output.z = 0;
      return this.output;
    }
    this.speed += (this.settings.speed - this.speed) * -Math.expm1(-dt / 0.4);
    if (this.duration === 0) this.choosePose();
    this.elapsed += dt * this.speed;
    if (this.elapsed >= this.duration + this.hold) {
      this.elapsed -= this.duration + this.hold;
      this.choosePose();
    }
    const t = Math.min(1, this.elapsed / this.duration);
    const eased = t * t * t * (10 + t * (-15 + t * 6));
    this.output.x = (this.from.x + (this.target.x - this.from.x) * eased) * this.gain;
    this.output.y = (this.from.y + (this.target.y - this.from.y) * eased) * this.gain;
    this.output.z = (this.from.z + (this.target.z - this.from.z) * eased) * this.gain;
    return this.output;
  }

  private choosePose(): void {
    this.from = this.target;
    const recenter = this.random() < 0.22;
    const glance = !recenter && this.random() < 0.3;
    const reach = recenter ? 0.25 : glance ? 1.25 : 1;
    // Occasional quicker glances excite low-mobility body rigs without adding jitter.
    const x = glance
      ? (this.random() < 0.5 ? -1 : 1) * (12 + this.random() * 3)
      : this.signed() * 12 * reach;
    this.target = {
      x,
      y: (this.signed() * 4.5 - Math.abs(x) * 0.08) * (recenter ? 0.5 : 1),
      z: Math.max(-11, Math.min(11, -x * 0.45 + this.signed() * 6 * reach)),
    };
    this.duration = glance ? 0.75 + this.random() * 0.4 : 1.15 + this.random() * 1.65;
    this.hold = this.random() < 0.35 ? 0.35 + this.random() * 0.8 : this.random() * 0.16;
  }

  private signed(): number {
    return this.random() * 2 - 1;
  }
}
