import type { PetState } from "../shared/contracts";
import { CLOSED_MOUTH, type LipShape } from "../renderer/pet-renderer";

interface PreviewValidationTarget {
  setState(value: PetState): void;
  setGaze(x: number, y: number): void;
  setLipShape(value: LipShape): void;
  playFirstMotion(): void;
  reset(): void;
}

const VOWELS: LipShape[] = [
  { volume: 0.9, a: 1, e: 0, i: 0, o: 0, u: 0 },
  { volume: 0.7, a: 0, e: 1, i: 0, o: 0, u: 0 },
  { volume: 0.55, a: 0, e: 0, i: 1, o: 0, u: 0 },
  { volume: 0.85, a: 0, e: 0, i: 0, o: 1, u: 0 },
  { volume: 0.6, a: 0, e: 0, i: 0, o: 0, u: 1 },
];

export class PreviewValidationSequence {
  private timers: ReturnType<typeof setTimeout>[] = [];

  constructor(private readonly target: PreviewValidationTarget) {}

  start(): void {
    this.cancel();
    this.target.setState("standby");
    this.target.setGaze(-0.35, 0.08);
    this.schedule(450, () => this.target.setGaze(0.35, -0.04));
    this.schedule(850, () => this.target.setState("speaking"));
    VOWELS.forEach((shape, index) => {
      this.schedule(950 + index * 380, () => this.target.setLipShape(shape));
    });
    this.schedule(2_950, () => {
      this.target.setLipShape(CLOSED_MOUTH);
      this.target.setState("standby");
      this.target.playFirstMotion();
    });
    this.schedule(4_200, () => this.target.reset());
  }

  cancel(): void {
    for (const timer of this.timers) clearTimeout(timer);
    this.timers = [];
  }

  private schedule(delayMs: number, callback: () => void): void {
    this.timers.push(setTimeout(callback, delayMs));
  }
}
