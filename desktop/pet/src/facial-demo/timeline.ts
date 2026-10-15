import type { Face } from "../shared/facial-demo";

/** Results are staged for their own spoken segment. A late old result never rewinds the face. */
export class FaceTimeline {
  active = -1;
  completed = -1;
  readonly results = new Map<number, Face>();
  expired = 0;
  receive(segment: number, face: Face): Face | null {
    if (segment <= this.completed || segment < this.active) { this.expired++; return null; }
    this.results.set(segment, face);
    return segment === this.active ? face : null;
  }
  enter(segment: number): Face | null {
    if (segment <= this.completed || segment < this.active) return null;
    this.active = segment;
    for (const old of this.results.keys()) if (old < segment) this.results.delete(old);
    return this.results.get(segment) ?? null;
  }
  finish(segment: number): void {
    this.completed = Math.max(this.completed, segment);
    if (this.active <= this.completed) this.active = -1;
    this.results.delete(segment);
  }
}
