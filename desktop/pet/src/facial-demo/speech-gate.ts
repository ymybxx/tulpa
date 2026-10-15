export interface GatedSpeechChunk<T> {
  segment: number;
  chunk: T;
}

/** Holds ordered speech chunks until their own sentence's face is ready. */
export class SpeechGate<T> {
  private readonly queue: Array<GatedSpeechChunk<T>> = [];
  private readonly readySegments = new Set<number>();

  push(segment: number, chunk: T): void {
    this.queue.push({ segment, chunk });
  }

  ready(segment: number): void {
    this.readySegments.add(segment);
  }

  drain(): Array<GatedSpeechChunk<T>> {
    let count = 0;
    while (count < this.queue.length && this.readySegments.has(this.queue[count]!.segment)) count++;
    return this.queue.splice(0, count);
  }

  /** Includes ready sentences still blocked behind an earlier queued sentence. */
  pendingSegments(): number[] {
    return [...new Set(this.queue.map(value => value.segment))];
  }

  hasPending(): boolean {
    return this.queue.length > 0;
  }

  clear(): void {
    this.queue.length = 0;
    this.readySegments.clear();
  }
}
