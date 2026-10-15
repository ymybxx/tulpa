export type CurrentTask = () => boolean;

/**
 * Runs expensive renderer work one item at a time while allowing callers to
 * replace work that has not finished yet. Serial execution keeps official
 * Cubism candidate uploads and the shared WebGL offscreen pool from being
 * mutated by two model loads at the same time.
 */
export class LatestTaskQueue<T> {
  private requested = 0;
  private tail: Promise<void> = Promise.resolve();

  enqueue(
    value: T,
    run: (value: T, isCurrent: CurrentTask) => Promise<void>,
  ): Promise<void> {
    const request = ++this.requested;
    const operation = this.tail
      .catch(() => undefined)
      .then(async () => {
        if (request !== this.requested) return;
        await run(value, () => request === this.requested);
      });
    this.tail = operation.catch(() => undefined);
    return operation;
  }

  invalidate(): void {
    this.requested += 1;
  }

  whenIdle(): Promise<void> {
    return this.tail;
  }
}
