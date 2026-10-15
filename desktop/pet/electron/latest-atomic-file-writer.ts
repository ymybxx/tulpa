import { randomUUID } from "node:crypto";
import { rename, writeFile } from "node:fs/promises";

interface PendingWrite {
  value: string;
  waiters: Array<{ resolve: () => void; reject: (error: unknown) => void }>;
}

/** Serializes atomic replacements while coalescing bursts to their newest value. */
export class LatestAtomicFileWriter {
  private pending: PendingWrite | null = null;
  private draining = false;
  private readonly flushWaiters: Array<() => void> = [];

  constructor(private readonly target: string) {}

  write(value: string): Promise<void> {
    const result = new Promise<void>((resolve, reject) => {
      if (this.pending) {
        this.pending.value = value;
        this.pending.waiters.push({ resolve, reject });
      } else {
        this.pending = { value, waiters: [{ resolve, reject }] };
      }
    });
    if (!this.draining) void this.drain();
    return result;
  }

  async flush(): Promise<void> {
    if (!this.draining && !this.pending) return;
    await new Promise<void>((resolve) => this.flushWaiters.push(resolve));
  }

  private async drain(): Promise<void> {
    this.draining = true;
    while (this.pending) {
      const current = this.pending;
      this.pending = null;
      const temporary = `${this.target}.${randomUUID()}.tmp`;
      try {
        await writeFile(temporary, current.value, "utf8");
        await rename(temporary, this.target);
        for (const waiter of current.waiters) waiter.resolve();
      } catch (error) {
        for (const waiter of current.waiters) waiter.reject(error);
      }
    }
    this.draining = false;
    for (const resolve of this.flushWaiters.splice(0)) resolve();
  }
}
