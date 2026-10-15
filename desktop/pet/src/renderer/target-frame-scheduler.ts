/**
 * The timer wakes this long before a frame is due and leaves the rest to rAF.
 * It must cover a display frame (8.3 ms at 120 Hz) plus the timer's own
 * lateness, or a late timer misses the due refresh and that frame lasts one
 * refresh longer: at 30 fps on 120 Hz, 42 ms instead of 33, a visible hitch.
 */
const DEFAULT_ANIMATION_FRAME_LEAD_MS = 14;
const FRAME_DUE_TOLERANCE_MS = 2;

export interface TargetFrameSchedulerClock {
  now(): number;
  requestAnimationFrame(callback: (timestamp: number) => void): number;
  cancelAnimationFrame(handle: number): void;
  setTimeout(callback: () => void, delayMs: number): number;
  clearTimeout(handle: number): void;
}

const browserClock: TargetFrameSchedulerClock = {
  now: () => performance.now(),
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  cancelAnimationFrame: (handle) => cancelAnimationFrame(handle),
  setTimeout: (callback, delayMs) => window.setTimeout(callback, delayMs),
  clearTimeout: (handle) => window.clearTimeout(handle),
};

/**
 * Requests one browser animation frame near each target-frame deadline.
 *
 * A permanently recursive requestAnimationFrame loop wakes at the display's
 * full refresh rate even when most callbacks are discarded. This scheduler
 * sleeps between target frames and only uses rAF to align the actual draw with
 * the compositor.
 */
export class TargetFrameScheduler {
  private frameRate: number;
  private running = false;
  private timer: number | null = null;
  private animationFrame: number | null = null;
  private lastFrameAt: number | null = null;

  constructor(
    frameRate: number,
    private readonly onFrame: (timestamp: number) => void,
    private readonly clock: TargetFrameSchedulerClock = browserClock,
    private readonly animationFrameLeadMs = DEFAULT_ANIMATION_FRAME_LEAD_MS,
  ) {
    assertFrameRate(frameRate);
    this.frameRate = frameRate;
  }

  resume(): void {
    if (this.running) return;
    this.running = true;
    this.lastFrameAt = null;
    this.requestFrame();
  }

  pause(): void {
    if (!this.running && this.timer === null && this.animationFrame === null) return;
    this.running = false;
    this.lastFrameAt = null;
    this.cancelPending();
  }

  setFrameRate(value: number): void {
    assertFrameRate(value);
    if (value === this.frameRate) return;
    this.frameRate = value;
    if (!this.running) return;
    this.cancelPending();
    this.scheduleNext();
  }

  destroy(): void {
    this.pause();
  }

  private readonly handleAnimationFrame = (timestamp: number): void => {
    this.animationFrame = null;
    if (!this.running) return;
    const intervalMs = 1000 / this.frameRate;
    if (
      this.lastFrameAt !== null
      && timestamp - this.lastFrameAt < intervalMs - FRAME_DUE_TOLERANCE_MS
    ) {
      // Close to due, wait on the display itself: a timer could wake too late.
      if (intervalMs - (timestamp - this.lastFrameAt) <= this.animationFrameLeadMs) this.requestFrame();
      else this.scheduleNext();
      return;
    }
    this.lastFrameAt = timestamp;
    this.onFrame(timestamp);
    if (this.running) this.scheduleNext();
  };

  private scheduleNext(): void {
    if (!this.running || this.timer !== null || this.animationFrame !== null) return;
    if (this.lastFrameAt === null) {
      this.requestFrame();
      return;
    }
    const elapsedMs = Math.max(0, this.clock.now() - this.lastFrameAt);
    const remainingMs = Math.max(0, 1000 / this.frameRate - elapsedMs);
    const delayMs = Math.max(0, remainingMs - this.animationFrameLeadMs);
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      if (this.running) this.requestFrame();
    }, delayMs);
  }

  private requestFrame(): void {
    if (!this.running || this.animationFrame !== null) return;
    this.animationFrame = this.clock.requestAnimationFrame(this.handleAnimationFrame);
  }

  private cancelPending(): void {
    if (this.timer !== null) {
      this.clock.clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.animationFrame !== null) {
      this.clock.cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }
}

function assertFrameRate(value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid target frame rate: ${value}`);
  }
}
