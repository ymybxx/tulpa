export interface ChatProgress {
  fishId: string;
  messageId: string;
  playbackId: string;
  revision: number;
  status: string;
  visibleContent: string;
  displayOffset: number;
  consumedOffset: number;
  /** Which outside service let this reply down ("chat:timeout"), if one did. */
  failure?: string;
}
export const terminalProgress = (status: string): boolean => ["completed", "interrupted", "failed"].includes(status);

/** A display-only observer. Never estimates or writes playback consumption. */
export class PetChatProgress {
  private sequence = -1;
  private current: ChatProgress | null = null;
  reset(): void { this.sequence = -1; this.current = null; }

  accept(value: unknown, fishId?: string): ChatProgress | null {
    if (!value || typeof value !== "object") return null;
    const event = value as { type?: string; sequence?: number; timestampMs?: number; payload?: unknown };
    if (event.type === "pet.snapshot") {
      const events = (event.payload as { events?: unknown[] })?.events;
      if (!Array.isArray(events)) return null;
      let latest: ChatProgress | null = null;
      for (const item of [...events].sort((a, b) => Number((a as { sequence: number }).sequence) - Number((b as { sequence: number }).sequence))) {
        const e = item as { payload?: ChatProgress; timestampMs?: number };
        // A reconnect must not replay old spoken text on the desktop.
        if (terminalProgress(e.payload?.status ?? "") || Date.now() - (e.timestampMs ?? 0) > 15_000) continue;
        latest = this.accept(item, fishId) ?? latest;
      }
      return latest;
    }
    if (event.type !== "pet.playback.progress" || !Number.isSafeInteger(event.sequence) || event.sequence! <= this.sequence) return null;
    const p = event.payload as ChatProgress;
    if (!p || (fishId && p.fishId !== fishId) || ![p.fishId, p.messageId, p.playbackId].every(v => typeof v === "string" && v.length > 0)) return null;
    if (![p.revision, p.displayOffset, p.consumedOffset].every(v => Number.isSafeInteger(v) && v >= 0) || p.consumedOffset > p.displayOffset) return null;
    if (typeof p.visibleContent !== "string" || Array.from(p.visibleContent).length !== p.displayOffset) return null;
    if (!["buffering", "playing", "paused", "completed", "interrupted", "failed"].includes(p.status)) return null;
    if (this.current?.playbackId === p.playbackId && this.current.revision >= p.revision) return null;
    this.sequence = event.sequence!;
    this.current = p;
    return p;
  }
}
