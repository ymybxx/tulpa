import { describe, expect, it } from "vitest";
import { layoutPetChat, type Rect } from "./pet-chat";
import { PetChatProgress } from "./pet-chat-progress";

describe("pet chat screen placement", () => {
  for (const work of [
    { x: 0, y: 25, width: 1440, height: 815 },
    { x: -1920, y: -400, width: 1920, height: 1040 },
    { x: 1440, y: 0, width: 600, height: 900 },
  ]) {
    for (const position of ["top-left", "top-right", "bottom-left", "bottom-right", "middle"]) {
      it(`keeps the cards, the capsule and the chat entry inside ${JSON.stringify(work)} at ${position}`, () => {
        const pet = { x: work.x + (position.includes("left") ? -40 : position.includes("right") ? work.width - 130 : work.width / 2),
          y: work.y + (position.includes("top") ? -70 : position.includes("bottom") ? work.height - 220 : work.height / 3), width: 210, height: 380 };
        const layout = layoutPetChat(pet, work);
        for (const r of [layout.entry, layout.home, layout.composer, layout.bubble, layout.capsule, layout.panel]) {
          const global = { ...r, x: r.x + layout.window.x, y: r.y + layout.window.y };
          expect(global.x).toBeGreaterThanOrEqual(work.x);
          expect(global.y).toBeGreaterThanOrEqual(work.y);
          expect(global.x + global.width).toBeLessThanOrEqual(work.x + work.width);
          expect(global.y + global.height).toBeLessThanOrEqual(work.y + work.height);
        }
        expect(contains(layout.composer, layout.entry)).toBe(true);
        // The main window button sits in the same column, right next to the chat entry.
        expect(layout.home.x).toBe(layout.entry.x);
        expect(Math.abs(layout.home.y - layout.entry.y)).toBeCloseTo(layout.entry.height + 8);
        expect(overlap(layout.composer, layout.bubble)).toBe(false);
      });
    }
  }
  it("puts the sense capsule over the head and opens the panel away from the screen edge", () => {
    const work = { x: 0, y: 24, width: 1440, height: 800 };
    const low = layoutPetChat({ x: 900, y: 460, width: 200, height: 380 }, work);
    const capsule = { ...low.capsule, x: low.capsule.x + low.window.x, y: low.capsule.y + low.window.y };
    expect(capsule.x + capsule.width / 2).toBe(1000);
    expect(capsule.y + capsule.height).toBeLessThanOrEqual(460);
    expect(low.panelPlacement).toBe("above");
    expect(overlap(low.panel, low.capsule)).toBe(false);
    const high = layoutPetChat({ x: 900, y: 40, width: 200, height: 380 }, work);
    expect(high.panelPlacement).toBe("below");
    expect(high.panel.y).toBeGreaterThan(high.capsule.y);
  });
  it("puts the entry beside the middle of the body, where the input then opens", () => {
    const work = { x: 0, y: 24, width: 1440, height: 800 };
    // A wide hat and hair make the whole figure 360 wide; halfway down the body is only 120 wide.
    const pet = { x: 800, y: 200, width: 360, height: 560, middle: { y: 480, left: 920, right: 1040 } };
    const l = layoutPetChat(pet, work);
    const entry = { ...l.entry, x: l.entry.x + l.window.x, y: l.entry.y + l.window.y };
    expect(l.side).toBe("left");
    expect(entry.x + entry.width).toBe(910);
    expect(entry.y + entry.height / 2).toBe(480);
    expect(l.bubble.y + l.bubble.height).toBeLessThan(l.composer.y);
  });
  it("keeps cards outside the character when there is side space", () => {
    const pet = { x: 900, y: 300, width: 200, height: 400 };
    const l = layoutPetChat(pet, { x: 0, y: 24, width: 1440, height: 800 });
    expect(l.side).toBe("left");
    expect(l.window.x + l.composer.x + l.composer.width).toBeLessThan(pet.x);
  });
});
function contains(outer: Rect, inner: Rect): boolean { return inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width && inner.y + inner.height <= outer.y + outer.height; }
function overlap(a: Rect, b: Rect): boolean { return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height; }

function event(sequence: number, revision: number, content: string, status = "playing", playbackId = "one") {
  return { type: "pet.playback.progress", sequence, timestampMs: Date.now(), payload: { fishId: "fish", messageId: "message", playbackId, revision, visibleContent: content, displayOffset: Array.from(content).length, consumedOffset: Array.from(content).length, status } };
}
describe("speech bubble playback authority", () => {
  it("accepts Unicode playback prefixes, rejects stale revisions and rolls back on interruption", () => {
    const p = new PetChatProgress();
    expect(p.accept(event(1, 1, "你好🐟"))?.visibleContent).toBe("你好🐟");
    expect(p.accept(event(2, 1, "这是迟到的内容"))).toBeNull();
    expect(p.accept(event(3, 2, "你好", "interrupted"))?.visibleContent).toBe("你好");
    expect(p.accept(event(2, 3, "不该出现"))).toBeNull();
  });
  it("does not replay terminal snapshots or foreign fish and resets across reconnects", () => {
    const p = new PetChatProgress();
    expect(p.accept({ type: "pet.snapshot", payload: { events: [event(80, 2, "旧回答", "completed")] } })).toBeNull();
    expect(p.accept(event(81, 3, "另一只宠物"), "different")).toBeNull();
    expect(p.accept(event(82, 4, "新的"))).not.toBeNull();
    p.reset();
    expect(p.accept(event(1, 1, "重新连接"))).not.toBeNull();
  });
  it("rejects generated content masquerading as a shorter playback prefix", () => {
    const p = new PetChatProgress();
    const invalid = event(1, 1, "未播放的文字"); invalid.payload.displayOffset = 1;
    expect(p.accept(invalid)).toBeNull();
  });
});

describe("the open sense panel and the chat column", () => {
  const work = { x: 0, y: 25, width: 1512, height: 957 };
  // The head near the top edge: the panel has to open downward, over the body.
  const pet = { x: 1030, y: 60, width: 420, height: 700, middle: { y: 420, left: 1150, right: 1330 } };
  const global = (l: ReturnType<typeof layoutPetChat>, r: Rect): Rect => ({ ...r, x: r.x + l.window.x, y: r.y + l.window.y });

  it("never leaves the bubble or the pills under the panel", () => {
    const closed = layoutPetChat(pet, work);
    expect(closed.panelPlacement).toBe("below");
    expect([closed.bubble, closed.entry, closed.home].some((r) => overlap(r, closed.panel))).toBe(true);
    const open = layoutPetChat(pet, work, null, true);
    expect(open.chatAside).toBe(false);
    for (const r of [open.bubble, open.entry, open.home]) expect(overlap(r, open.panel)).toBe(false);
    expect(open.home.x).toBe(open.entry.x);
    expect(open.bubble.y + open.bubble.height).toBeLessThan(open.composer.y);
    for (const r of [open.bubble, open.entry, open.home, open.panel]) {
      const g = global(open, r);
      expect(g.y + g.height).toBeLessThanOrEqual(work.y + work.height);
    }
  });

  it("steps the column aside only when no place is free", () => {
    const tall = { x: 1030, y: 60, width: 420, height: 880, middle: { y: 560, left: 1150, right: 1330 } };
    const open = layoutPetChat(tall, { x: 1000, y: 25, width: 470, height: 930 }, null, true);
    if (!open.chatAside) for (const r of [open.bubble, open.entry, open.home]) expect(overlap(r, open.panel)).toBe(false);
    expect(layoutPetChat(tall, work).chatAside).toBe(false);
  });

  it("shrinks the pills and capsule beside a small TA, but keeps them readable", () => {
    const work = { x: 0, y: 0, width: 1512, height: 982 };
    const big = layoutPetChat({ x: 700, y: 200, width: 200, height: 420 }, work);
    expect(big.hud).toBe(1);
    expect(big.entry.width).toBe(92);
    expect(big.capsule.height).toBe(36);
    const mid = layoutPetChat({ x: 700, y: 500, width: 140, height: 300 }, work);
    expect(mid.hud).toBeCloseTo(300 / 360);
    expect(mid.entry.height).toBeCloseTo(36 * 300 / 360);
    const tiny = layoutPetChat({ x: 1400, y: 820, width: 60, height: 120 }, work);
    expect(tiny.hud).toBe(0.75);
    expect(tiny.entry.width).toBeCloseTo(69);
    expect(tiny.capsule.width).toBeCloseTo(103.5);
    // Still beside the body and over the head, and still inside the screen.
    expect(tiny.capsule.y + tiny.window.y + tiny.capsule.height).toBeLessThanOrEqual(820);
    expect(tiny.window.x + tiny.window.width).toBeLessThanOrEqual(1512);
  });
});
