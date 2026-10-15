import { BrowserWindow, ipcMain, screen, type IpcMainEvent } from "electron";
import { chatAttachmentUpload } from "./chat-attachment-upload";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { contains, layoutPetChat, validRect, type PetBody, type PetChatGeometry, type PetChatLayout, type PetChatState, type Rect } from "../src/shared/pet-chat";
import { PetChatProgress, terminalProgress } from "../src/shared/pet-chat-progress";
import { replyFailureNote, serviceFailure, type ServiceFailure } from "../src/shared/service-failures";
import { LatestAtomicFileWriter } from "./latest-atomic-file-writer";
import { PetPanelFocus } from "./pet-panel-focus";
import type { PetStage } from "./pet-stage";
import { loadPetApplicationActivation, type PetApplicationActivation, type PetMouseMonitor } from "./pet-application-activation";
import type { PetChatService } from "./pet-chat-service";
import { parseSenseAction, type SenseAction, type SensePanel, type SensesSnapshot } from "../src/shared/pet-senses";
import { parseQuickAction, parseQuickTiles, placeQuickPanel, type QuickAction, type QuickLook, type QuickTileKey } from "../src/shared/pet-quick";
import type { DesktopUiLocale } from "../src/shared/contracts";

/** What the quick panel needs from the rest of the app each time it opens. */
export interface QuickInfo { speaker: boolean; character: string | null; petState: string | null; looks: QuickLook[]; lookId: string | null }
/** Quick panel actions the app carries out; the HUD handles its own (bubbles, composer, tile layout). */
export type QuickAppAction =
  | { type: "tile"; key: "speaker" | "passthrough" | "chat" | "settings" }
  | Extract<QuickAction, { type: "look" | "hide" | "main" | "quit" }>;

interface Options {
  /** TA's panel: its bounds on screen, and move / resize / show / hide. */
  pet: Pick<PetStage, "getBounds" | "isVisible" | "isDestroyed" | "on" | "removeListener">;
  preferencePath: string;
  load(window: BrowserWindow, file: string): Promise<void>;
  openHistory(): void;
  service: PetChatService;
  menuChanged(): void;
  locale?(): DesktopUiLocale;
  nativePointerActive?(): boolean;
  /** Capsule actions other than opening/closing the panel. */
  senseAction?(action: Exclude<SenseAction, { type: "panel" }>): void;
  /** The capsule just appeared: a good moment to re-read permissions. */
  sensesShown?(): void;
  quickInfo?(): Promise<QuickInfo>;
  quickAction?(action: QuickAppAction): Promise<void>;
  /** 「新对话」 pressed here; main tells every window once it has started. */
  newConversation(): Promise<{ ok: boolean; error?: string }>;
}

/** One small, display-only HUD. It never resizes the Live2D canvas or handles PCM. */
export class PetChatWindow {
  readonly window: BrowserWindow;
  private readonly writer: LatestAtomicFileWriter;
  private readonly progress = new PetChatProgress();
  private readonly focus: PetPanelFocus | null;
  private readonly native: (PetApplicationActivation & PetMouseMonitor) | null;
  private watchingPresses = false;
  private geometry: PetBody | null = null;
  private layout: PetChatLayout | null = null;
  private petHovered = false;
  private hudHovered = false;
  private entryVisible = false;
  private enabled = true;
  private locked = false;
  private editing = false;
  /** The input method's candidate window belongs to another app: its clicks are not clicks away. */
  private imeComposing = false;
  private loaded = false;
  private text = "";
  private layoutPanelOpen = false;
  /** Why the reply did not reach the user, under its text; shown even with none. */
  private note = "";
  private noteContext: { failure: ServiceFailure | null; character: string; said: boolean } | null = null;
  private noteVersion = 0;
  private textStatus = "";
  private textOnly = false;
  private ignoreMouse = true;
  private senses: SensesSnapshot | null = null;
  private sensePanel: SensePanel = "closed";
  private senseNote = "";
  private readiness: PetChatState["readiness"] = null;
  private quickTiles: QuickTileKey[] = parseQuickTiles(undefined);
  private quick: { rect: Rect; edge: "top" | "bottom"; editing: boolean; info: QuickInfo } | null = null;
  private lastState = "";
  private capsulePointer = "";
  private hoverTimer: NodeJS.Timeout | null = null;
  private bubbleTimer: NodeJS.Timeout | null = null;
  /** How to wake TA, waiting for the bubble to clear; and the one now showing. */
  private pendingSleepHint = "";
  private shownSleepHint = "";
  private pointerTimer: NodeJS.Timeout | null = null;
  private readonly listeners: Array<[string, (...args: any[]) => void]> = [];

  constructor(private readonly options: Options) {
    this.writer = new LatestAtomicFileWriter(options.preferencePath);
    this.window = new BrowserWindow({
      width: 340, height: 320, show: false, transparent: true, frame: false,
      focusable: process.platform !== "win32",
      type: process.platform === "darwin" ? "panel" : undefined,
      acceptFirstMouse: true, hasShadow: false, alwaysOnTop: true, skipTaskbar: true,
      resizable: false, minimizable: false, maximizable: false, fullscreenable: false,
      backgroundColor: "#00000000",
      webPreferences: { preload: path.join(__dirname, "pet-chat-preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: false, backgroundThrottling: true },
    });
    // One step above the pet's own level (floating + 1), so the pet never
    // covers the cards, even after a click brings the pet window forward.
    this.window.setAlwaysOnTop(true, "floating", 2);
    this.window.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
    this.window.setIgnoreMouseEvents(true, { forward: true });
    this.window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    this.window.webContents.on("will-navigate", event => event.preventDefault());
    this.native = process.platform === "darwin" || process.platform === "win32" ? loadPetApplicationActivation() : null;
    this.focus = process.platform === "darwin" && this.native ? new PetPanelFocus(this.window, this.native) : null;
    this.window.on("blur", () => {
      this.focus?.forgetFocus();
      // Clicking anywhere else dismisses the quick panel, like a menu.
      this.closeQuick();
    });
    this.window.on("close", event => {
      // Cmd+W closes the composer, not the shared HUD for the rest of the run.
      event.preventDefault();
      this.quick = null;
      this.editing = false;
      this.focus?.setHovered(false);
      this.publish();
    });
    // A reloaded page starts blank, so it needs the whole state again.
    this.listen("pet:chat:ready", () => { this.loaded = true; this.lastState = ""; this.publish(); });
    this.listen("pet:chat:open", () => this.openComposer());
    this.listen("pet:chat:close", () => this.closeComposer());
    const composingListener = (event: IpcMainEvent, value: unknown): void => {
      if (event.sender === this.window.webContents) this.imeComposing = value === true;
    };
    this.listeners.push(["pet:chat:composing", composingListener]);
    ipcMain.on("pet:chat:composing", composingListener);
    this.listen("pet:chat:history", () => options.openHistory());
    this.listen("pet:main:open", () => {
      this.handOver();
      void options.quickAction?.({ type: "main" }).catch(error => console.warn("打开主窗口失败", error));
    });
    const senseListener = (event: IpcMainEvent, value: unknown): void => {
      if (event.sender !== this.window.webContents || this.locked) return;
      const action = parseSenseAction(value);
      if (!action) return;
      if (action.type === "panel") {
        this.sensePanel = action.panel;
        if (action.panel === "closed") this.senseNote = "";
        this.publish();
        this.pointer(screen.getCursorScreenPoint());
      } else {
        if (action.type === "settings") this.handOver();
        else if (action.type !== "fix") this.sensePanel = "open";
        options.senseAction?.(action);
        this.publish();
      }
    };
    this.listeners.push(["pet:senses:action", senseListener]);
    ipcMain.on("pet:senses:action", senseListener);
    const quickListener = (event: IpcMainEvent, value: unknown): void => {
      if (event.sender !== this.window.webContents || this.locked || !this.quick) return;
      const action = parseQuickAction(value);
      if (action) void this.quickAction(action).catch(error => console.warn("快捷面板操作失败", error));
    };
    this.listeners.push(["pet:quick:action", quickListener]);
    ipcMain.on("pet:quick:action", quickListener);
    ipcMain.handle("pet:chat:send", (event, id: unknown, content: unknown, attachmentIds: unknown) => {
      if (event.sender !== this.window.webContents || typeof id !== "string" || typeof content !== "string") return { ok: false, error: "无效的发送请求" };
      const ids = Array.isArray(attachmentIds) && attachmentIds.every((item) => typeof item === "string") ? attachmentIds as string[] : [];
      return options.service.submit(id, content, ids);
    });
    ipcMain.handle("pet:chat:new-conversation", (event) =>
      event.sender === this.window.webContents ? options.newConversation() : { ok: false, error: "无效的请求" });
    ipcMain.handle("pet:chat:attach", (event, upload: unknown) => {
      const valid = event.sender === this.window.webContents ? chatAttachmentUpload(upload) : null;
      return valid ? options.service.attach(valid) : { ok: false, error: "附件无效" };
    });
    this.window.on("show", () => {
      // The native cursor monitor calls pointer() on macOS. This bounded
      // fallback also covers Windows/Linux and unavailable native helpers.
      if (!options.nativePointerActive?.() && !this.pointerTimer) this.pointerTimer = setInterval(() => this.pointer(screen.getCursorScreenPoint()), 100);
    });
    this.window.on("hide", () => { if (this.pointerTimer) clearInterval(this.pointerTimer); this.pointerTimer = null; });
    options.pet.on("move", this.reflow);
    options.pet.on("resize", this.reflow);
    options.pet.on("show", this.reflow);
    options.pet.on("hide", this.onPetHidden);
    screen.on("display-metrics-changed", this.reflow);
    screen.on("display-removed", this.reflow);
    screen.on("display-added", this.reflow);
  }

  async initialize(): Promise<void> {
    try {
      const saved = JSON.parse(await readFile(this.options.preferencePath, "utf8")) as { speechBubbles?: unknown; quickTiles?: unknown };
      this.enabled = saved.speechBubbles !== false;
      this.quickTiles = parseQuickTiles(saved.quickTiles);
    } catch { /* defaults for a new installation */ }
    await this.options.load(this.window, "pet-chat.html");
    this.options.menuChanged();
  }

  /** The composer or quick panel is open and owns keyboard focus; hovering the pet must not take it away. */
  get holdsFocus(): boolean { return this.editing || this.quick !== null; }
  syncRuntime(): void { this.options.service.syncRuntime(); }
  get bubblesEnabled(): boolean { return this.enabled; }
  async setBubblesEnabled(value: boolean): Promise<void> {
    this.enabled = value;
    await this.savePreferences();
    this.publish();
    this.options.menuChanged();
  }
  setLocked(value: boolean): void {
    this.locked = value;
    if (value) { this.editing = false; this.sensePanel = "closed"; this.quick = null; this.focus?.setHovered(false); }
    this.reflow();
    this.publish();
  }

  /** Right-click on the pet: the quick panel opens next to the pointer. */
  async openQuick(point: { x: number; y: number }): Promise<void> {
    if (this.locked || this.window.isDestroyed()) return;
    const info = await this.quickInfo();
    const { rect, edge } = placeQuickPanel(point, screen.getDisplayNearestPoint(point).workArea);
    this.quick = { rect, edge, editing: false, info };
    this.options.sensesShown?.();
    this.reflow();
    if (this.focus) this.focus.setHovered(true);
    else { if (process.platform === "win32") this.window.setFocusable(true); this.window.focus(); }
  }
  /** TA just ignored speech that may be the owner's: offer to add it, next to the pet. */
  private get ownerPrompt(): boolean {
    return !this.readiness && Boolean(this.senses?.ownerVoiceMissed);
  }

  /** Speech-model preparation shown next to the pet until it is ready. */
  setReadiness(value: { state: "ready" | "preparing" | "failed"; text: string; detail: string } | null): void {
    this.readiness = value && value.state !== "ready" ? { state: value.state, text: value.text, detail: value.detail } : null;
    this.publish();
    this.pointer(screen.getCursorScreenPoint());
  }

  /** Latest sense states, and a one-line hint shown in the panel (e.g. after opening System Settings). */
  setSenses(snapshot: SensesSnapshot | null, note?: string): void {
    this.senses = snapshot;
    if (note !== undefined) this.senseNote = note;
    this.publish();
  }

  reportGeometry(value: unknown): void {
    if (!value || typeof value !== "object") return;
    const input = value as PetChatGeometry;
    if (typeof input.hovered !== "boolean" || (input.bounds !== null && !validRect(input.bounds))) return;
    const bounds = input.bounds && validMiddle(input.bounds.middle) ? input.bounds : input.bounds && { ...input.bounds, middle: undefined };
    const changed = JSON.stringify(this.geometry) !== JSON.stringify(bounds);
    this.geometry = bounds;
    this.petHovered = input.hovered;
    if (changed) this.reflow();
    this.updateHover();
  }

  pointer(point: { x: number; y: number }): void {
    if (!this.layout || this.window.isDestroyed()) return;
    const { window, entry, home, composer, capsule, panel, status } = this.layout;
    const x = point.x - window.x, y = point.y - window.y;
    const overQuick = this.quick !== null && contains(this.quick.rect, point.x, point.y);
    const overSenses = Boolean(this.senses) && this.hudShown
      && (contains(capsule, x, y, 4) || (this.sensePanel !== "closed" && contains(panel, x, y)));
    // Only a failed chip is clickable (retry); a progress chip lets clicks through.
    const overStatus = (this.readiness?.state === "failed" || this.ownerPrompt) && contains(status, x, y);
    const hovered = !this.locked && (overQuick || overSenses || overStatus
      || (this.editing ? contains(composer, x, y) : this.hudShown && (contains(entry, x, y, 4) || contains(home, x, y, 4))));
    if (hovered !== this.hudHovered) { this.hudHovered = hovered; this.updateHover(); }
    // Lets the page label the icon under the pointer; :hover stays put in the background.
    const onCapsule = Boolean(this.senses) && this.hudShown && contains(capsule, x, y);
    const pointer = onCapsule ? JSON.stringify({ x, y }) : "";
    if (pointer !== this.capsulePointer) {
      this.capsulePointer = pointer;
      this.window.webContents.send("pet:chat:pointer", onCapsule ? { x, y } : null);
    }
    const ignore = !hovered;
    if (ignore !== this.ignoreMouse) { this.ignoreMouse = ignore; this.window.setIgnoreMouseEvents(ignore, { forward: true }); }
  }

  control(event: unknown): void {
    const progress = this.progress.accept(event);
    if (!progress) return;
    this.showText(progress.visibleContent, progress.status, false, serviceFailure(progress.failure));
  }
  disconnected(): void {
    this.progress.reset();
    this.showText("", "", false);
  }
  /** A new conversation started: what TA was saying belongs to the old one. */
  newConversation(): void {
    this.progress.reset();
    this.showText("", "", true);
  }

  /**
   * How to wake TA, after TA fell asleep; null once TA is awake again. Waits for
   * the bubble to be free: it never covers what TA just said.
   */
  sleepHint(text: string | null): void {
    if (text === null) {
      this.pendingSleepHint = "";
      if (this.text && this.text === this.shownSleepHint) this.showText("", "", true);
      return;
    }
    if (this.text || this.note) { this.pendingSleepHint = text; return; }
    this.showText(text, "completed");
    this.shownSleepHint = text;
  }

  showText(text: string, status: string, textOnly = true, failure: ServiceFailure | null = null): void {
    if (this.bubbleTimer) clearTimeout(this.bubbleTimer);
    this.bubbleTimer = null;
    this.pendingSleepHint = "";
    this.shownSleepHint = "";
    this.text = text; this.textStatus = status; this.textOnly = textOnly;
    this.note = "";
    this.noteContext = null;
    const version = ++this.noteVersion;
    if (terminalProgress(status)) this.bubbleTimer = setTimeout(() => {
      const hint = this.pendingSleepHint;
      this.text = ""; this.note = ""; this.noteContext = null; ++this.noteVersion;
      if (hint) this.sleepHint(hint);
      else this.publish();
    }, 6_000);
    this.publish();
    // A reply that failed with nothing said, or with a known cause, says why.
    if (!terminalProgress(status) || !(failure || (status === "failed" && !text))) return;
    void this.quickInfo().then(({ character }) => {
      if (version !== this.noteVersion) return;
      this.noteContext = { failure, character: character || "TA", said: Boolean(text) };
      this.refreshLocale();
    });
  }

  /** Reword cached UI notes without changing the conversation or restarting speech. */
  refreshLocale(): void {
    if (this.noteContext) {
      const { failure, character, said } = this.noteContext;
      this.note = replyFailureNote(failure, character, { said }, this.options.locale?.() ?? "zh-CN");
    }
    this.publish();
  }

  async dispose(): Promise<void> {
    if (this.hoverTimer) clearTimeout(this.hoverTimer);
    if (this.bubbleTimer) clearTimeout(this.bubbleTimer);
    if (this.pointerTimer) clearInterval(this.pointerTimer);
    for (const [name, listener] of this.listeners) ipcMain.removeListener(name, listener);
    ipcMain.removeHandler("pet:chat:send");
    ipcMain.removeHandler("pet:chat:attach");
    ipcMain.removeHandler("pet:chat:new-conversation");
    this.options.pet.removeListener("move", this.reflow);
    this.options.pet.removeListener("resize", this.reflow);
    this.options.pet.removeListener("show", this.reflow);
    this.options.pet.removeListener("hide", this.onPetHidden);
    screen.removeListener("display-metrics-changed", this.reflow);
    screen.removeListener("display-removed", this.reflow);
    screen.removeListener("display-added", this.reflow);
    this.options.service.stop();
    if (!this.window.isDestroyed()) this.window.destroy();
    await this.writer.flush();
  }

  private openComposer(): void {
    if (this.locked) return;
    this.editing = true;
    this.sensePanel = "closed";
    this.publish();
    if (this.focus) this.focus.setHovered(true);
    else { if (process.platform === "win32") this.window.setFocusable(true); this.window.focus(); }
  }

  /** tulpa is taking the foreground for good: never hand it back on leave. */
  forgetFocus(): void {
    this.focus?.forgetFocus();
  }

  /**
   * The main window is about to open: the sense panel folds away and the HUD
   * gives up input focus, or it would keep it and the pet stays on top.
   */
  private handOver(): void {
    this.sensePanel = "closed";
    this.senseNote = "";
    this.quick = null;
    // No going back to the app that was active before hovering the pet.
    this.focus?.forgetFocus();
    // A click on this non-activating panel makes it key without Electron
    // seeing focus; left key, it takes input back when tulpa activates.
    this.native?.resignKeyWindow(this.window.getNativeWindowHandle());
    this.publish();
  }

  private closeQuick(): void {
    if (!this.quick) return;
    this.quick = null;
    if (!this.editing) this.focus?.setHovered(false);
    this.reflow();
    this.publish();
  }

  private async quickInfo(): Promise<QuickInfo> {
    const fallback: QuickInfo = { speaker: true, character: null, petState: null, looks: [], lookId: null };
    return await this.options.quickInfo?.().catch(() => fallback) ?? fallback;
  }

  private async quickAction(action: QuickAction): Promise<void> {
    const quick = this.quick!;
    switch (action.type) {
      case "close": this.closeQuick(); return;
      case "edit": quick.editing = action.editing; this.publish(); return;
      case "tiles": this.quickTiles = action.tiles; this.publish(); await this.savePreferences(); return;
      // Hand focus straight from the panel to the composer; giving it back to the
      // previous app in between races the reactivation and leaves typing elsewhere.
      case "compose": this.editing = true; this.closeQuick(); this.openComposer(); return;
      case "look": case "hide": case "main": case "quit":
        this.closeQuick();
        await this.options.quickAction?.(action);
        return;
      case "tile": break;
    }
    const key = action.key;
    if (key === "bubbles") return this.setBubblesEnabled(!this.enabled);
    if (key === "screen" || key === "audio" || key === "microphone" || key === "inner") {
      this.options.senseAction?.({ type: "toggle", key });
      return;
    }
    if (key === "companion") {
      this.options.senseAction?.(this.senses?.companionPaused ? { type: "resume" } : { type: "pause", closeSenses: false });
      return;
    }
    // Opening a page or letting clicks through the pet ends the panel's job.
    if (key !== "speaker") this.closeQuick();
    await this.options.quickAction?.({ type: "tile", key });
    if (key === "speaker" && this.quick) { this.quick.info = await this.quickInfo(); this.publish(); }
  }

  private async savePreferences(): Promise<void> {
    await this.writer.write(JSON.stringify({ version: 1, speechBubbles: this.enabled, quickTiles: this.quickTiles }));
  }

  private listen(name: string, callback: () => void): void {
    const listener = (event: IpcMainEvent): void => { if (event.sender === this.window.webContents) callback(); };
    this.listeners.push([name, listener]);
    ipcMain.on(name, listener);
  }
  private onPetHidden = (): void => {
    this.petHovered = false; this.hudHovered = false; this.entryVisible = false; this.sensePanel = "closed";
    this.editing = false; this.quick = null; this.focus?.setHovered(false); this.publish();
  };
  private reflow = (): void => {
    if (!this.geometry || this.options.pet.isDestroyed() || this.window.isDestroyed()) return;
    const host = this.options.pet.getBounds();
    const { middle } = this.geometry;
    const pet: PetBody = {
      ...this.geometry, x: host.x + this.geometry.x, y: host.y + this.geometry.y,
      middle: middle && { y: host.y + middle.y, left: host.x + middle.left, right: host.x + middle.right },
    };
    // Choose the pet's monitor, never the pointer or the primary display.
    this.layoutPanelOpen = this.panelShown;
    this.layout = layoutPetChat(pet, screen.getDisplayMatching(host).workArea, this.quick?.rect ?? null, this.layoutPanelOpen);
    this.window.setBounds(this.layout.window);
    this.publish();
    this.pointer(screen.getCursorScreenPoint());
  };
  private updateHover(): void {
    if (this.petHovered || this.hudHovered || this.editing) {
      if (this.hoverTimer) clearTimeout(this.hoverTimer);
      this.hoverTimer = null;
      if (!this.entryVisible) { this.entryVisible = true; this.options.sensesShown?.(); this.publish(); }
    } else if (!this.hoverTimer && this.entryVisible) {
      this.hoverTimer = setTimeout(() => {
        this.hoverTimer = null; this.entryVisible = false; this.sensePanel = "closed"; this.senseNote = ""; this.publish();
      }, 700);
    }
  }
  /**
   * The chat entry and sense capsule beside the pet. The quick panel already
   * holds the same switches, so they step aside while it is open.
   */
  private get hudShown(): boolean { return this.entryVisible && !this.locked && !this.quick; }
  private get panelShown(): boolean { return this.hudShown && this.sensePanel !== "closed"; }

  private quickView(): PetChatState["quick"] {
    if (!this.quick || !this.layout) return null;
    const { rect, edge, editing, info } = this.quick;
    return {
      rect: { ...rect, x: rect.x - this.layout.window.x, y: rect.y - this.layout.window.y },
      edge, editing, tiles: this.quickTiles, bubbles: this.enabled, ...info,
    };
  }
  /** Put the composer away; what was typed stays for next time. */
  private closeComposer(): void {
    this.editing = false;
    this.imeComposing = false;
    this.focus?.setHovered(false);
    this.updateHover();
    this.publish();
  }

  /**
   * A press outside the open quick panel, sense panel or composer closes it,
   * like a menu. Losing focus cannot tell: tulpa is often not the active app
   * while the pointer is on the pet, and then a press elsewhere never takes
   * focus from the panel.
   */
  private watchPresses(): void {
    const sensesOpen = this.hudShown && this.sensePanel !== "closed";
    const wanted = (Boolean(this.quick) || sensesOpen || this.editing) && !this.window.isDestroyed();
    if (wanted === this.watchingPresses || !this.native) return;
    this.watchingPresses = wanted
      ? this.native.startClickMonitor(() => {
        const { x, y } = screen.getCursorScreenPoint();
        if (this.quick) {
          if (!contains(this.quick.rect, x, y)) this.closeQuick();
          return;
        }
        if (!this.layout) return;
        const { window, panel, capsule, entry, home, composer } = this.layout;
        const local = { x: x - window.x, y: y - window.y };
        if (this.editing) {
          if (!this.imeComposing && !contains(composer, local.x, local.y)) this.closeComposer();
          return;
        }
        if (this.sensePanel === "closed") return;
        if (contains(panel, local.x, local.y) || contains(capsule, local.x, local.y, 4)) return;
        // The pills beside it act on their own (and move once the panel closes).
        if (!this.layout.chatAside && (contains(entry, local.x, local.y, 4) || contains(home, local.x, local.y, 4))) return;
        this.sensePanel = "closed"; this.senseNote = ""; this.publish();
      })
      : (this.native.stopClickMonitor(), false);
  }

  private publish(): void {
    // The Windows HUD accepts keyboard focus only after an explicit open action.
    // Do not reactivate whichever application used to be in front when it closes.
    if (process.platform === "win32" && !this.window.isDestroyed() && !this.editing && !this.quick) this.window.setFocusable(false);
    // The bubble and pills make room for the sense panel: lay them out again.
    if (this.panelShown !== this.layoutPanelOpen && this.geometry) { this.reflow(); return; }
    this.watchPresses();
    if (!this.loaded || !this.layout || this.window.isDestroyed()) return;
    const bubbleVisible = Boolean(this.text || this.note) && (this.enabled || (this.textOnly && this.editing));
    const state: PetChatState = { layout: this.layout, entryVisible: this.hudShown, editing: this.editing,
      bubbleText: this.text, bubbleNote: this.note, bubbleStatus: this.textStatus, bubbleVisible,
      senses: this.senses, sensePanel: this.hudShown ? this.sensePanel : "closed", senseNote: this.senseNote,
      readiness: this.readiness, quick: this.quickView() };
    const serialized = JSON.stringify(state);
    if (serialized !== this.lastState) { this.lastState = serialized; this.window.webContents.send("pet:chat:state", state); }
    // Shown whenever TA is, even with nothing on it (the page draws nothing and
    // the pointer passes through). A panel that is hidden and shown again while
    // another app's fullscreen desktop is in front never appears there; TA's
    // own window, never hidden, always does.
    const visible = this.options.pet.isVisible();
    if (visible && !this.window.isVisible()) this.window.showInactive();
    else if (!visible && this.window.isVisible()) this.window.hide();
  }
}

function validMiddle(value: unknown): value is PetBody["middle"] {
  if (!value || typeof value !== "object") return false;
  const middle = value as NonNullable<PetBody["middle"]>;
  return [middle.y, middle.left, middle.right].every(Number.isFinite) && middle.right >= middle.left;
}
