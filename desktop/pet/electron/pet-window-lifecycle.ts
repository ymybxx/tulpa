interface CloseEvent {
  preventDefault(): void;
}

interface PetWindow {
  isDestroyed(): boolean;
  isVisible(): boolean;
  showInactive(): void;
  hide(): void;
  on(event: "close", listener: (event: CloseEvent) => void): unknown;
  on(event: "show" | "hide" | "closed", listener: () => void): unknown;
}

/** Closing a companion hides it; showing also recovers a destroyed window. */
export class PetWindowLifecycle {
  private creation: Promise<void> | null = null;

  constructor(private readonly options: {
    current(): PetWindow | null;
    create(): Promise<void>;
    quitting(): boolean;
    visibilityChanged(): void;
  }) {}

  bind(window: PetWindow): void {
    window.on("close", (event) => {
      if (this.options.quitting()) return;
      event.preventDefault();
      window.hide();
    });
    for (const event of ["show", "hide", "closed"] as const) {
      window.on(event, () => {
        if (!this.options.quitting()) this.options.visibilityChanged();
      });
    }
  }

  show(): Promise<void> {
    if (this.options.quitting()) return Promise.resolve();
    if (this.creation) return this.creation;
    const window = this.options.current();
    if (window && !window.isDestroyed()) {
      window.showInactive();
      return Promise.resolve();
    }
    this.creation = Promise.resolve()
      .then(() => {
        if (!this.options.quitting()) return this.options.create();
      })
      .finally(() => { this.creation = null; });
    return this.creation;
  }

  toggle(): Promise<void> {
    if (this.options.quitting()) return Promise.resolve();
    if (this.creation) return this.creation;
    const window = this.options.current();
    if (window && !window.isDestroyed() && window.isVisible()) {
      window.hide();
      return Promise.resolve();
    }
    return this.show();
  }
}
