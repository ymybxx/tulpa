import type { CubismModel } from "@cubism/model/cubismmodel";

/** How much of its brightness the rest of the model keeps while pieces are pointed at. */
const SPOTLIGHT_DIM = 0.32;

/**
 * - `spotlight`: the live preview. The pieces keep their look and the rest of
 *   the model turns dark but stays solid, so pointing at a row never makes the
 *   character look washed out.
 * - `solo`: thumbnails. Only the pieces are drawn, so a strand of back hair is
 *   not covered by the head in front of it.
 */
export type LayerHighlightMode = "spotlight" | "solo";

type Colors = ReturnType<CubismModel["getOverrideMultiplyAndScreenColor"]>;
type Color = ReturnType<Colors["getDrawableMultiplyColor"]>;

/** A draw-time override keeps animation state and clipping masks intact.
 * Core opacity arrays remain untouched, so removing a hide reveals the current
 * authored state, including changes made while that layer was hidden.
 */
export class CubismLayerVisibility {
  private readonly indices: Map<string, number>;
  private readonly hidden = new Set<number>();
  private readonly highlights = new Set<number>();
  private mode: LayerHighlightMode = "spotlight";
  private readonly originalOpacity: CubismModel["getDrawableOpacity"];
  private readonly colors: Colors | null;
  private readonly originalMultiply: Colors["getDrawableMultiplyColor"] | null;
  private readonly originalVisible: CubismModel["getDrawableDynamicFlagIsVisible"];
  private installed = false;
  private dimming = false;
  private revealing = false;
  private readonly opacity = (index: number): number => {
    const original = this.originalOpacity.call(this.model, index);
    if (this.highlights.size) {
      // The pieces being looked at show even when this pose or the user hides them.
      if (this.highlights.has(index)) return original > 0.02 ? original : 1;
      if (this.mode === "solo") return 0;
    }
    return this.hidden.has(index) ? 0 : original;
  };
  /** Thumbnails draw a piece even while its Part is switched off (another hairstyle). */
  private readonly visible = (index: number): boolean =>
    this.highlights.has(index) || this.originalVisible.call(this.model, index);
  private readonly multiply = (index: number): Color => {
    const color = this.originalMultiply!.call(this.colors, index);
    if (this.highlights.has(index)) return color;
    // A copy: the returned color is the model's own stored value.
    return { ...color, r: color.r * SPOTLIGHT_DIM, g: color.g * SPOTLIGHT_DIM, b: color.b * SPOTLIGHT_DIM } as Color;
  };

  constructor(private readonly model: CubismModel) {
    const ids: readonly string[] = model.getModel().drawables.ids;
    this.indices = new Map(ids.map((id, index) => [id, index]));
    this.originalOpacity = model.getDrawableOpacity;
    this.originalVisible = model.getDrawableDynamicFlagIsVisible;
    this.colors = model.getOverrideMultiplyAndScreenColor?.() ?? null;
    this.originalMultiply = this.colors?.getDrawableMultiplyColor ?? null;
  }
  configure(ids: readonly string[] = []): void {
    this.hidden.clear();
    for (const id of ids) {
      const index = this.indices.get(id);
      if (index !== undefined) this.hidden.add(index);
    }
    this.syncInstall();
  }

  /** No wrapper or extra work in the normal draw path when nothing is hidden or highlighted. */
  private syncInstall(): void {
    const needed = this.hidden.size > 0 || this.highlights.size > 0;
    if (needed && !this.installed) {
      this.model.getDrawableOpacity = this.opacity;
      this.installed = true;
    } else if (!needed && this.installed) {
      this.model.getDrawableOpacity = this.originalOpacity;
      this.installed = false;
    }
    const reveal = this.highlights.size > 0 && this.mode === "solo";
    if (reveal !== this.revealing) {
      this.model.getDrawableDynamicFlagIsVisible = reveal ? this.visible : this.originalVisible;
      this.revealing = reveal;
    }
    const dim = this.highlights.size > 0 && this.mode === "spotlight" && Boolean(this.colors);
    if (dim && !this.dimming) {
      this.colors!.getDrawableMultiplyColor = this.multiply;
      this.dimming = true;
    } else if (!dim && this.dimming) {
      this.colors!.getDrawableMultiplyColor = this.originalMultiply!;
      this.dimming = false;
    }
  }
  authoredOpacity(index: number): number { return this.originalOpacity.call(this.model, index); }
  /** Whether the character shows this drawable, whatever is being pointed at. */
  isShown(index: number): boolean {
    return !this.hidden.has(index) && this.originalVisible.call(this.model, index) &&
      this.originalOpacity.call(this.model, index) > 0.02;
  }
  /**
   * Points out `ids`. A tint on the piece itself vanishes on white clothing;
   * darkening or hiding the rest reads on any color.
   */
  highlight(ids: readonly string[], mode: LayerHighlightMode = "spotlight"): void {
    this.highlights.clear();
    for (const id of new Set(ids)) {
      const index = this.indices.get(id);
      if (index !== undefined) this.highlights.add(index);
    }
    this.mode = mode;
    this.syncInstall();
  }
  release(): void {
    this.highlight([]);
    this.configure([]);
  }
}
