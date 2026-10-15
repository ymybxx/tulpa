import { bindLocalizedAttribute } from "../i18n/dom";
import type { TextureEditUnit } from "../shared/contracts";

/**
 * Thumbnails that show where a texture piece sits on the character: a close-up
 * of the full-body guide with the piece lit up and everything else dimmed. A
 * bare UV patch says little about which part it is; its place on the body does.
 */
export class PieceLocator {
  private guide: Promise<HTMLImageElement> | null = null;
  private readonly masks = new Map<string, Promise<HTMLImageElement>>();

  constructor(
    private readonly modelId: string,
    private readonly imageUrl: (relativePath: string) => string,
  ) {}

  /** A canvas that fills in once the guide and the piece's mask have loaded. */
  thumbnail(unit: TextureEditUnit, size = 96): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = canvas.height = Math.round(size * ratio);
    canvas.style.width = canvas.style.height = `${size}px`;
    canvas.className = "le-locator";
    void this.draw(canvas, unit).catch(() => this.drawTexture(canvas, unit));
    return canvas;
  }

  private loadGuide(): Promise<HTMLImageElement> {
    this.guide ??= window.desktopPet.getModelAlignmentGuide(this.modelId).then((guide) => loadImage(guide.url));
    return this.guide;
  }

  private loadMask(unit: TextureEditUnit): Promise<HTMLImageElement> {
    let mask = this.masks.get(unit.id);
    if (!mask) {
      mask = loadImage(this.imageUrl(unit.modelMaskPath));
      this.masks.set(unit.id, mask);
    }
    return mask;
  }

  private async draw(canvas: HTMLCanvasElement, unit: TextureEditUnit): Promise<void> {
    const [guide, mask] = await Promise.all([this.loadGuide(), this.loadMask(unit)]);
    const box = unit.modelMaskBounds;
    if (!box.width || !box.height || isEmpty(mask)) {
      // Pieces only shown during an action are invisible in the neutral pose.
      await this.drawTexture(canvas, unit, true);
      return;
    }
    const scale = guide.naturalWidth / box.canvasWidth;
    const minimum = Math.min(box.canvasWidth, box.canvasHeight) * 0.28;
    const side = Math.min(Math.max(box.width, box.height) * 2.2, Math.max(box.canvasWidth, box.canvasHeight));
    const crop = Math.max(side, minimum);
    // Keep the crop inside the picture; when it is wider or taller than the picture, center it.
    const place = (center: number, extent: number) => crop >= extent ? (extent - crop) / 2 : clamp(center - crop / 2, 0, extent - crop);
    const left = place(box.x + box.width / 2, box.canvasWidth);
    const top = place(box.y + box.height / 2, box.canvasHeight);
    const size = canvas.width;
    const k = size / crop;
    const maskAt = [(box.x - left) * k, (box.y - top) * k, box.width * k, box.height * k] as const;
    const context = canvas.getContext("2d")!;
    context.clearRect(0, 0, size, size);
    // The body around it, grey and dark, only for orientation.
    context.save();
    context.filter = "grayscale(1) brightness(0.3)";
    context.drawImage(guide, left * scale, top * scale, crop * scale, crop * scale, 0, 0, size, size);
    context.restore();
    // A thick warm outline so the piece's shape reads even when it is thin or pale.
    const rim = document.createElement("canvas");
    rim.width = rim.height = size;
    const rimContext = rim.getContext("2d")!;
    rimContext.drawImage(mask, ...maskAt);
    rimContext.globalCompositeOperation = "source-in";
    rimContext.fillStyle = "rgb(242, 193, 78)";
    rimContext.fillRect(0, 0, size, size);
    context.save();
    context.filter = `blur(${Math.max(2, size / 40)}px)`;
    for (let pass = 0; pass < 3; pass += 1) context.drawImage(rim, 0, 0);
    context.restore();
    // The piece itself, in its own colors.
    const piece = document.createElement("canvas");
    piece.width = piece.height = size;
    const pieceContext = piece.getContext("2d")!;
    pieceContext.drawImage(guide, left * scale, top * scale, crop * scale, crop * scale, 0, 0, size, size);
    pieceContext.globalCompositeOperation = "destination-in";
    pieceContext.drawImage(mask, ...maskAt);
    context.drawImage(piece, 0, 0);
    this.drawMinimap(context, guide, box, size);
  }

  /** A tiny whole body in the corner with a box where the piece is. */
  private drawMinimap(context: CanvasRenderingContext2D, guide: HTMLImageElement, box: TextureEditUnit["modelMaskBounds"], size: number): void {
    const height = size * 0.42;
    const width = height * box.canvasWidth / box.canvasHeight;
    const x = size - width - size * 0.04;
    const y = size - height - size * 0.04;
    context.save();
    context.fillStyle = "rgba(15, 18, 28, 0.85)";
    context.beginPath();
    context.roundRect(x - 2, y - 2, width + 4, height + 4, size * 0.04);
    context.fill();
    context.filter = "grayscale(1) brightness(0.55)";
    context.drawImage(guide, x, y, width, height);
    context.filter = "none";
    const k = width / box.canvasWidth;
    context.strokeStyle = "rgb(242, 193, 78)";
    context.lineWidth = Math.max(1.5, size / 64);
    context.strokeRect(x + box.x * k, y + box.y * k, Math.max(3, box.width * k), Math.max(3, box.height * k));
    context.restore();
  }

  /** Fallback: the texture itself, centered. */
  private async drawTexture(canvas: HTMLCanvasElement, unit: TextureEditUnit, hidden = false): Promise<void> {
    const image = await loadImage(this.imageUrl(unit.previewPath));
    const size = canvas.width;
    const context = canvas.getContext("2d")!;
    context.clearRect(0, 0, size, size);
    const fit = Math.min(size / image.naturalWidth, size / image.naturalHeight) * 0.9;
    const width = image.naturalWidth * fit;
    const height = image.naturalHeight * fit;
    context.drawImage(image, (size - width) / 2, (size - height) / 2, width, height);
    if (hidden) bindLocalizedAttribute(canvas, "title", "这块平时不显示，做动作或切表情时才出现");
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`图片加载失败：${url}`));
    image.src = url;
  });
}

function isEmpty(mask: HTMLImageElement): boolean {
  const canvas = document.createElement("canvas");
  const step = Math.max(1, Math.floor(Math.max(mask.naturalWidth, mask.naturalHeight) / 64));
  canvas.width = Math.max(1, Math.ceil(mask.naturalWidth / step));
  canvas.height = Math.max(1, Math.ceil(mask.naturalHeight / step));
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(mask, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let index = 3; index < pixels.length; index += 4) if (pixels[index]! > 8) return false;
  return true;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
