import { characterFrame } from "../shared/character-frame";

/**
 * PNG of a Live2D canvas cropped around the character's body centerline, so
 * every 形象 card thumbnail (shipped or imported) is framed the same way.
 * The canvas must be created with preserveDrawingBuffer.
 */
export async function characterSnapshot(source: HTMLCanvasElement): Promise<Blob | null> {
  const copy = document.createElement("canvas");
  copy.width = source.width;
  copy.height = source.height;
  const context = copy.getContext("2d");
  if (!context) return null;
  context.drawImage(source, 0, 0);
  const frame = characterFrame(context.getImageData(0, 0, copy.width, copy.height).data, copy.width, copy.height);
  if (!frame) return null;
  const cropped = document.createElement("canvas");
  cropped.width = frame.width;
  cropped.height = frame.height;
  cropped.getContext("2d")!.drawImage(copy, -frame.left, -frame.top);
  return new Promise((resolve) => cropped.toBlob(resolve, "image/png"));
}
