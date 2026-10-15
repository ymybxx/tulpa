import { readFile } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import vm from "node:vm";

const require = createRequire(import.meta.url);

/**
 * Reads the neutral-pose geometry embedded in a Cubism .moc3 file.
 * The output is plain JSON data so the runtime Cubism Core never leaks into
 * the generated manifest or the Electron renderer.
 */
export async function inspectCubismModel(coreFile, mocFile) {
  const source = await readFile(coreFile, "utf8");
  const context = {
    require,
    process,
    Buffer,
    console,
    module: { exports: {} },
    exports: {},
    __dirname: path.dirname(coreFile),
    __filename: coreFile,
    TextDecoder,
    TextEncoder,
    URL,
    setTimeout,
    clearTimeout,
    atob: (value) => Buffer.from(value, "base64").toString("binary"),
  };
  context.globalThis = context;
  context.global = context;
  context.self = context;
  vm.createContext(context);
  vm.runInContext(source, context, { filename: coreFile });
  const core = context.Live2DCubismCore;
  if (!core?.Moc || !core?.Model) throw new Error("无法加载 Live2D Cubism Core");

  const mocBuffer = await readFile(mocFile);
  const arrayBuffer = mocBuffer.buffer.slice(
    mocBuffer.byteOffset,
    mocBuffer.byteOffset + mocBuffer.byteLength,
  );
  const moc = core.Moc.fromArrayBuffer(arrayBuffer);
  if (!moc) throw new Error(`无法读取 Cubism 模型: ${mocFile}`);
  const model = core.Model.fromMoc(moc);
  if (!model) throw new Error(`无法创建 Cubism 模型: ${mocFile}`);
  try {
    model.update();
    return {
      canvas: {
        width: model.canvasinfo.CanvasWidth,
        height: model.canvasinfo.CanvasHeight,
        originX: model.canvasinfo.CanvasOriginX,
        originY: model.canvasinfo.CanvasOriginY,
        pixelsPerUnit: model.canvasinfo.PixelsPerUnit,
      },
      parameters: Array.from(model.parameters.ids),
      parts: Array.from(model.parts.ids).map((id, index) => ({
        id,
        parentIndex: model.parts.parentIndices[index],
      })),
      drawables: Array.from(model.drawables.ids).map((id, index) => ({
        id,
        texture: model.drawables.textureIndices[index],
        drawOrder: model.drawables.drawOrders[index],
        renderOrder: model.drawables.renderOrders[index],
        opacity: model.drawables.opacities[index],
        partIndex: model.drawables.parentPartIndices[index],
        vertices: Array.from(model.drawables.vertexPositions[index]),
        uvs: Array.from(model.drawables.vertexUvs[index]),
        indices: Array.from(model.drawables.indices[index]),
      })),
    };
  } finally {
    model.release();
    moc._release();
  }
}
