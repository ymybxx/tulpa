import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import vm from "node:vm";

export interface InspectedCubismModel {
  mocVersion: number;
  usesAdvancedBlendMode: boolean;
  usesOffscreenDrawing: boolean;
  canvas: {
    width: number;
    height: number;
    originX: number;
    originY: number;
    pixelsPerUnit: number;
  };
  parameters: string[];
  /** `opacity` is the Part's own neutral-pose opacity, before its parents'. */
  parts: Array<{ id: string; parentIndex: number; opacity?: number }>;
  drawables: Array<{
    id: string;
    texture: number;
    drawOrder: number;
    renderOrder: number;
    opacity: number;
    partIndex: number;
    vertices: number[];
    uvs: number[];
    indices: number[];
    /** Drawables this one is clipped by. */
    masks: number[];
  }>;
}

interface CoreModel {
  canvasinfo: {
    CanvasWidth: number;
    CanvasHeight: number;
    CanvasOriginX: number;
    CanvasOriginY: number;
    PixelsPerUnit: number;
  };
  parameters: {
    ids: ArrayLike<string>;
    values: { [index: number]: number; length: number };
    defaultValues: ArrayLike<number>;
    minimumValues: ArrayLike<number>;
    maximumValues: ArrayLike<number>;
  };
  parts: { ids: ArrayLike<string>; parentIndices: ArrayLike<number>; opacities?: ArrayLike<number> };
  getRenderOrders(): ArrayLike<number>;
  drawables: {
    ids: ArrayLike<string>;
    textureIndices: ArrayLike<number>;
    drawOrders: ArrayLike<number>;
    blendModes?: ArrayLike<number>;
    opacities: ArrayLike<number>;
    parentPartIndices: ArrayLike<number>;
    vertexPositions: ArrayLike<ArrayLike<number>>;
    vertexUvs: ArrayLike<ArrayLike<number>>;
    indices: ArrayLike<ArrayLike<number>>;
    masks?: ArrayLike<ArrayLike<number>>;
    maskCounts?: ArrayLike<number>;
  };
  offscreens?: { count: number };
  update(): void;
  release(): void;
}

interface CoreMoc { _release(): void }

interface CubismCore {
  Moc: { fromArrayBuffer(value: ArrayBuffer): CoreMoc | null };
  Model: { fromMoc(value: CoreMoc): CoreModel | null };
  Version: { csmGetMocVersion(value: ArrayBuffer): number };
}

const inspectionCache = new Map<string, Promise<InspectedCubismModel>>();

export async function inspectCubismModel(
  coreFile: string,
  mocFile: string,
): Promise<InspectedCubismModel> {
  const [coreInfo, mocInfo] = await Promise.all([stat(coreFile), stat(mocFile)]);
  const key = [
    coreFile,
    coreInfo.size,
    coreInfo.mtimeMs,
    mocFile,
    mocInfo.size,
    mocInfo.mtimeMs,
  ].join("\u0000");
  const existing = inspectionCache.get(key);
  if (existing) return existing;
  const inspection = inspectCubismModelUncached(coreFile, mocFile);
  inspectionCache.set(key, inspection);
  try {
    return await inspection;
  } catch (error) {
    if (inspectionCache.get(key) === inspection) inspectionCache.delete(key);
    throw error;
  }
}

async function loadCoreModel(
  coreFile: string,
  mocFile: string,
): Promise<{ core: CubismCore; moc: CoreMoc; model: CoreModel; arrayBuffer: ArrayBuffer }> {
  const source = await readFile(coreFile, "utf8");
  const require = createRequire(__filename);
  const context: Record<string, unknown> = {
    require,
    // The Emscripten loader registers abort handlers with process.on. On the
    // real process they would keep this context and its WebAssembly heap
    // alive for the app's lifetime, one heap per inspection.
    process: Object.create(process, { on: { value: () => undefined } }),
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
    atob: (value: string) => Buffer.from(value, "base64").toString("binary"),
  };
  context.globalThis = context;
  context.global = context;
  context.self = context;
  vm.createContext(context);
  vm.runInContext(source, context, { filename: coreFile });
  const core = context.Live2DCubismCore as CubismCore | undefined;
  if (!core?.Moc || !core.Model) throw new Error("无法加载 Live2D Cubism Core");

  const bytes = await readFile(mocFile);
  const arrayBuffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const moc = core.Moc.fromArrayBuffer(arrayBuffer);
  if (!moc) throw new Error(`无法读取 Cubism 模型: ${mocFile}`);
  const model = core.Model.fromMoc(moc);
  if (!model) {
    moc._release();
    throw new Error(`无法创建 Cubism 模型: ${mocFile}`);
  }
  return { core, moc, model, arrayBuffer };
}

async function inspectCubismModelUncached(
  coreFile: string,
  mocFile: string,
): Promise<InspectedCubismModel> {
  const { core, moc, model, arrayBuffer } = await loadCoreModel(coreFile, mocFile);
  try {
    model.update();
    const renderOrders = model.getRenderOrders();
    const blendModes = model.drawables.blendModes ?? [];
    const mocVersion = core.Version.csmGetMocVersion(arrayBuffer);
    return {
      mocVersion,
      usesAdvancedBlendMode:
        mocVersion >= 6 && Array.from(blendModes).some((value) => value > 3),
      usesOffscreenDrawing: (model.offscreens?.count ?? 0) > 0,
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
        parentIndex: model.parts.parentIndices[index] ?? -1,
        opacity: model.parts.opacities?.[index] ?? 1,
      })),
      drawables: Array.from(model.drawables.ids).map((id, index) => ({
        id,
        texture: model.drawables.textureIndices[index] ?? -1,
        drawOrder: model.drawables.drawOrders[index] ?? 0,
        renderOrder: renderOrders[index] ?? 0,
        opacity: model.drawables.opacities[index] ?? 0,
        partIndex: model.drawables.parentPartIndices[index] ?? -1,
        vertices: Array.from(model.drawables.vertexPositions[index] ?? []),
        uvs: Array.from(model.drawables.vertexUvs[index] ?? []),
        indices: Array.from(model.drawables.indices[index] ?? []),
        masks: Array.from(model.drawables.masks?.[index] ?? [])
          .slice(0, model.drawables.maskCounts?.[index] ?? 0),
      })),
    };
  } finally {
    model.release();
    moc._release();
  }
}

/** What moving one parameter alone does to the drawables: which appear, disappear and move. */
export interface ParameterEffect {
  shown: number[];
  hidden: number[];
  moved: number[];
}

/**
 * Moves each parameter alone to both ends from the default pose (no physics)
 * and records which drawables change visibility or shape. Model data only:
 * the same for every model, whatever its parameters are called.
 */
export async function probeParameterEffects(
  coreFile: string,
  mocFile: string,
  parameterIds: readonly string[],
): Promise<Record<string, ParameterEffect>> {
  const { moc, model } = await loadCoreModel(coreFile, mocFile);
  try {
    const ids = Array.from(model.parameters.ids);
    const count = model.drawables.ids.length;
    const reset = () => { for (let i = 0; i < ids.length; i++) model.parameters.values[i] = model.parameters.defaultValues[i]!; };
    const visibleAndShape = () => {
      model.update();
      const visible: boolean[] = [], shapes: Float32Array[] = [];
      for (let i = 0; i < count; i++) {
        let opacity = model.drawables.opacities[i] ?? 0;
        for (let part = model.drawables.parentPartIndices[i] ?? -1; part >= 0; part = model.parts.parentIndices[part] ?? -1) {
          opacity *= model.parts.opacities?.[part] ?? 1;
        }
        visible.push(opacity > 0.3);
        shapes.push(Float32Array.from(model.drawables.vertexPositions[i] ?? []));
      }
      return { visible, shapes };
    };
    reset();
    const base = visibleAndShape();
    const effects: Record<string, ParameterEffect> = {};
    for (const id of parameterIds) {
      const index = ids.indexOf(id);
      if (index < 0) continue;
      const shown = new Set<number>(), hidden = new Set<number>(), moved = new Set<number>();
      for (const value of [model.parameters.minimumValues[index]!, model.parameters.maximumValues[index]!]) {
        if (value === model.parameters.defaultValues[index]) continue;
        reset();
        model.parameters.values[index] = value;
        const probe = visibleAndShape();
        for (let i = 0; i < count; i++) {
          if (probe.visible[i] && !base.visible[i]) shown.add(i);
          if (!probe.visible[i] && base.visible[i]) hidden.add(i);
          const a = base.shapes[i]!, b = probe.shapes[i]!;
          for (let k = 0; k < a.length; k++) if (Math.abs(a[k]! - b[k]!) > 1e-3) { moved.add(i); break; }
        }
      }
      effects[id] = { shown: [...shown], hidden: [...hidden], moved: [...moved] };
    }
    return effects;
  } finally {
    model.release();
    moc._release();
  }
}
