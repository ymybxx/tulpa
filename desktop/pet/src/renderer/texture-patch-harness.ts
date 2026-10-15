import type { TextureAtlasPatchSet } from "../shared/contracts";
import { CubismTexturePatchCompositor } from "./cubism-texture-patch-compositor";

interface HarnessResult {
  ok: boolean;
  pixels: number[];
  expected: number[];
  glError: number;
  largePatchMs: number;
  targetMs: number;
  resources: ResourceReport;
}

interface ResourceReport {
  balanced: boolean;
  created: Record<ResourceKind, number>;
  deleted: Record<ResourceKind, number>;
  outstanding: Record<ResourceKind, number>;
}

type ResourceKind =
  | "textures"
  | "framebuffers"
  | "buffers"
  | "vertexArrays"
  | "programs"
  | "shaders";

void run().then((result) => {
  (window as unknown as { texturePatchResult: HarnessResult }).texturePatchResult = result;
}).catch((error) => {
  (window as unknown as { texturePatchError: string }).texturePatchError = String(
    error instanceof Error ? error.stack ?? error.message : error,
  );
});

async function run(): Promise<HarnessResult> {
  const canvas = document.getElementById("texture-patch-canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgl2", { premultipliedAlpha: true });
  if (!context) throw new Error("WebGL2 unavailable");
  const tracked = trackResources(context);
  const gl = tracked.gl;
  let candidate: ReturnType<typeof createCandidate> | null = null;
  let compositor: CubismTexturePatchCompositor | null = null;
  let pixels: number[] = [];
  let expected: number[] = [];
  let largePatchMs = Number.POSITIVE_INFINITY;
  let operationGlError: number = gl.NO_ERROR;
  const basePixels = new Uint8ClampedArray([
    10, 20, 30, 255, 11, 21, 31, 255, 12, 22, 32, 255, 13, 23, 33, 255,
    40, 50, 60, 100, 41, 51, 61, 255, 42, 52, 62, 255, 43, 53, 63, 255,
    70, 80, 90, 255, 71, 81, 91, 255, 72, 82, 92, 255, 73, 83, 93, 255,
    100, 110, 120, 255, 101, 111, 121, 255, 102, 112, 122, 255, 103, 113, 123, 255,
  ]);
  const baseUrl = await imageUrl(4, 4, basePixels);
  candidate = await createBitmapCandidate(gl, baseUrl);
  const patchUrl = await imageUrl(3, 2, [
    200, 100, 50, 128,
    210, 110, 60, 255,
    180, 80, 40, 0,
    20, 120, 220, 255,
    30, 130, 230, 255,
    40, 140, 240, 255,
  ]);
  const maskUrl = await imageUrl(3, 2, [
    255, 255, 255, 255,
    255, 255, 255, 0,
    255, 255, 255, 255,
    255, 255, 255, 255,
    255, 255, 255, 255,
    255, 255, 255, 0,
  ]);
  const overlapPatchUrl = await imageUrl(2, 1, [
    50, 200, 100, 255,
    90, 40, 220, 255,
  ]);
  const overlapMaskUrl = await imageUrl(2, 1, [
    255, 255, 255, 255,
    255, 255, 255, 0,
  ]);
  const patchSet: TextureAtlasPatchSet = {
    atlasIndex: 0,
    atlasFile: "atlas.png",
    sourceWidth: 4,
    sourceHeight: 4,
    baseUrl: "unused",
    patches: [{
      textureUnitId: "top-strip",
      bounds: { x: 0, y: 0, width: 3, height: 2 },
      imageUrl: patchUrl,
      maskUrl,
    }, {
      textureUnitId: "top-strip-overlap",
      bounds: { x: 1, y: 0, width: 2, height: 1 },
      imageUrl: overlapPatchUrl,
      maskUrl: overlapMaskUrl,
    }],
  };
  try {
    compositor = new CubismTexturePatchCompositor(gl);
    await compositor.apply(candidate, patchSet, () => true);
    pixels = readTexture(gl, candidate.texture, 4, 4);
    // Production uploads an ImageBitmap without a vertical flip, so readPixels
    // row zero must match the CPU composer's PNG top row. The asymmetric base
    // and top-row patches catch both whole-atlas displacement and local flips.
    expected = [...basePixels];
    const alphaReduced = 0;
    expected.splice(alphaReduced, 4, 100, 50, 25, 128);
    const overlapFilled = 4;
    expected.splice(overlapFilled, 4, 50, 200, 100, 255);
    const removed = 8;
    expected.splice(removed, 4, 0, 0, 0, 0);
    expected.splice(16, 4, 8, 47, 86, 100);
    expected.splice(20, 4, 30, 130, 230, 255);
    largePatchMs = await benchmarkLargePatch(gl, compositor);
    operationGlError = gl.getError();
  } finally {
    URL.revokeObjectURL(baseUrl);
    URL.revokeObjectURL(patchUrl);
    URL.revokeObjectURL(maskUrl);
    URL.revokeObjectURL(overlapPatchUrl);
    URL.revokeObjectURL(overlapMaskUrl);
    compositor?.release();
    if (candidate) gl.deleteTexture(candidate.texture);
  }
  const cleanupGlError = gl.getError();
  const resources = tracked.report();
  return {
    ok: pixels.every((value, index) => Math.abs(value - expected[index]!) <= 1) &&
      operationGlError === gl.NO_ERROR && cleanupGlError === gl.NO_ERROR &&
      largePatchMs < 1_000 && resources.balanced,
    pixels,
    expected,
    glError: operationGlError || cleanupGlError,
    largePatchMs: Math.round(largePatchMs * 100) / 100,
    targetMs: 1_000,
    resources,
  };
}

async function createBitmapCandidate(
  gl: WebGL2RenderingContext,
  url: string,
): Promise<{ texture: WebGLTexture; width: number; height: number }> {
  const response = await fetch(url);
  const bitmap = await createImageBitmap(await response.blob(), {
    premultiplyAlpha: "premultiply",
    colorSpaceConversion: "default",
  });
  const texture = gl.createTexture();
  if (!texture) {
    bitmap.close();
    throw new Error("bitmap candidate texture unavailable");
  }
  try {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { texture, width: bitmap.width, height: bitmap.height };
  } catch (error) {
    gl.deleteTexture(texture);
    throw error;
  } finally {
    bitmap.close();
  }
}

async function benchmarkLargePatch(
  gl: WebGL2RenderingContext,
  compositor: CubismTexturePatchCompositor,
): Promise<number> {
  const candidate = createCandidate(gl, 2048, 2048, [36, 24, 20, 255]);
  const width = 750;
  const height = 1209;
  const patch = new Uint8ClampedArray(width * height * 4);
  const mask = new Uint8ClampedArray(width * height * 4);
  for (let offset = 0; offset < patch.length; offset += 4) {
    patch.set([120, 48, 72, 255], offset);
    mask.set([255, 255, 255, 255], offset);
  }
  const [patchUrl, maskUrl] = await Promise.all([
    imageUrl(width, height, patch),
    imageUrl(width, height, mask),
  ]);
  try {
    const started = performance.now();
    await compositor.apply(candidate, {
      atlasIndex: 0,
      atlasFile: "large-atlas.png",
      sourceWidth: 2048,
      sourceHeight: 2048,
      baseUrl: "unused",
      patches: [{
        textureUnitId: "large-material",
        bounds: { x: 7, y: 16, width, height },
        imageUrl: patchUrl,
        maskUrl,
      }],
    }, () => true);
    gl.finish();
    return performance.now() - started;
  } finally {
    URL.revokeObjectURL(patchUrl);
    URL.revokeObjectURL(maskUrl);
    gl.deleteTexture(candidate.texture);
  }
}

function createCandidate(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  rgba: [number, number, number, number],
) {
  const texture = gl.createTexture();
  if (!texture) throw new Error("candidate texture unavailable");
  const pixels = new Uint8Array(width * height * 4);
  for (let offset = 0; offset < pixels.length; offset += 4) pixels.set(rgba, offset);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { texture, width, height };
}

async function imageUrl(
  width: number,
  height: number,
  rgba: number[] | Uint8ClampedArray,
): Promise<string> {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D context unavailable");
  context.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  return URL.createObjectURL(await canvas.convertToBlob({ type: "image/png" }));
}

function readTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  width: number,
  height: number,
): number[] {
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) throw new Error("read framebuffer unavailable");
  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const values = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, values);
    return [...values];
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);
  }
}

function trackResources(context: WebGL2RenderingContext): {
  gl: WebGL2RenderingContext;
  report(): ResourceReport;
} {
  const resources: Record<ResourceKind, Set<object>> = {
    textures: new Set(),
    framebuffers: new Set(),
    buffers: new Set(),
    vertexArrays: new Set(),
    programs: new Set(),
    shaders: new Set(),
  };
  const deleted: Record<ResourceKind, Set<object>> = {
    textures: new Set(),
    framebuffers: new Set(),
    buffers: new Set(),
    vertexArrays: new Set(),
    programs: new Set(),
    shaders: new Set(),
  };
  const creates = new Map<PropertyKey, ResourceKind>([
    ["createTexture", "textures"],
    ["createFramebuffer", "framebuffers"],
    ["createBuffer", "buffers"],
    ["createVertexArray", "vertexArrays"],
    ["createProgram", "programs"],
    ["createShader", "shaders"],
  ]);
  const deletes = new Map<PropertyKey, ResourceKind>([
    ["deleteTexture", "textures"],
    ["deleteFramebuffer", "framebuffers"],
    ["deleteBuffer", "buffers"],
    ["deleteVertexArray", "vertexArrays"],
    ["deleteProgram", "programs"],
    ["deleteShader", "shaders"],
  ]);
  const originals = new Map<PropertyKey, PropertyDescriptor | undefined>();
  for (const [property, kind] of [...creates, ...deletes]) {
    if (originals.has(property)) continue;
    const original = context[property as keyof WebGL2RenderingContext];
    if (typeof original !== "function") throw new Error(`WebGL resource method missing: ${String(property)}`);
    originals.set(property, Object.getOwnPropertyDescriptor(context, property));
    const createKind = creates.get(property);
    const deleteKind = deletes.get(property);
    Object.defineProperty(context, property, {
      configurable: true,
      value: (...args: unknown[]) => {
        const result = Reflect.apply(original, context, args);
        if (createKind && result && typeof result === "object") {
          resources[createKind].add(result);
        }
        const resource = args[0];
        if (
          deleteKind && resource && typeof resource === "object" &&
          resources[deleteKind].has(resource)
        ) deleted[deleteKind].add(resource);
        return result;
      },
    });
  }
  return {
    gl: context,
    report: () => {
      const kinds = Object.keys(resources) as ResourceKind[];
      const created = Object.fromEntries(kinds.map((kind) => [kind, resources[kind].size])) as
        Record<ResourceKind, number>;
      const deletedCounts = Object.fromEntries(kinds.map((kind) => [kind, deleted[kind].size])) as
        Record<ResourceKind, number>;
      const outstanding = Object.fromEntries(kinds.map((kind) => [
        kind,
        resources[kind].size - deleted[kind].size,
      ])) as Record<ResourceKind, number>;
      const live = liveResourceCount(context, resources);
      const result = {
        balanced: kinds.every((kind) => outstanding[kind] === 0 && live[kind] === 0),
        created,
        deleted: deletedCounts,
        outstanding,
      };
      for (const [property, descriptor] of originals) {
        if (descriptor) Object.defineProperty(context, property, descriptor);
        else delete (context as unknown as Record<PropertyKey, unknown>)[property];
      }
      return result;
    },
  };
}

function liveResourceCount(
  gl: WebGL2RenderingContext,
  resources: Record<ResourceKind, Set<object>>,
): Record<ResourceKind, number> {
  const checks: Record<ResourceKind, (resource: object) => boolean> = {
    textures: (resource) => gl.isTexture(resource as WebGLTexture),
    framebuffers: (resource) => gl.isFramebuffer(resource as WebGLFramebuffer),
    buffers: (resource) => gl.isBuffer(resource as WebGLBuffer),
    vertexArrays: (resource) => gl.isVertexArray(resource as WebGLVertexArrayObject),
    programs: (resource) => gl.isProgram(resource as WebGLProgram),
    shaders: (resource) => gl.isShader(resource as WebGLShader),
  };
  return Object.fromEntries((Object.keys(resources) as ResourceKind[]).map((kind) => [
    kind,
    [...resources[kind]].filter(checks[kind]).length,
  ])) as Record<ResourceKind, number>;
}
