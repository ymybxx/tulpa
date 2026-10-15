import { afterEach, describe, expect, it, vi } from "vitest";

import type { TextureAtlasPatchSet } from "../shared/contracts";
import { CubismTexturePatchCompositor } from "./cubism-texture-patch-compositor";

describe("CubismTexturePatchCompositor resource ownership", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each([
    ["shader source", "shaderSource", 1],
    ["second shader", "createShader", 2],
    ["program", "createProgram", 1],
    ["vertex array", "createVertexArray", 1],
    ["vertex buffer", "createBuffer", 1],
    ["candidate framebuffer", "createFramebuffer", 1],
    ["scratch framebuffer", "createFramebuffer", 2],
    ["uniform", "getUniformLocation", 2],
    ["vertex upload", "bufferData", 1],
  ])("releases every constructor resource when %s allocation fails", (_name, method, call) => {
    const fake = createFakeGl({ method, call });

    expect(() => new CubismTexturePatchCompositor(fake.gl)).toThrow("injected failure");

    expect(fake.liveResources()).toEqual([]);
  });

  it.each([
    ["second texture creation", "createTexture", 2],
    ["third texture creation", "createTexture", 3],
    ["texture parameter initialization", "texParameteri", 1],
    ["patch bitmap upload", "texImage2D", 2],
  ])("releases all temporary textures when %s fails", async (_name, method, call) => {
    const fake = createFakeGl();
    const compositor = new CubismTexturePatchCompositor(fake.gl);
    const closed: number[] = [];
    let bitmapId = 0;
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(),
    })));
    vi.stubGlobal("createImageBitmap", vi.fn(async () => {
      const id = bitmapId += 1;
      return {
        width: 2,
        height: 2,
        close: () => closed.push(id),
      } as ImageBitmap;
    }));
    fake.failAt(method, call);

    await expect(compositor.apply(
      { texture: { external: true } as unknown as WebGLTexture, width: 4, height: 4 },
      patchSet(),
      () => true,
    )).rejects.toThrow("injected failure");

    expect(fake.liveResources("texture")).toEqual([]);
    expect(closed.sort()).toEqual([1, 2]);
    compositor.release();
    expect(fake.liveResources()).toEqual([]);
  });
});

function patchSet(): TextureAtlasPatchSet {
  return {
    atlasIndex: 0,
    atlasFile: "textures/atlas.png",
    sourceWidth: 4,
    sourceHeight: 4,
    baseUrl: "pet-model://model/textures/atlas.png?textureBase=1",
    patches: [{
      textureUnitId: "leg",
      bounds: { x: 1, y: 1, width: 2, height: 2 },
      imageUrl: "pet-model://model/material.png",
      maskUrl: "pet-model://model/mask.png",
    }],
  };
}

interface FailurePoint {
  method: string;
  call: number;
}

interface ResourceRecord {
  kind: string;
  id: number;
}

function createFakeGl(initialFailure?: FailurePoint): {
  gl: WebGL2RenderingContext;
  failAt: (method: string, call: number) => void;
  liveResources: (kind?: string) => string[];
} {
  let nextId = 0;
  let failure = initialFailure;
  const calls = new Map<string, number>();
  const created = new Map<ResourceRecord, string>();
  const deleted = new Set<ResourceRecord>();
  const make = (kind: string): ResourceRecord => {
    const resource = { kind, id: nextId += 1 };
    created.set(resource, `${kind}:${resource.id}`);
    return resource;
  };
  const step = (method: string): void => {
    const call = (calls.get(method) ?? 0) + 1;
    calls.set(method, call);
    if (failure?.method === method && failure.call === call) {
      throw new Error(`injected failure: ${method}:${call}`);
    }
  };
  const create = (method: string, kind: string): ResourceRecord => {
    step(method);
    return make(kind);
  };
  const remove = (method: string, value: ResourceRecord | null): void => {
    step(method);
    if (value) deleted.add(value);
  };

  const constants = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    ARRAY_BUFFER_BINDING: 0x8894,
    STATIC_DRAW: 0x88e4,
    FLOAT: 0x1406,
    FRAMEBUFFER: 0x8d40,
    READ_FRAMEBUFFER: 0x8ca8,
    DRAW_FRAMEBUFFER: 0x8ca9,
    READ_FRAMEBUFFER_BINDING: 0x8caa,
    DRAW_FRAMEBUFFER_BINDING: 0x8ca6,
    FRAMEBUFFER_COMPLETE: 0x8cd5,
    COLOR_ATTACHMENT0: 0x8ce0,
    TEXTURE_2D: 0x0de1,
    TEXTURE0: 0x84c0,
    TEXTURE1: 0x84c1,
    TEXTURE2: 0x84c2,
    ACTIVE_TEXTURE: 0x84e0,
    TEXTURE_BINDING_2D: 0x8069,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    CLAMP_TO_EDGE: 0x812f,
    LINEAR: 0x2601,
    NEAREST: 0x2600,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    UNPACK_PREMULTIPLY_ALPHA_WEBGL: 0x9241,
    VIEWPORT: 0x0ba2,
    CURRENT_PROGRAM: 0x8b8d,
    VERTEX_ARRAY_BINDING: 0x85b5,
    COLOR_WRITEMASK: 0x0c23,
    BLEND: 0x0be2,
    DEPTH_TEST: 0x0b71,
    STENCIL_TEST: 0x0b90,
    SCISSOR_TEST: 0x0c11,
    NO_ERROR: 0,
  } as const;

  const gl = {
    ...constants,
    createShader: () => create("createShader", "shader"),
    shaderSource: () => step("shaderSource"),
    compileShader: () => step("compileShader"),
    getShaderParameter: () => true,
    getShaderInfoLog: () => null,
    deleteShader: (value: ResourceRecord | null) => remove("deleteShader", value),
    createProgram: () => create("createProgram", "program"),
    attachShader: () => step("attachShader"),
    linkProgram: () => step("linkProgram"),
    getProgramParameter: () => true,
    getProgramInfoLog: () => null,
    deleteProgram: (value: ResourceRecord | null) => remove("deleteProgram", value),
    createVertexArray: () => create("createVertexArray", "vertex-array"),
    deleteVertexArray: (value: ResourceRecord | null) => remove("deleteVertexArray", value),
    createBuffer: () => create("createBuffer", "buffer"),
    deleteBuffer: (value: ResourceRecord | null) => remove("deleteBuffer", value),
    createFramebuffer: () => create("createFramebuffer", "framebuffer"),
    deleteFramebuffer: (value: ResourceRecord | null) => remove("deleteFramebuffer", value),
    getUniformLocation: () => {
      step("getUniformLocation");
      return { uniform: true };
    },
    bindVertexArray: () => step("bindVertexArray"),
    bindBuffer: () => step("bindBuffer"),
    bufferData: () => step("bufferData"),
    enableVertexAttribArray: () => step("enableVertexAttribArray"),
    vertexAttribPointer: () => step("vertexAttribPointer"),
    getParameter: (parameter: number) => {
      step("getParameter");
      if (parameter === constants.ACTIVE_TEXTURE) return constants.TEXTURE0;
      if (parameter === constants.VIEWPORT) return new Int32Array([0, 0, 4, 4]);
      if (parameter === constants.COLOR_WRITEMASK) return [true, true, true, true];
      if (parameter === constants.UNPACK_PREMULTIPLY_ALPHA_WEBGL) return false;
      return null;
    },
    activeTexture: () => step("activeTexture"),
    isEnabled: () => false,
    bindFramebuffer: () => step("bindFramebuffer"),
    viewport: () => step("viewport"),
    useProgram: () => step("useProgram"),
    bindTexture: () => step("bindTexture"),
    pixelStorei: () => step("pixelStorei"),
    enable: () => step("enable"),
    disable: () => step("disable"),
    colorMask: () => step("colorMask"),
    uniform1i: () => step("uniform1i"),
    framebufferTexture2D: () => step("framebufferTexture2D"),
    checkFramebufferStatus: () => constants.FRAMEBUFFER_COMPLETE,
    createTexture: () => create("createTexture", "texture"),
    deleteTexture: (value: ResourceRecord | null) => remove("deleteTexture", value),
    texParameteri: () => step("texParameteri"),
    texImage2D: () => step("texImage2D"),
    getError: () => constants.NO_ERROR,
  };

  return {
    gl: gl as unknown as WebGL2RenderingContext,
    failAt: (method, call) => {
      failure = { method, call };
      calls.set(method, 0);
    },
    liveResources: (kind) => [...created]
      .filter(([resource]) => !deleted.has(resource) && (!kind || resource.kind === kind))
      .map(([, label]) => label)
      .sort(),
  };
}
