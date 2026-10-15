import { describe, expect, it } from "vitest";

import {
  CubismRenderFrameStateCache_WebGL,
  CubismRenderTargetIndexBuffer_WebGL,
} from "@cubism/rendering/cubismwebglresourcecache";
import { CubismRenderTarget_WebGL } from "@cubism/rendering/cubismrendertarget_webgl";

describe("CubismRenderFrameStateCache_WebGL", () => {
  it("consumes the application-provided root state without synchronous reads", () => {
    const fake = createFakeGl("first");
    const cache = new CubismRenderFrameStateCache_WebGL();
    const viewport = [4, 8, 640, 480];
    const framebuffer = { id: "root" } as unknown as WebGLFramebuffer;

    cache.set(framebuffer, viewport);
    viewport[2] = 1;

    expect(cache.consume(fake.gl)).toEqual({
      framebuffer,
      viewport: [4, 8, 640, 480],
    });
    expect(fake.calls("getParameter")).toBe(0);
  });

  it("falls back to the bound root state when no state was supplied", () => {
    const fake = createFakeGl("fallback");
    const cache = new CubismRenderFrameStateCache_WebGL();

    expect(cache.consume(fake.gl)).toEqual({
      framebuffer: fake.framebuffer,
      viewport: [0, 0, 320, 240],
    });
    expect(fake.parameters).toEqual([
      fake.gl.VIEWPORT,
      fake.gl.FRAMEBUFFER_BINDING,
    ]);
  });
});

describe("CubismRenderTargetIndexBuffer_WebGL", () => {
  it("uploads once, rebinds for later quad draws, and releases once", () => {
    const fake = createFakeGl("first");
    const cache = new CubismRenderTargetIndexBuffer_WebGL();
    const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);

    cache.reset(fake.gl);
    expect(cache.bind(indices)).toBe(true);
    expect(cache.bind(indices)).toBe(true);
    expect(fake.calls("createBuffer")).toBe(1);
    expect(fake.calls("bufferData")).toBe(1);
    expect(fake.calls("bindBuffer")).toBe(2);

    cache.release();
    cache.release();
    expect(fake.calls("deleteBuffer")).toBe(1);
  });

  it("drops the old context resource before allocating after restore", () => {
    const first = createFakeGl("first");
    const restored = createFakeGl("restored");
    const cache = new CubismRenderTargetIndexBuffer_WebGL();
    const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);

    cache.reset(first.gl);
    expect(cache.bind(indices)).toBe(true);
    cache.reset(restored.gl);
    expect(first.calls("deleteBuffer")).toBe(1);

    expect(cache.bind(indices)).toBe(true);
    expect(restored.calls("createBuffer")).toBe(1);
    cache.release();
    expect(restored.calls("deleteBuffer")).toBe(1);
  });

  it("invalidates the buffer when the same WebGL context object is restored", () => {
    const fake = createFakeGl("restored-in-place");
    const cache = new CubismRenderTargetIndexBuffer_WebGL();
    const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);

    cache.reset(fake.gl);
    expect(cache.bind(indices)).toBe(true);
    cache.reset(fake.gl);
    expect(fake.calls("deleteBuffer")).toBe(1);

    expect(cache.bind(indices)).toBe(true);
    expect(fake.calls("createBuffer")).toBe(2);
    cache.release();
    expect(fake.calls("deleteBuffer")).toBe(2);
  });
});

describe("CubismRenderTarget_WebGL", () => {
  it("restores a known default framebuffer without querying it", () => {
    const fake = createFakeGl("target");
    const target = new CubismRenderTarget_WebGL() as unknown as {
      _gl: WebGLRenderingContext;
      _renderTexture: WebGLFramebuffer;
      beginDrawWithRestoreFbo: (fbo: WebGLFramebuffer | null) => void;
      endDraw: () => void;
    };
    const renderTexture = { id: "offscreen" } as unknown as WebGLFramebuffer;
    target._gl = fake.gl;
    target._renderTexture = renderTexture;

    target.beginDrawWithRestoreFbo(null);
    target.endDraw();

    expect(fake.calls("getParameter")).toBe(0);
    expect(fake.boundFramebuffers).toEqual([renderTexture, null]);
  });
});

function createFakeGl(label: string): {
  gl: WebGLRenderingContext;
  framebuffer: WebGLFramebuffer;
  parameters: number[];
  boundFramebuffers: Array<WebGLFramebuffer | null>;
  calls: (name: string) => number;
} {
  const counts = new Map<string, number>();
  const parameters: number[] = [];
  const boundFramebuffers: Array<WebGLFramebuffer | null> = [];
  const framebuffer = { id: `${label}-framebuffer` } as unknown as WebGLFramebuffer;
  const buffer = { id: `${label}-buffer` } as unknown as WebGLBuffer;
  const step = (name: string): void => {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  };
  const constants = {
    FRAMEBUFFER: 0x8d40,
    FRAMEBUFFER_BINDING: 0x8ca6,
    VIEWPORT: 0x0ba2,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
  } as const;
  const gl = {
    ...constants,
    getParameter: (parameter: number) => {
      step("getParameter");
      parameters.push(parameter);
      return parameter === constants.VIEWPORT
        ? new Int32Array([0, 0, 320, 240])
        : framebuffer;
    },
    createBuffer: () => {
      step("createBuffer");
      return buffer;
    },
    bindBuffer: () => step("bindBuffer"),
    bufferData: () => step("bufferData"),
    deleteBuffer: () => step("deleteBuffer"),
    bindFramebuffer: (_target: number, value: WebGLFramebuffer | null) => {
      step("bindFramebuffer");
      boundFramebuffers.push(value);
    },
  } as unknown as WebGLRenderingContext;
  return {
    gl,
    framebuffer,
    parameters,
    boundFramebuffers,
    calls: (name) => counts.get(name) ?? 0,
  };
}
