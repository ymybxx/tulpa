import { describe, expect, it, vi } from "vitest";
import { CubismOffscreenRenderTarget_WebGL } from "@cubism/rendering/cubismoffscreenrendertarget_webgl";
import { CubismWebGLOffscreenManager } from "@cubism/rendering/cubismoffscreenmanager";
import { finishCubismOffscreenFrame } from "./cubism-frame-lifecycle";

describe("Cubism shared offscreen render target lifecycle", () => {
  it("detaches a retiring model without deleting context-owned pool resources", () => {
    const target = new CubismOffscreenRenderTarget_WebGL();
    const colorBuffer = {} as WebGLTexture;
    const framebuffer = {} as WebGLFramebuffer;
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    const stopUsingRenderTexture = vi.fn();
    const gl = { deleteTexture, deleteFramebuffer } as unknown as WebGL2RenderingContext;
    const manager = { stopUsingRenderTexture };
    const internal = target as unknown as {
      _gl: WebGL2RenderingContext;
      _colorBuffer: WebGLTexture | null;
      _renderTexture: WebGLFramebuffer | null;
      _bufferWidth: number;
      _bufferHeight: number;
      _webGLOffscreenManager: typeof manager;
    };
    internal._gl = gl;
    internal._colorBuffer = colorBuffer;
    internal._renderTexture = framebuffer;
    internal._bufferWidth = 1920;
    internal._bufferHeight = 1080;
    internal._webGLOffscreenManager = manager;

    target.destroyRenderTarget();

    expect(stopUsingRenderTexture).toHaveBeenCalledOnce();
    expect(stopUsingRenderTexture).toHaveBeenCalledWith(gl, framebuffer);
    expect(deleteTexture).not.toHaveBeenCalled();
    expect(deleteFramebuffer).not.toHaveBeenCalled();
    expect(target.getColorBuffer()).toBeNull();
    expect(target.getRenderTexture()).toBeNull();
    expect(target.getBufferWidth()).toBe(0);
    expect(target.getBufferHeight()).toBe(0);
  });

  it("shrinks the shared pool only after the frame has ended", () => {
    const order: string[] = [];
    const manager = CubismWebGLOffscreenManager.getInstance();
    const begin = vi.spyOn(manager, "beginFrameProcess").mockImplementation(() => order.push("begin"));
    const stop = vi.spyOn(manager, "stopUsingAllRenderTextures").mockImplementation(() => order.push("stop"));
    const end = vi.spyOn(manager, "endFrameProcess").mockImplementation(() => order.push("end"));
    const release = vi.spyOn(manager, "releaseStaleRenderTextures").mockImplementation(() => order.push("release"));
    const gl = { bindFramebuffer: vi.fn(), FRAMEBUFFER: 1 } as unknown as WebGL2RenderingContext;
    manager.beginFrameProcess(gl);
    finishCubismOffscreenFrame(gl, vi.fn());

    expect(order).toEqual(["begin", "stop", "end", "release"]);
    begin.mockRestore();
    stop.mockRestore();
    end.mockRestore();
    release.mockRestore();
  });
});
