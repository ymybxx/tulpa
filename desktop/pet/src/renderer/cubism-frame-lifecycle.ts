import { CubismWebGLOffscreenManager } from "@cubism/rendering/cubismoffscreenmanager";

export function finishCubismOffscreenFrame(
  gl: WebGL2RenderingContext,
  unbindRendererTextures: () => void,
): void {
  const offscreen = CubismWebGLOffscreenManager.getInstance();
  unbindRendererTextures();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  offscreen.stopUsingAllRenderTextures(gl);
  offscreen.endFrameProcess(gl);
  offscreen.releaseStaleRenderTextures(gl);
}
