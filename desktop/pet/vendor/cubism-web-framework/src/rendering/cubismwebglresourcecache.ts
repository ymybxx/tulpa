/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

/**
 * Root render state supplied by the application for one model draw.
 */
export interface CubismRenderFrameState_WebGL {
  framebuffer: WebGLFramebuffer;
  viewport: number[];
}

/**
 * Keeps the render state passed to setRenderState until the next model draw.
 *
 * The application already knows this state when it binds the model target, so
 * reading it back synchronously with getParameter would only stall WebGL. The
 * cached value is consumed once; integrations that do not call setRenderState
 * for a frame retain the original getParameter fallback.
 */
export class CubismRenderFrameStateCache_WebGL {
  public set(framebuffer: WebGLFramebuffer, viewport: number[]): void {
    this._framebuffer = framebuffer;
    this._viewport[0] = viewport[0];
    this._viewport[1] = viewport[1];
    this._viewport[2] = viewport[2];
    this._viewport[3] = viewport[3];
    this._hasPendingState = true;
  }

  public consume(
    gl: WebGLRenderingContext | WebGL2RenderingContext
  ): CubismRenderFrameState_WebGL {
    if (this._hasPendingState) {
      this._hasPendingState = false;
      const framebuffer = this._framebuffer;
      this._framebuffer = null;
      return {
        framebuffer: framebuffer,
        viewport: this._viewport.slice()
      };
    }

    const viewport = gl.getParameter(gl.VIEWPORT) as ArrayLike<number>;
    return {
      framebuffer: gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer,
      viewport: [viewport[0], viewport[1], viewport[2], viewport[3]]
    };
  }

  public clear(): void {
    this._framebuffer = null;
    this._hasPendingState = false;
  }

  public constructor() {
    this._framebuffer = null;
    this._viewport = [0, 0, 0, 0];
    this._hasPendingState = false;
  }

  private _framebuffer: WebGLFramebuffer;
  private _viewport: number[];
  private _hasPendingState: boolean;
}

/**
 * Owns the immutable element buffer shared by render-target quad draws.
 */
export class CubismRenderTargetIndexBuffer_WebGL {
  public reset(gl: WebGLRenderingContext | WebGL2RenderingContext): void {
    this.release();
    this._gl = gl;
  }

  public bind(indices: Uint16Array): boolean {
    if (this._gl == null) {
      return false;
    }

    if (this._buffer == null) {
      const buffer = this._gl.createBuffer();
      if (buffer == null) {
        return false;
      }
      this._buffer = buffer;
      this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER, buffer);
      this._gl.bufferData(
        this._gl.ELEMENT_ARRAY_BUFFER,
        indices,
        this._gl.STATIC_DRAW
      );
      return true;
    }

    this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER, this._buffer);
    return true;
  }

  public release(): void {
    if (this._gl != null && this._buffer != null) {
      this._gl.deleteBuffer(this._buffer);
    }
    this._buffer = null;
    this._gl = null;
  }

  public constructor() {
    this._gl = null;
    this._buffer = null;
  }

  private _gl: WebGLRenderingContext | WebGL2RenderingContext;
  private _buffer: WebGLBuffer;
}
