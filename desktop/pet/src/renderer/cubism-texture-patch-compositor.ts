import type { TextureAtlasPatchSet, TexturePatchUpdate } from "../shared/contracts";
import {
  texturePatchDestination,
  texturePatchFramebufferRect,
} from "../shared/texture-patch";
import { allOrRelease } from "./all-or-release";

export interface TexturePatchCandidateResource {
  texture: WebGLTexture;
  width: number;
  height: number;
}

interface DecodedPatch {
  value: TexturePatchUpdate;
  image: ImageBitmap;
  mask: ImageBitmap;
}

interface GlState {
  readFramebuffer: WebGLFramebuffer | null;
  drawFramebuffer: WebGLFramebuffer | null;
  viewport: Int32Array;
  program: WebGLProgram | null;
  vertexArray: WebGLVertexArrayObject | null;
  arrayBuffer: WebGLBuffer | null;
  activeTexture: number;
  textureBindings: Array<WebGLTexture | null>;
  unpackPremultiplyAlpha: boolean;
  blend: boolean;
  depthTest: boolean;
  stencilTest: boolean;
  scissorTest: boolean;
  colorMask: boolean[];
}

interface CompositorResources {
  program: WebGLProgram;
  vertexArray: WebGLVertexArrayObject;
  vertexBuffer: WebGLBuffer;
  candidateFramebuffer: WebGLFramebuffer;
  scratchFramebuffer: WebGLFramebuffer;
  currentLocation: WebGLUniformLocation;
  patchLocation: WebGLUniformLocation;
  maskLocation: WebGLUniformLocation;
}

const VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
  v_uv = a_uv;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D u_current;
uniform sampler2D u_patch;
uniform sampler2D u_mask;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec4 current = texture(u_current, v_uv);
  vec4 replacement = texture(u_patch, v_uv);
  float mask_alpha = texture(u_mask, v_uv).a;
  if (mask_alpha <= 0.0) {
    out_color = current;
    return;
  }
  float alpha = min(current.a, replacement.a);
  out_color = vec4(replacement.rgb * alpha, alpha);
}`;

export class StaleTexturePatchError extends Error {
  constructor() {
    super("纹理 patch 已被更新的 revision 取代");
  }
}

export class CubismTexturePatchCompositor {
  private readonly program: WebGLProgram;
  private readonly vertexArray: WebGLVertexArrayObject;
  private readonly vertexBuffer: WebGLBuffer;
  private readonly candidateFramebuffer: WebGLFramebuffer;
  private readonly scratchFramebuffer: WebGLFramebuffer;
  private readonly currentLocation: WebGLUniformLocation;
  private readonly patchLocation: WebGLUniformLocation;
  private readonly maskLocation: WebGLUniformLocation;
  private released = false;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const resources = createCompositorResources(gl);
    this.program = resources.program;
    this.vertexArray = resources.vertexArray;
    this.vertexBuffer = resources.vertexBuffer;
    this.candidateFramebuffer = resources.candidateFramebuffer;
    this.scratchFramebuffer = resources.scratchFramebuffer;
    this.currentLocation = resources.currentLocation;
    this.patchLocation = resources.patchLocation;
    this.maskLocation = resources.maskLocation;
  }

  async apply(
    candidate: TexturePatchCandidateResource,
    patchSet: TextureAtlasPatchSet,
    isCurrent: () => boolean,
  ): Promise<void> {
    if (this.released) throw new Error("纹理 patch compositor 已释放");
    const decoded = await allOrRelease(
      patchSet.patches.map((value) => decodePatch(value)),
      closeDecodedPatch,
    );
    const state = saveState(this.gl);
    try {
      if (!isCurrent()) throw new StaleTexturePatchError();
      this.prepareDrawState();
      this.attachCandidate(candidate.texture);
      for (const patch of decoded) {
        if (!isCurrent()) throw new StaleTexturePatchError();
        this.applyPatch(candidate, patchSet, patch);
      }
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
      this.gl.bindTexture(this.gl.TEXTURE_2D, candidate.texture);
      this.gl.generateMipmap(this.gl.TEXTURE_2D);
      this.gl.bindTexture(this.gl.TEXTURE_2D, null);
      assertGraphicsHealthy(this.gl);
      if (!isCurrent()) throw new StaleTexturePatchError();
    } finally {
      this.detachFramebuffers();
      restoreState(this.gl, state);
      for (const patch of decoded) closeDecodedPatch(patch);
    }
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    const gl = this.gl;
    gl.deleteFramebuffer(this.candidateFramebuffer);
    gl.deleteFramebuffer(this.scratchFramebuffer);
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vertexArray);
    gl.deleteProgram(this.program);
  }

  private applyPatch(
    candidate: TexturePatchCandidateResource,
    patchSet: TextureAtlasPatchSet,
    patch: DecodedPatch,
  ): void {
    const gl = this.gl;
    const destination = texturePatchDestination(
      patch.value.bounds,
      { width: patchSet.sourceWidth, height: patchSet.sourceHeight },
      { width: candidate.width, height: candidate.height },
    );
    const rect = texturePatchFramebufferRect(destination, candidate.height);
    let scratch: WebGLTexture | null = null;
    let patchTexture: WebGLTexture | null = null;
    let maskTexture: WebGLTexture | null = null;
    try {
      scratch = createEmptyTexture(gl, rect.width, rect.height, gl.LINEAR);
      patchTexture = createBitmapTexture(gl, patch.image, gl.LINEAR);
      maskTexture = createBitmapTexture(gl, patch.mask, gl.NEAREST);

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.candidateFramebuffer);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.scratchFramebuffer);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, scratch, 0);
      assertFramebufferComplete(gl, gl.DRAW_FRAMEBUFFER);
      gl.blitFramebuffer(
        rect.x,
        rect.y,
        rect.x + rect.width,
        rect.y + rect.height,
        0,
        0,
        rect.width,
        rect.height,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST,
      );

      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.candidateFramebuffer);
      gl.viewport(rect.x, rect.y, rect.width, rect.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, scratch);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, patchTexture);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, maskTexture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      assertGraphicsHealthy(gl);
    } finally {
      cleanupIgnoringErrors(() => {
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.scratchFramebuffer);
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      });
      if (scratch) cleanupIgnoringErrors(() => gl.deleteTexture(scratch));
      if (patchTexture) cleanupIgnoringErrors(() => gl.deleteTexture(patchTexture));
      if (maskTexture) cleanupIgnoringErrors(() => gl.deleteTexture(maskTexture));
    }
  }

  private prepareDrawState(): void {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vertexArray);
    gl.uniform1i(this.currentLocation, 0);
    gl.uniform1i(this.patchLocation, 1);
    gl.uniform1i(this.maskLocation, 2);
  }

  private attachCandidate(texture: WebGLTexture): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.candidateFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    assertFramebufferComplete(gl, gl.FRAMEBUFFER);
  }

  private detachFramebuffers(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.candidateFramebuffer);
    gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.scratchFramebuffer);
    gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  }
}

function createCompositorResources(gl: WebGL2RenderingContext): CompositorResources {
  let program: WebGLProgram | null = null;
  let vertexArray: WebGLVertexArrayObject | null = null;
  let vertexBuffer: WebGLBuffer | null = null;
  let candidateFramebuffer: WebGLFramebuffer | null = null;
  let scratchFramebuffer: WebGLFramebuffer | null = null;
  try {
    program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    vertexArray = requiredResource(gl.createVertexArray(), "创建 patch VAO 失败");
    vertexBuffer = requiredResource(gl.createBuffer(), "创建 patch VBO 失败");
    candidateFramebuffer = requiredResource(gl.createFramebuffer(), "创建 patch framebuffer 失败");
    scratchFramebuffer = requiredResource(gl.createFramebuffer(), "创建 patch scratch framebuffer 失败");
    const currentLocation = requiredUniform(gl, program, "u_current");
    const patchLocation = requiredUniform(gl, program, "u_patch");
    const maskLocation = requiredUniform(gl, program, "u_mask");
    const state = saveState(gl);
    try {
      gl.bindVertexArray(vertexArray);
      gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1, 0, 0,
        1, -1, 1, 0,
        -1, 1, 0, 1,
        1, 1, 1, 1,
      ]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    } finally {
      restoreState(gl, state);
    }
    return {
      program,
      vertexArray,
      vertexBuffer,
      candidateFramebuffer,
      scratchFramebuffer,
      currentLocation,
      patchLocation,
      maskLocation,
    };
  } catch (error) {
    if (scratchFramebuffer) {
      cleanupIgnoringErrors(() => gl.deleteFramebuffer(scratchFramebuffer));
    }
    if (candidateFramebuffer) {
      cleanupIgnoringErrors(() => gl.deleteFramebuffer(candidateFramebuffer));
    }
    if (vertexBuffer) cleanupIgnoringErrors(() => gl.deleteBuffer(vertexBuffer));
    if (vertexArray) cleanupIgnoringErrors(() => gl.deleteVertexArray(vertexArray));
    if (program) cleanupIgnoringErrors(() => gl.deleteProgram(program));
    throw error;
  }
}

async function decodePatch(value: TexturePatchUpdate): Promise<DecodedPatch> {
  const bitmaps = await allOrRelease(
    [loadBitmap(value.imageUrl), loadBitmap(value.maskUrl)],
    (bitmap) => bitmap.close(),
  );
  const [image, mask] = bitmaps;
  if (
    image!.width !== value.bounds.width || image!.height !== value.bounds.height ||
    mask!.width !== value.bounds.width || mask!.height !== value.bounds.height
  ) {
    image!.close();
    mask!.close();
    throw new Error(`纹理单元 ${value.textureUnitId} 的 patch 或蒙版尺寸不一致`);
  }
  return { value, image: image!, mask: mask! };
}

function closeDecodedPatch(value: DecodedPatch): void {
  value.image.close();
  value.mask.close();
}

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`读取纹理 patch 失败 (${response.status}): ${url}`);
  return createImageBitmap(await response.blob(), {
    premultiplyAlpha: "none",
    colorSpaceConversion: "default",
  });
}

function createBitmapTexture(
  gl: WebGL2RenderingContext,
  bitmap: ImageBitmap,
  filter: number,
): WebGLTexture {
  const texture = requiredResource(gl.createTexture(), "创建 patch 纹理失败");
  try {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    return texture;
  } catch (error) {
    cleanupIgnoringErrors(() => gl.deleteTexture(texture));
    throw error;
  }
}

function createEmptyTexture(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  filter: number,
): WebGLTexture {
  const texture = requiredResource(gl.createTexture(), "创建 patch scratch 纹理失败");
  try {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    return texture;
  } catch (error) {
    cleanupIgnoringErrors(() => gl.deleteTexture(texture));
    throw error;
  }
}

function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string,
): WebGLProgram {
  let vertex: WebGLShader | null = null;
  let fragment: WebGLShader | null = null;
  let program: WebGLProgram | null = null;
  try {
    vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    program = requiredResource(gl.createProgram(), "创建 patch shader program 失败");
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`链接 patch shader 失败：${gl.getProgramInfoLog(program) ?? "unknown"}`);
    }
    return program;
  } catch (error) {
    if (program) cleanupIgnoringErrors(() => gl.deleteProgram(program));
    throw error;
  } finally {
    if (vertex) cleanupIgnoringErrors(() => gl.deleteShader(vertex));
    if (fragment) cleanupIgnoringErrors(() => gl.deleteShader(fragment));
  }
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = requiredResource(gl.createShader(type), "创建 patch shader 失败");
  try {
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader) ?? "unknown";
      throw new Error(`编译 patch shader 失败：${message}`);
    }
    return shader;
  } catch (error) {
    cleanupIgnoringErrors(() => gl.deleteShader(shader));
    throw error;
  }
}

function requiredUniform(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation {
  return requiredResource(gl.getUniformLocation(program, name), `缺少 patch shader uniform ${name}`);
}

function requiredResource<T>(value: T | null, message: string): T {
  if (!value) throw new Error(message);
  return value;
}

function assertFramebufferComplete(gl: WebGL2RenderingContext, target: number): void {
  const status = gl.checkFramebufferStatus(target);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`纹理 patch framebuffer 不完整：0x${status.toString(16)}`);
  }
}

function assertGraphicsHealthy(gl: WebGL2RenderingContext): void {
  const error = gl.getError();
  if (error !== gl.NO_ERROR) throw new Error(`纹理 patch WebGL 错误：0x${error.toString(16)}`);
}

function saveState(gl: WebGL2RenderingContext): GlState {
  const activeTexture = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
  const textureBindings: Array<WebGLTexture | null> = [];
  for (let unit = 0; unit < 3; unit += 1) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    textureBindings.push(gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null);
  }
  gl.activeTexture(activeTexture);
  return {
    readFramebuffer: gl.getParameter(gl.READ_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null,
    drawFramebuffer: gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null,
    viewport: gl.getParameter(gl.VIEWPORT) as Int32Array,
    program: gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null,
    vertexArray: gl.getParameter(gl.VERTEX_ARRAY_BINDING) as WebGLVertexArrayObject | null,
    arrayBuffer: gl.getParameter(gl.ARRAY_BUFFER_BINDING) as WebGLBuffer | null,
    activeTexture,
    textureBindings,
    unpackPremultiplyAlpha: Boolean(gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL)),
    blend: gl.isEnabled(gl.BLEND),
    depthTest: gl.isEnabled(gl.DEPTH_TEST),
    stencilTest: gl.isEnabled(gl.STENCIL_TEST),
    scissorTest: gl.isEnabled(gl.SCISSOR_TEST),
    colorMask: [...(gl.getParameter(gl.COLOR_WRITEMASK) as boolean[])],
  };
}

function restoreState(gl: WebGL2RenderingContext, state: GlState): void {
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, state.readFramebuffer);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, state.drawFramebuffer);
  gl.viewport(state.viewport[0]!, state.viewport[1]!, state.viewport[2]!, state.viewport[3]!);
  gl.useProgram(state.program);
  gl.bindVertexArray(state.vertexArray);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.arrayBuffer);
  for (let unit = 0; unit < 3; unit += 1) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, state.textureBindings[unit] ?? null);
  }
  gl.activeTexture(state.activeTexture);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, state.unpackPremultiplyAlpha ? 1 : 0);
  toggle(gl, gl.BLEND, state.blend);
  toggle(gl, gl.DEPTH_TEST, state.depthTest);
  toggle(gl, gl.STENCIL_TEST, state.stencilTest);
  toggle(gl, gl.SCISSOR_TEST, state.scissorTest);
  gl.colorMask(
    state.colorMask[0]!,
    state.colorMask[1]!,
    state.colorMask[2]!,
    state.colorMask[3]!,
  );
}

function toggle(gl: WebGL2RenderingContext, capability: number, enabled: boolean): void {
  if (enabled) gl.enable(capability); else gl.disable(capability);
}

function cleanupIgnoringErrors(cleanup: () => void): void {
  try {
    cleanup();
  } catch {
    // Keep the allocation or draw error that triggered cleanup.
  }
}
