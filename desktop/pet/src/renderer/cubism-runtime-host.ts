import {
  CubismFramework,
  LogLevel,
  Option,
} from "@cubism/live2dcubismframework";

const REQUIRED_MOC_VERSION = 6;

/**
 * Process-local owner of the official Cubism Framework runtime.
 *
 * Electron can host several independent renderer processes (desktop pet and
 * model manager), so each process initializes Framework/Core exactly once.
 * Individual canvases own only their WebGL resources and never dispose the
 * shared Framework while another preview might still be alive.
 */
export class CubismRuntimeHost {
  private static initialized = false;

  static ensureInitialized(): void {
    if (this.initialized) return;
    const core = (globalThis as typeof globalThis & {
      Live2DCubismCore?: { Version: { csmGetLatestMocVersion(): number } };
    }).Live2DCubismCore;
    if (!core) throw new Error("缺少 Live2D Cubism Core 运行资源");
    const mocVersion = core.Version.csmGetLatestMocVersion();
    if (mocVersion < REQUIRED_MOC_VERSION) {
      throw new Error(
        `Live2D Cubism Core 版本过旧（moc3 ${mocVersion}），需要支持 moc3 version ${REQUIRED_MOC_VERSION}`,
      );
    }

    const option = new Option();
    option.loggingLevel = LogLevel.LogLevel_Warning;
    option.logFunction = (message: string) => console.warn(`[Cubism] ${message}`);
    if (!CubismFramework.startUp(option)) {
      throw new Error("Live2D Cubism Framework 启动失败");
    }
    CubismFramework.initialize();
    this.initialized = true;
  }

  static createContext(
    canvas: HTMLCanvasElement,
    preserveDrawingBuffer: boolean,
  ): WebGL2RenderingContext {
    this.ensureInitialized();
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: true,
      depth: false,
      stencil: true,
      premultipliedAlpha: true,
      preserveDrawingBuffer,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("当前环境不支持 Cubism 5.3 所需的 WebGL2");
    return gl;
  }
}
