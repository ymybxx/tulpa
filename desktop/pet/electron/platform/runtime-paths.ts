import path from "node:path";

export interface RuntimeLocations {
  packaged: boolean;
  resourcesPath: string;
  projectRoot: string;
  platform?: NodeJS.Platform;
  arch?: string;
}

export function runtimePlatformDirectory(platform: NodeJS.Platform = process.platform, arch: string = process.arch): string {
  return `${platform === "darwin" ? "mac" : platform}-${arch}`;
}

export function runtimeAssetsRoot(options: RuntimeLocations): string {
  return options.packaged ? options.resourcesPath : path.join(options.projectRoot, "desktop", "pet", "runtime",
    runtimePlatformDirectory(options.platform, options.arch));
}

export function packagedRuntimeExecutable(resourcesPath: string, platform: NodeJS.Platform = process.platform): string {
  return path.join(resourcesPath, "runtime", "fish-memory-runtime", `fish-memory-runtime${platform === "win32" ? ".exe" : ""}`);
}

/** Windows development shares the verified backend while Vite serves the UI. */
export function frozenRuntimeExecutable(options: RuntimeLocations): string | null {
  const platform = options.platform ?? process.platform;
  if (options.packaged) return packagedRuntimeExecutable(options.resourcesPath, platform);
  return platform === "win32"
    ? path.join(runtimeAssetsRoot(options), "fish-memory-runtime", "fish-memory-runtime.exe")
    : null;
}

export function platformPerceptionExecutable(options: RuntimeLocations): string {
  const root = runtimeAssetsRoot(options);
  return (options.platform ?? process.platform) === "win32"
    ? path.join(root, ...(options.packaged ? ["perception"] : []), "fish-memory-perception-helper.exe")
    : path.join(root, ...(options.packaged ? ["perception"] : []), "FishMemoryPerceptionHelper.app", "Contents", "MacOS", "fish-memory-perception-helper");
}
