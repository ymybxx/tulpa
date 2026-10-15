import { release } from "node:os";

/** The Darwin major version: 22 is macOS 13, 24 is macOS 15. */
export function darwinMajor(kernelRelease: string = release()): number {
  const major = Number(kernelRelease.split(".")[0]);
  return Number.isInteger(major) ? major : 0;
}

/** Shipping targets: macOS 13+ and Windows 11 (build 22000+). */
export function supportsSystemAudio(
  platform: string = process.platform,
  kernelRelease: string = release(),
): boolean {
  const major = Number(kernelRelease.split(".")[0]);
  if (platform === "win32") {
    const build = Number(kernelRelease.split(".")[2]);
    return major >= 10 && Number.isInteger(build) && build >= 22000;
  }
  return platform === "darwin" && Number.isInteger(major) && major >= 22;
}
