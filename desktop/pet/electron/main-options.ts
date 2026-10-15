interface CommandLineSwitches {
  appendSwitch(name: string): void;
}

export type DesktopRendererWindow = "pet" | "main";

export function configureMainProcess(
  commandLine: CommandLineSwitches,
  platform: NodeJS.Platform,
): void {
  if (platform === "darwin") commandLine.appendSwitch("use-mock-keychain");
}

export function shouldThrottleRenderer(window: DesktopRendererWindow): boolean {
  return window === "main";
}
