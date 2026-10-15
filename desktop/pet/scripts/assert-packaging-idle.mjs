import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function findRunningBuildTargets(processList, petRoot, platform = process.platform) {
  const paths = platform === "win32" ? path.win32 : path.posix;
  const normalize = (value) => platform === "win32" ? value.toLowerCase() : value;
  const roots = ["runtime/mac-arm64", "runtime/win32-x64", "release", "release-win"].map((name) =>
    normalize(paths.resolve(petRoot, name) + paths.sep),
  );
  return processList.split("\n").flatMap((line) => {
    const match = /^\s*(\d+)\s+(.+?)\s*$/.exec(line);
    if (!match || !roots.some((root) => normalize(match[2]).startsWith(root))) return [];
    return [{ pid: Number(match[1]), executable: match[2] }];
  });
}

export function assertPackagingIdle(
  petRoot,
  readProcesses = readSystemProcesses,
  platform = process.platform,
) {
  let processList;
  try {
    processList = readProcesses();
  } catch {
    throw new Error("无法检查正在运行的进程，已停止打包，避免覆盖运行中的程序。");
  }
  const running = findRunningBuildTargets(processList, petRoot, platform);
  if (running.length === 0) return;
  throw new Error([
    "已停止打包：以下进程仍在使用即将重建的应用或运行时：",
    ...running.map(({ pid, executable }) => `  PID ${pid}: ${executable}`),
    "请先退出 tulpa（及使用该运行时的开发进程），再重新打包。",
    "运行中覆盖 PyInstaller 文件会导致后续模块解压失败。",
  ].join("\n"));
}

function readSystemProcesses() {
  if (process.platform !== "win32") {
    return execFileSync("/bin/ps", ["-axo", "pid=,comm="], { encoding: "utf8" });
  }
  const output = execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"],
  { encoding: "utf8", windowsHide: true, maxBuffer: 8 * 1024 * 1024 });
  const decoded = JSON.parse(output.trim().replace(/^\uFEFF/, "") || "[]");
  return (Array.isArray(decoded) ? decoded : [decoded])
    .filter((item) => Number.isInteger(item?.ProcessId) && typeof item.ExecutablePath === "string")
    .map((item) => `${item.ProcessId} ${item.ExecutablePath}`).join("\n");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    assertPackagingIdle(path.resolve(fileURLToPath(new URL("..", import.meta.url))));
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
