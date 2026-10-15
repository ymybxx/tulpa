// Verify prerequisites without rebuilding Rust/Python on each UI development run.
import { access, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform !== "win32" || process.arch !== "x64") {
  throw new Error("dev:win requires Windows x64.");
}
const petRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.join(petRoot, "runtime", "win32-x64");
try {
  await Promise.all([
    "fish-memory-runtime/fish-memory-runtime.exe",
    "fish-memory-runtime/_internal/pg0/bin/pg0.exe",
    "pg0/pg0.exe",
    "fish-plugin-tools/uv.exe",
    "fish-plugin-tools/python/python.exe",
  ].map(relative => access(path.join(root, relative))));
  const tools = await readdir(path.join(root, "fish-plugin-tools"));
  if (!tools.some(file => /^fish_plugin_sdk-.+\.whl$/.test(file))) throw new Error("SDK wheel missing");
} catch {
  throw new Error("Windows development runtime is not prepared. Run npm run prepare:runtime:win, then npm run dev:win. Re-run preparation after Python backend changes.");
}
