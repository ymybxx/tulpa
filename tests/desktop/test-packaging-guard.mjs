import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import {
  assertPackagingIdle,
  findRunningBuildTargets,
} from "../../desktop/pet/scripts/assert-packaging-idle.mjs";

const root = "/workspace/Fish Memory/desktop/pet";
const app = `${root}/release/mac-arm64/Fish Memory.app/Contents/MacOS/Fish Memory`;
const runtime = `${root}/runtime/mac-arm64/fish-memory-runtime/fish-memory-runtime`;

test("finds the app and runtime, preserving spaces in executable paths", () => {
  const output = `  123 ${app}\n  456 ${runtime}\n  789 /Applications/Other.app/Contents/MacOS/Other\n`;
  assert.deepEqual(findRunningBuildTargets(output, root, "darwin"), [
    { pid: 123, executable: app },
    { pid: 456, executable: runtime },
  ]);
  assert.throws(() => assertPackagingIdle(root, () => output, "darwin"), /PID 123/);
});

test("a surviving packaged backend also blocks overwriting the release", () => {
  const backend = `${root}/release/mac-arm64/Fish Memory.app/Contents/Resources/runtime/fish-memory-runtime/fish-memory-runtime`;
  assert.throws(() => assertPackagingIdle(root, () => `42 ${backend}`, "darwin"), /PID 42/);
});

test("other checkouts and similarly prefixed directories are not blocked", () => {
  const output = [
    `1 ${root}/release-backup/Fish Memory`,
    `2 ${root}/runtime/mac-arm64-old/fish-memory-runtime`,
    "3 /other/desktop/pet/release/Fish Memory",
    "4 /bin/ps",
  ].join("\n");
  assert.deepEqual(findRunningBuildTargets(output, root, "darwin"), []);
  assert.doesNotThrow(() => assertPackagingIdle(root, () => output, "darwin"));
});

test("failure to inspect processes stops the build", () => {
  assert.throws(() => assertPackagingIdle(root, () => {
    throw new Error("permission denied");
  }), /无法检查正在运行的进程/);
});

test("Windows runtime and installer output are guarded, with case-insensitive paths", () => {
  const root = "C:\\Users\\A User\\fish-memory\\desktop\\pet";
  const executable = `${root.toUpperCase()}\\runtime\\win32-x64\\fish-memory-runtime\\fish-memory-runtime.exe`;
  const output = `42 ${executable}\n43 ${root}\\release-win-old\\tulpa.exe`;
  assert.deepEqual(findRunningBuildTargets(output, root, "win32"), [{ pid: 42, executable }]);
  assert.throws(() => assertPackagingIdle(root, () => output, "win32"), /PID 42/);
});

test("both package entry and replacement stages have guards", () => {
  const pet = new URL("../../desktop/pet/", import.meta.url);
  const config = JSON.parse(readFileSync(new URL("package.json", pet), "utf8"));
  assert.equal(config.scripts["predist:mac:local"], "node scripts/assert-packaging-idle.mjs");
  assert.match(config.scripts["dist:mac:local"], /assert-packaging-idle\.mjs && rimraf release dist/);
  const runtimeBuild = readFileSync(new URL("scripts/build-runtime-mac.sh", pet), "utf8");
  assert.match(runtimeBuild, /assert-packaging-idle\.mjs"\s*\nrm -rf/);
  assert.ok(runtimeBuild.indexOf("assert-packaging-idle.mjs") < runtimeBuild.indexOf("build-onnxruntime-compat"));
  assert.ok(fileURLToPath(pet).endsWith(path.join("desktop", "pet") + path.sep));
});
