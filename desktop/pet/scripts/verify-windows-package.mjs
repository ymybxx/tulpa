// Component gate for the actual unpacked resources. Does not read a user profile,
// open a device, capture a desktop, start a database, or claim interactive UI QA.
import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform !== "win32" || process.arch !== "x64") throw new Error("Run this gate on Windows x64");
const scripts = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(process.argv[2] ?? "release-win/win-unpacked");
const resources = path.join(root, "resources");
const executable = path.join(root, "tulpa.exe");
const native = path.join(resources, "native", "pet-application-activation.node");
const helper = path.join(resources, "perception", "fish-memory-perception-helper.exe");
const runtime = path.join(resources, "runtime", "fish-memory-runtime", "fish-memory-runtime.exe");
await Promise.all([executable, native, helper, runtime, path.join(resources, "app.asar"),
  path.join(resources, "fish-plugin-tools", "uv.exe"), path.join(resources, "fish-plugin-tools", "python", "python.exe"),
].map(file => access(file)));

const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("FISH_MEMORY_") && key !== "ELECTRON_RUN_AS_NODE"));
Object.assign(env, { PYTHONUTF8: "1", HF_HUB_OFFLINE: "1", FISH_MEMORY_BUNDLED_MODELS_DIR: path.join(resources, "models") });
function run(command, args, environment = env, timeout = 30_000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env: environment, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let output = "", errors = "";
    const timer = setTimeout(() => { child.kill(); reject(new Error(`Package probe timed out: ${path.basename(command)}`)); }, timeout);
    child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
    child.stdout.on("data", value => { output = (output + value).slice(-1_000_000); });
    child.stderr.on("data", value => { errors = (errors + value).slice(-8_192); });
    child.on("error", error => { clearTimeout(timer); reject(error); });
    child.on("close", code => { clearTimeout(timer); code === 0 ? resolve(output) : reject(new Error(`Package probe failed (${code}): ${errors}`)); });
  });
}

const probe = `
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const native = require(${JSON.stringify(native)});
for (const name of ['setNonActivating','startMouseMonitor','startClickMonitor','initializeHostJob','attachProcessToJob','terminateProcessJob']) assert.equal(typeof native[name], 'function');
assert.equal(native.initializeHostJob(), true);
const child = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 30000)'], {windowsHide:true, stdio:'ignore'});
child.on('error', error => { console.error(error); process.exit(1); });
assert.equal(native.attachProcessToJob(child.pid), true);
child.once('exit', () => console.log(JSON.stringify({nativeAbi:true, jobs:true})));
assert.equal(native.terminateProcessJob(child.pid), true);
`;
const nativeReport = await run(executable, ["-e", probe], { ...env, ELECTRON_RUN_AS_NODE: "1" });
assert.equal(JSON.parse(nativeReport.trim().split(/\r?\n/).at(-1)).nativeAbi, true);
await run(process.execPath, [path.join(scripts, "verify-windows-native.mjs"), helper]);
const doctor = JSON.parse((await run(runtime, ["doctor"], env, 120_000)).trim().split(/\r?\n/).at(-1));
assert.equal(doctor.ok, true);
console.log(JSON.stringify({ ok: true, packagedNativeAbi: true, packagedHelperProtocol: true, packagedRuntimeDoctor: true, interactiveWindows11QA: "pending" }));
