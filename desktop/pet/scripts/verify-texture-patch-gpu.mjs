import { app, BrowserWindow } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

app.once("ready", () => void run());

async function run() {
  let window;
  let exitCode = 0;
  try {
    window = new BrowserWindow({
      width: 64,
      height: 64,
      show: false,
      webPreferences: { contextIsolation: true, nodeIntegration: false },
    });
    await window.loadFile(path.join(root, "dist-renderer", "texture-patch-harness.html"));
    const result = await waitForResult(window);
    console.log(JSON.stringify(result));
    if (!result.ok) exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.stack : String(error));
    exitCode = 1;
  } finally {
    window?.destroy();
    app.exit(exitCode);
  }
}

async function waitForResult(window) {
  const started = Date.now();
  while (Date.now() - started < 15_000) {
    const state = await window.webContents.executeJavaScript(
      "({ result: window.texturePatchResult || null, error: window.texturePatchError || null })",
    );
    if (state.error) throw new Error(state.error);
    if (state.result) return state.result;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error("texture patch GPU harness timed out");
}
