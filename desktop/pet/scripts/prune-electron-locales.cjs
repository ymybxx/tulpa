// electron-builder afterPack hook. On macOS `electronLanguages` only prunes the
// app's own Resources, while Chromium's ~55 locale folders live inside Electron
// Framework. Keep the languages the app ships in; Chromium falls back to en.
const fs = require("node:fs");
const path = require("node:path");

const KEEP = new Set(["en", "zh_CN", "zh_TW"]);

exports.default = async function pruneElectronLocales(context) {
  if (context.electronPlatformName !== "darwin") return;
  const app = `${context.packager.appInfo.productFilename}.app`;
  const resources = path.join(
    context.appOutDir,
    app,
    "Contents/Frameworks/Electron Framework.framework/Versions/A/Resources",
  );
  const removed = [];
  for (const entry of fs.readdirSync(resources)) {
    if (!entry.endsWith(".lproj") || KEEP.has(entry.slice(0, -".lproj".length))) continue;
    fs.rmSync(path.join(resources, entry), { recursive: true, force: true });
    removed.push(entry);
  }
  if (!fs.existsSync(path.join(resources, "en.lproj"))) throw new Error("Electron en.lproj is missing");
  // macOS picks the language of system panels (open file, alerts) from the
  // app's own .lproj folders; without them a Chinese Mac shows them in English.
  const own = path.join(context.appOutDir, app, "Contents/Resources");
  for (const language of ["zh_CN", "en"]) fs.mkdirSync(path.join(own, `${language}.lproj`), { recursive: true });
  console.log(`  • pruned ${removed.length} Electron locales, kept ${[...KEEP].join(", ")}`);
};
