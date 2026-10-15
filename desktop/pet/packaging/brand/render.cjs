// Renders SVG files to transparent PNGs with Electron's own Chromium.
// Usage: electron render.cjs jobs.json   where jobs is [{ "svg": path, "png": path, "width": n, "height": n }]
const { app, BrowserWindow } = require("electron");
const fs = require("fs");
const os = require("os");
const path = require("path");

app.disableHardwareAcceleration();
app.whenReady().then(async () => {
  const jobs = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], "utf8"));
  // One offscreen window for every job; a second window in the same run fails to load.
  const win = new BrowserWindow({
    width: 64, height: 64, show: false, transparent: true, frame: false,
    useContentSize: true, enableLargerThanScreen: true, webPreferences: { offscreen: true },
  });
  try {
    for (const [index, job] of jobs.entries()) {
      win.setContentSize(job.width, job.height);
      const svg = fs.readFileSync(job.svg, "utf8");
      const page = path.join(os.tmpdir(), `tulpa-render-${process.pid}-${index}.html`);
      fs.writeFileSync(page, `<html><body style="margin:0;background:transparent;overflow:hidden">`
        + `<img style="display:block;width:${job.width}px;height:${job.height}px" `
        + `src="data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}"></body></html>`);
      await win.loadFile(page);
      await new Promise((resolve) => setTimeout(resolve, 200));
      const image = await win.webContents.capturePage({ x: 0, y: 0, width: job.width, height: job.height });
      fs.mkdirSync(path.dirname(job.png), { recursive: true });
      fs.writeFileSync(job.png, image.resize({ width: job.width, height: job.height, quality: "best" }).toPNG());
      fs.rmSync(page, { force: true });
    }
  } finally {
    win.destroy();
    app.quit();
  }
});
