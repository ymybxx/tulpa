#!/usr/bin/env node
// Small CDP tool for driving the dev build started by start-dev-app.sh.
// See docs/desktop-dev-testing.md.
//
//   node scripts/dev/cdp.mjs pages
//   node scripts/dev/cdp.mjs eval   <page> '<js expression>'
//   node scripts/dev/cdp.mjs main   '<js expression>'          (main process; require() works)
//   node scripts/dev/cdp.mjs shot   <page> <out.png>
//   node scripts/dev/cdp.mjs click  <page> <x> <y> [right]
//   node scripts/dev/cdp.mjs drag   <page> <x> <y> <dx> <dy> [steps]
//   node scripts/dev/cdp.mjs wheel  <page> <x> <y> <deltaY> [count] [gapMs]
//   node scripts/dev/cdp.mjs frames <page> '<js number expression>' <ms>
//
// <page> is the end of the page URL: main.html, index.html (TA), pet-chat.html (HUD).
// Coordinates are the page's CSS pixels. Input goes through Chromium, so
// page handlers run as for a real mouse; macOS-level behaviour (Spaces,
// activation, window order) still needs a real mouse.
const cdpPort = Number(process.env.TULPA_CDP_PORT ?? 9230);
const inspectPort = Number(process.env.TULPA_INSPECT_PORT ?? 9231);
const [command, ...args] = process.argv.slice(2);

async function connect(url) {
  const ws = new WebSocket(url);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = async (event) => {
    const message = JSON.parse(typeof event.data === "string" ? event.data : await event.data.text());
    if (pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id); }
  };
  const call = (method, params = {}) => new Promise((resolve) => {
    const next = ++id;
    pending.set(next, resolve);
    ws.send(JSON.stringify({ id: next, method, params }));
  });
  return { call, close: () => ws.close() };
}

async function page(suffix) {
  const list = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
  const target = list.find((item) => item.type === "page" && item.url.split(/[?#]/)[0].endsWith(suffix));
  if (!target) throw new Error(`没有找到页面 ${suffix}；已打开：${list.map((item) => item.url.split("/").pop()).join(", ")}`);
  return connect(target.webSocketDebuggerUrl);
}

async function evaluate(session, expression, commandLine = false) {
  const reply = await session.call("Runtime.evaluate", {
    expression, awaitPromise: true, returnByValue: true, includeCommandLineAPI: commandLine,
  });
  if (reply.result?.exceptionDetails) throw new Error(reply.result.exceptionDetails.exception?.description ?? "执行出错");
  return reply.result?.result?.value;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const mouse = (session, type, x, y, extra = {}) => session.call("Input.dispatchMouseEvent", { type, x, y, ...extra });
const print = (value) => console.log(typeof value === "string" ? value : JSON.stringify(value, null, 2));

switch (command) {
  case "pages": {
    const list = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
    for (const item of list) console.log(item.type, item.url);
    break;
  }
  case "eval": {
    const session = await page(args[0]);
    print(await evaluate(session, args[1]));
    break;
  }
  case "main": {
    const [target] = await (await fetch(`http://127.0.0.1:${inspectPort}/json/list`)).json();
    const session = await connect(target.webSocketDebuggerUrl);
    await session.call("Runtime.enable");
    print(await evaluate(session, args[0], true));
    break;
  }
  case "shot": {
    const session = await page(args[0]);
    // A window on another Space or covered counts as hidden and stops drawing;
    // this makes it draw once more for the capture.
    await session.call("Emulation.setFocusEmulationEnabled", { enabled: true });
    await session.call("Page.bringToFront");
    const shot = await session.call("Page.captureScreenshot", { format: "png" });
    (await import("node:fs")).writeFileSync(args[1], Buffer.from(shot.result.data, "base64"));
    console.log("已保存", args[1]);
    break;
  }
  case "click": {
    const session = await page(args[0]);
    const [x, y] = [Number(args[1]), Number(args[2])];
    const button = args[3] === "right" ? "right" : "left";
    await mouse(session, "mouseMoved", x, y);
    await mouse(session, "mousePressed", x, y, { button, buttons: button === "right" ? 2 : 1, clickCount: 1 });
    await mouse(session, "mouseReleased", x, y, { button, buttons: 0, clickCount: 1 });
    break;
  }
  case "drag": {
    const session = await page(args[0]);
    const [x, y, dx, dy, steps = 10] = args.slice(1).map(Number);
    await mouse(session, "mouseMoved", x, y);
    await mouse(session, "mousePressed", x, y, { button: "left", buttons: 1, clickCount: 1 });
    for (let step = 1; step <= steps; step += 1) {
      await mouse(session, "mouseMoved", x + dx * step / steps, y + dy * step / steps, { button: "left", buttons: 1 });
      await sleep(16);
    }
    await mouse(session, "mouseReleased", x + dx, y + dy, { button: "left", buttons: 0, clickCount: 1 });
    break;
  }
  case "wheel": {
    const session = await page(args[0]);
    const [x, y, deltaY, count = 1, gap = 16] = args.slice(1).map(Number);
    for (let index = 0; index < count; index += 1) {
      void mouse(session, "mouseWheel", x, y, { deltaX: 0, deltaY });
      await sleep(gap);
    }
    await sleep(300);
    break;
  }
  case "frames": {
    // The value on every display refresh for <ms>: shows jumps, reversals
    // and dropped steps that a screenshot cannot.
    const session = await page(args[0]);
    const ms = Number(args[2]);
    const values = await evaluate(session, `new Promise((resolve) => {
      const out = [], start = performance.now();
      const tick = (now) => {
        out.push([Math.round(now - start), (${args[1]})]);
        if (now - start < ${ms}) requestAnimationFrame(tick); else resolve(out);
      };
      requestAnimationFrame(tick);
    })`);
    for (const [at, value] of values) console.log(at, JSON.stringify(value));
    break;
  }
  default:
    console.log("用法见文件开头的注释");
    process.exitCode = 1;
}
process.exit();
