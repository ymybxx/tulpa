import { app, BrowserWindow, session } from "electron";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { WebSocketServer } from "ws";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EXPECTED_BEARER = "Bearer fm_electron_desktop_auth_test";
const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "desktop-auth-qa-"));
app.setPath("userData", path.join(temporaryRoot, "profile"));

void app.whenReady().then(run);

async function run() {
  let window;
  let otherWindow;
  let server;
  let externalServer;
  let webSocketServer;
  let exitCode = 0;
  try {
    const { authorizeDesktopRequest } = require(
      path.join(root, "dist-electron", "electron", "desktop-request-auth.js"),
    );
    const handshakes = deferred();
    const handshakeResults = [];
    const httpResults = [];
    const httpResponses = new Map(
      ["authorized", "unauthenticated", "other-window"].map((name) => [name, deferred()]),
    );
    webSocketServer = new WebSocketServer({ noServer: true });
    server = createServer((request, response) => {
      const requestUrl = new URL(request.url, "http://127.0.0.1");
      if (requestUrl.pathname !== "/frame") {
        response.writeHead(404).end();
        return;
      }
      const authorized = request.headers.authorization === EXPECTED_BEARER;
      const result = {
        case: requestUrl.searchParams.get("case"),
        method: request.method,
        status: authorized ? 200 : 404,
        authorization: request.headers.authorization ?? null,
      };
      httpResults.push(result);
      if (!authorized) {
        response.writeHead(404, { "Content-Type": "application/json" });
        response.end('{"error":{"code":"not_found","message":"not found"}}');
        httpResponses.get(result.case)?.resolve(result);
        return;
      }
      const { port } = server.address();
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(
        `<main>Authenticated chat</main><script>
          parent.postMessage("chat-ready", "*");
          new WebSocket("ws://127.0.0.1:${port}/v1/tts/stream");
        </script>`,
      );
      httpResponses.get(result.case)?.resolve(result);
    });
    server.on("upgrade", (request, socket, head) => {
      handshakeResults.push({
        authorization: request.headers.authorization ?? null,
        origin: request.headers.origin ?? null,
      });
      webSocketServer.handleUpgrade(request, socket, head, (webSocket) => {
        webSocket.close();
        if (handshakeResults.length === 2) handshakes.resolve(handshakeResults);
      });
    });
    await listen(server);
    const { port } = server.address();
    const apiOrigin = `http://127.0.0.1:${port}`;
    let authorization = { origin: apiOrigin, bearer: EXPECTED_BEARER };

    const externalHttpResults = [];
    externalServer = createServer((request, response) => {
      externalHttpResults.push({ authorization: request.headers.authorization ?? null });
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(
        `<script>new WebSocket("ws://127.0.0.1:${port}/v1/tts/stream")</script>`,
      );
    });
    await listen(externalServer);
    const externalOrigin = `http://127.0.0.1:${externalServer.address().port}`;

    const page = path.join(temporaryRoot, "index.html");
    const otherPage = path.join(temporaryRoot, "other.html");
    await writeFile(
      page,
      `<script>
        window.chatReady = false;
        addEventListener("message", (event) => {
          if (event.origin === ${JSON.stringify(apiOrigin)} && event.data === "chat-ready") {
            window.chatReady = true;
          }
        });
      </script>
      <iframe id="chat" src="${apiOrigin}/frame?case=authorized"></iframe>
      <iframe src="${externalOrigin}/frame"></iframe>`,
    );
    await writeFile(
      otherPage,
      `<iframe src="${apiOrigin}/frame?case=other-window"></iframe>`,
    );

    window = new BrowserWindow({
      show: false,
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });
    const requestDetails = [];
    session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
      if (
        details.url.startsWith(`${apiOrigin}/frame`)
        || details.url === `ws://127.0.0.1:${port}/v1/tts/stream`
      ) {
        requestDetails.push({
          url: details.url,
          resourceType: details.resourceType,
          webContentsId: details.webContentsId ?? null,
          attachedWebContentsId: details.webContents?.id ?? null,
          frameOrigin: details.frame?.origin ?? null,
        });
      }
      callback({
        requestHeaders: authorizeDesktopRequest(
          details.url,
          {
            webContentsId: details.webContentsId,
            attachedWebContentsId: details.webContents?.id,
            resourceType: details.resourceType,
            frameOrigin: details.frame?.origin,
          },
          window.webContents.id,
          details.requestHeaders,
          authorization,
        ),
      });
    });

    await withTimeout(window.loadFile(page), "embedded chat page load timed out");
    const results = await withTimeout(
      handshakes.promise,
      "desktop WebSocket authentication handshake timed out",
    );
    const authenticated = await withTimeout(
      httpResponses.get("authorized").promise,
      "authenticated embedded chat request timed out",
    );
    const chatRendered = await window.webContents.executeJavaScript("window.chatReady === true");

    authorization = null;
    await window.webContents.executeJavaScript(
      `document.getElementById("chat").src = ${JSON.stringify(`${apiOrigin}/frame?case=unauthenticated`)}`,
    );
    const unauthenticated = await withTimeout(
      httpResponses.get("unauthenticated").promise,
      "unauthenticated embedded chat request timed out",
    );

    authorization = { origin: apiOrigin, bearer: EXPECTED_BEARER };
    otherWindow = new BrowserWindow({
      show: false,
      webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
    });
    await withTimeout(otherWindow.loadFile(otherPage), "other-window chat page load timed out");
    const other = await withTimeout(
      httpResponses.get("other-window").promise,
      "other-window embedded chat request timed out",
    );

    const local = results.find((result) => result.origin === apiOrigin);
    const external = results.find((result) => result.origin === externalOrigin);
    const checks = {
      embeddedChatAuthorized: authenticated.method === "GET"
        && authenticated.status === 200
        && authenticated.authorization === EXPECTED_BEARER
        && chatRendered,
      unauthenticatedChatRejected: unauthenticated.status === 404
        && unauthenticated.authorization === null,
      otherWindowRejected: other.status === 404 && other.authorization === null,
      localWebSocketAuthorized: local?.authorization === EXPECTED_BEARER,
      externalWebSocketIsolated: external?.authorization === null,
      externalHttpIsolated: externalHttpResults.length > 0
        && externalHttpResults.every((result) => result.authorization === null),
      realEmbeddedRequests: requestDetails.filter((details) => details.resourceType === "subFrame").length === 3
        && requestDetails.filter((details) => details.resourceType === "webSocket").length === 2,
    };
    const ok = Object.values(checks).every(Boolean);
    process.stdout.write(`${JSON.stringify({ ok, checks, requestDetails, http: httpResults, handshakes: results })}\n`);
    if (!ok) exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.stack : String(error));
    exitCode = 1;
  } finally {
    session.defaultSession.webRequest.onBeforeSendHeaders(null);
    window?.destroy();
    otherWindow?.destroy();
    for (const client of webSocketServer?.clients ?? []) client.terminate();
    if (webSocketServer) await close(webSocketServer);
    if (server) await close(server);
    if (externalServer) await close(externalServer);
    if (temporaryRoot && path.basename(temporaryRoot).startsWith("desktop-auth-qa-")) {
      await rm(temporaryRoot, { recursive: true, force: true });
    }
    app.exit(exitCode);
  }
}

function deferred() {
  let resolve;
  const promise = new Promise((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
}

function close(server) {
  return new Promise((resolve) => {
    server.close(() => resolve());
    server.closeAllConnections?.();
  });
}

async function withTimeout(promise, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), 10_000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
