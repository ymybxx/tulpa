import { describe, expect, it } from "vitest";

import { authorizeDesktopRequest } from "./desktop-request-auth";

const authorization = {
  origin: "http://127.0.0.1:59501",
  bearer: "Bearer fm_test",
};

describe("authorizeDesktopRequest", () => {
  it("adds authorization to the main window's local API requests", () => {
    expect(authorizeDesktopRequest(
      "http://127.0.0.1:59501/v1/fishes",
      { webContentsId: 12 },
      12,
      { Accept: "application/json" },
      authorization,
    )).toEqual({
      Accept: "application/json",
      Authorization: "Bearer fm_test",
    });
  });

  it("does not expose authorization to another local origin", () => {
    const headers = { Accept: "text/html" };
    expect(authorizeDesktopRequest(
      "http://127.0.0.1:8123/",
      { webContentsId: 12 },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });

  it("does not expose authorization outside the main window", () => {
    const headers = { Accept: "text/html" };
    expect(authorizeDesktopRequest(
      "http://127.0.0.1:59501/",
      { webContentsId: 99 },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });

  it("uses the attached WebContents when WebSocket webContentsId is absent", () => {
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      { attachedWebContentsId: 12, resourceType: "webSocket" },
      12,
      { Origin: "http://127.0.0.1:59501" },
      authorization,
    )).toEqual({
      Origin: "http://127.0.0.1:59501",
      Authorization: "Bearer fm_test",
    });
  });

  it("authorizes an identifier-less WebSocket only from the local API origin", () => {
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      { resourceType: "webSocket" },
      12,
      { origin: "http://127.0.0.1:59501" },
      authorization,
    )).toEqual({
      origin: "http://127.0.0.1:59501",
      Authorization: "Bearer fm_test",
    });
  });

  it("rejects an identifier-less WebSocket from an external origin", () => {
    const headers = { Origin: "https://example.com" };
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      { resourceType: "webSocket" },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });

  it("does not use origin fallback when another WebContents owns the request", () => {
    const headers = { Origin: "http://127.0.0.1:59501" };
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      { webContentsId: 99, resourceType: "webSocket" },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });

  it("rejects conflicting WebContents identifiers", () => {
    const headers = { Origin: "http://127.0.0.1:59501" };
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      {
        webContentsId: 12,
        attachedWebContentsId: 99,
        resourceType: "webSocket",
      },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });

  it("rejects an external WebSocket frame even when the main WebContents owns it", () => {
    const headers = { Origin: "https://example.com" };
    expect(authorizeDesktopRequest(
      "ws://127.0.0.1:59501/v1/tts/stream",
      { webContentsId: 12, resourceType: "webSocket", frameOrigin: "https://example.com" },
      12,
      headers,
      authorization,
    )).toBe(headers);
  });
});
