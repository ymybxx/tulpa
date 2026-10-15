import { describe, expect, it } from "vitest";

import { chatAttachmentUpload } from "./chat-attachment-upload";

describe("chat attachment uploads from a chat box", () => {
  it("passes a picture or a text file through", () => {
    expect(chatAttachmentUpload({ kind: "image", name: "a.png", mimeType: "image/png", data: "AAAA", width: 10, height: 20 }))
      .toEqual({ kind: "image", name: "a.png", mimeType: "image/png", data: "AAAA", width: 10, height: 20 });
    expect(chatAttachmentUpload({ kind: "text", name: "a.txt", mimeType: "text/plain", text: "hi", extra: 1 }))
      .toEqual({ kind: "text", name: "a.txt", mimeType: "text/plain", text: "hi" });
  });

  it("refuses anything else", () => {
    expect(chatAttachmentUpload({ kind: "image", name: "a.svg", mimeType: "image/svg+xml", data: "AAAA" })).toBeNull();
    expect(chatAttachmentUpload({ kind: "text", name: "a.txt", mimeType: "text/plain", text: "  " })).toBeNull();
    expect(chatAttachmentUpload({ kind: "file", name: "a.pdf", mimeType: "application/pdf", data: "AAAA" })).toBeNull();
    expect(chatAttachmentUpload(null)).toBeNull();
  });
});
