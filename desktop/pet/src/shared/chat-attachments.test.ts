import { describe, expect, it } from "vitest";

import {
  attachmentKind,
  attachmentLabel,
  contentWithAttachments,
  contentWithoutAttachments,
} from "./chat-attachments";

describe("chat attachments", () => {
  it("tells pictures, text files and documents apart", () => {
    expect(attachmentKind({ name: "a.png", type: "image/png" })).toBe("image");
    expect(attachmentKind({ name: "notes.md", type: "" })).toBe("text");
    expect(attachmentKind({ name: "data.json", type: "application/json" })).toBe("text");
    expect(attachmentKind({ name: "run.log", type: "text/plain" })).toBe("text");
    expect(attachmentKind({ name: "paper.pdf", type: "application/pdf" })).toBe("document");
    expect(attachmentKind({ name: "sheet.xlsx", type: "" })).toBe("document");
    expect(attachmentKind({ name: "movie.mp4", type: "video/mp4" })).toBe("other");
  });

  it("names what a message carries after the words, and strips the names back off", () => {
    const items = [{ kind: "image" as const, name: "error.png" }, { kind: "text" as const, name: "log.txt" }];
    expect(attachmentLabel("image", "error.png")).toBe("[发来图片：error.png]");
    const content = contentWithAttachments(" 帮我看看这个报错 ", items);
    expect(content).toBe("帮我看看这个报错 [发来图片：error.png] [发来文件：log.txt]");
    expect(contentWithoutAttachments(content, items)).toBe("帮我看看这个报错");
    expect(contentWithAttachments("", items.slice(0, 1))).toBe("[发来图片：error.png]");
  });
});
