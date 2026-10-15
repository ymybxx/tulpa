import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { assertOwnerVoiceSender } from "./owner-voice-ipc";

describe("owner voice IPC authority", () => {
  it("allows only the main window's top-level frame", () => {
    const main = { mainFrame: {} };
    expect(() => assertOwnerVoiceSender({ sender: main, senderFrame: main.mainFrame }, main)).not.toThrow();
    expect(() => assertOwnerVoiceSender({ sender: {}, senderFrame: main.mainFrame }, main)).toThrow();
    expect(() => assertOwnerVoiceSender({ sender: main, senderFrame: {} }, main)).toThrow();
    expect(() => assertOwnerVoiceSender({ sender: main, senderFrame: main.mainFrame }, undefined)).toThrow();
  });
  it("preload sends no renderer-controlled path, and every handler checks authority", async () => {
    const preload = await readFile(new URL("./preload.ts", import.meta.url), "utf8");
    const main = await readFile(new URL("./main.ts", import.meta.url), "utf8");
    for (const action of ["status", "state", "cancel"]) {
      expect(preload).toContain(`ipcRenderer.invoke("desktop:owner-voice:${action}")`);
      expect(main).toMatch(new RegExp(`"desktop:owner-voice:${action}", \\(event\\) => \\{\\s+assertOwnerVoiceSender`));
    }
    // Deleting takes at most a recording time the status listed, checked in main.
    expect(preload).toContain('ipcRenderer.invoke("desktop:owner-voice:delete", typeof createdAt === "string" ? createdAt : undefined)');
    expect(main).toMatch(/"desktop:owner-voice:delete", \(event, createdAt: unknown\) => \{\s+assertOwnerVoiceSender\(event, mainWindow\?\.webContents\);/);
    expect(main).toContain("/^[0-9T:.+\\-]{10,64}$/.test(createdAt)");
    // Enrolling takes one flag, coerced to a boolean on both sides: append or replace.
    expect(preload).toContain('ipcRenderer.invoke("desktop:owner-voice:enroll", append === true)');
    expect(main).toMatch(/"desktop:owner-voice:enroll", \(event, append: unknown\) => \{\s+assertOwnerVoiceSender\(event, mainWindow\?\.webContents\);\s+return runtime\.enrollOwnerVoice\(append === true\);/);
  });
});
