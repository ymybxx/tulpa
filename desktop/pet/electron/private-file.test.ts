import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { readPrivateText, writePrivateText } from "./private-file";
import { assertPrivateFile, literalPath, windowsPermissionsScript } from "../test-support/private-file";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), "private-file-中文-"));
  roots.push(root);
  return { root, file: path.join(root, "凭据.json") };
}

describe("private credential files", () => {
  it("replaces credentials atomically with private permissions under concurrent writes", async () => {
    const { root, file } = await fixture();
    const values = Array.from({ length: 8 }, (_, index) => JSON.stringify({ testKey: `fake-${index}` }));
    await Promise.all(values.map(value => writePrivateText(file, value)));
    expect(values).toContain(await readPrivateText(file));
    expect(await readdir(root)).toEqual([path.basename(file)]);
    await assertPrivateFile(file);
  });

  it("tightens existing permissive files before reading without changing their contents", async () => {
    const { root, file } = await fixture();
    await writeFile(file, "existing-fake-key", { mode: 0o666 });
    if (process.platform === "win32") {
      await windowsPermissionsScript(`
foreach ($target in @(${literalPath(root)}, ${literalPath(file)})) {
  $item = Get-Item -LiteralPath $target
  $acl = $item.GetAccessControl([Security.AccessControl.AccessControlSections]::Access)
  $everyone = New-Object Security.Principal.SecurityIdentifier('S-1-1-0')
  $rule = New-Object Security.AccessControl.FileSystemAccessRule($everyone, 'FullControl', 'Allow')
  $acl.AddAccessRule($rule)
  $item.SetAccessControl($acl)
}
`);
    }
    expect(await readPrivateText(file)).toBe("existing-fake-key");
    expect(await readFile(file, "utf8")).toBe("existing-fake-key");
    await assertPrivateFile(file);
  });

  it("removes its exclusive temporary when replacement fails", async () => {
    const { root } = await fixture();
    const target = path.join(root, "existing-directory");
    await mkdir(target);
    await expect(writePrivateText(target, "fake-key")).rejects.toThrow();
    expect((await readdir(root)).filter(name => name.endsWith(".tmp"))).toEqual([]);
  });
});
