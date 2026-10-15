import { execFile } from "node:child_process";
import { stat } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import assert from "node:assert/strict";

const exec = promisify(execFile);

export async function windowsPermissionsScript(script: string): Promise<string> {
  const { stdout } = await exec(path.join(process.env.SystemRoot ?? "C:\\Windows", "System32/WindowsPowerShell/v1.0/powershell.exe"),
    ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from("$ErrorActionPreference='Stop'\n" + script, "utf16le").toString("base64")],
    { windowsHide: true, timeout: 15_000 });
  return stdout;
}

export function literalPath(file: string): string {
  return `[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${Buffer.from(file, "utf8").toString("base64")}'))`;
}

/** Assert actual Windows ACEs; Unix mode bits are meaningful only on Unix. */
export async function assertPrivateFile(file: string): Promise<void> {
  if (process.platform !== "win32") {
    assert.equal((await stat(file)).mode & 0o777, 0o600);
    return;
  }
  const output = await windowsPermissionsScript(`
$target = ${literalPath(file)}
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$items = @($target, [IO.Path]::GetDirectoryName($target)) | ForEach-Object {
  $acl = (Get-Item -LiteralPath $_ -Force).GetAccessControl([Security.AccessControl.AccessControlSections]::Access)
  @{ protected = $acl.AreAccessRulesProtected; rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | ForEach-Object { @{sid=$_.IdentityReference.Value; allow=($_.AccessControlType -eq 'Allow'); rights=[int]$_.FileSystemRights} }) }
}
@{ sid=$sid; items=@($items) } | ConvertTo-Json -Depth 5 -Compress
`);
  const value = JSON.parse(output) as { sid: string; items: Array<{ protected: boolean; rules: Array<{ sid: string; allow: boolean; rights: number }> }> };
  assert.equal(value.items.length, 2);
  assert.equal(value.items[1]!.protected, true, "parent inheritance must be disabled");
  for (const item of value.items) {
    assert.equal(item.rules.length, 1, "no broad or additional ACE may remain");
    assert.equal(item.rules[0]!.sid, value.sid);
    assert.equal(item.rules[0]!.allow, true);
    assert.equal(item.rules[0]!.rights & 0x1f01ff, 0x1f01ff, "current user keeps full control");
  }
}
