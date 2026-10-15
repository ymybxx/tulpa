import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { chmod, lstat, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const directories = new Map<string, Promise<void>>();
const files = new Map<string, Promise<void>>();
const writes = new Map<string, Promise<void>>();

function fileKey(file: string): string {
  return process.platform === "win32" ? path.resolve(file).toLowerCase() : path.resolve(file);
}

function cachedPreparation(cache: Map<string, Promise<void>>, file: string, prepare: () => Promise<void>): Promise<void> {
  const key = fileKey(file);
  const existing = cache.get(key);
  if (existing) return existing;
  const pending = prepare().catch(error => { cache.delete(key); throw error; });
  cache.set(key, pending);
  return pending;
}

/** Protect the directory before any secret bytes are written, once per host. */
export function preparePrivateDirectory(directory: string): Promise<void> {
  return cachedPreparation(directories, directory, async () => {
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Private directory must be a real directory");
    if (process.platform === "win32") await restrictWindowsAccess(directory, true);
    else await chmod(directory, 0o700);
  });
}

/** Existing files are migrated before reading; ACL failures must reach callers. */
export async function readPrivateText(file: string): Promise<string> {
  await preparePrivateDirectory(path.dirname(file));
  await cachedPreparation(files, file, async () => {
    const info = await lstat(file);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Private file must be a regular file");
    if (process.platform === "win32") await restrictWindowsAccess(file, false);
    else await chmod(file, 0o600);
  });
  return readFile(file, "utf8");
}

/** Unique exclusive temporaries inherit the protected directory's Windows ACL. */
export function writePrivateText(file: string, value: string): Promise<void> {
  const key = fileKey(file);
  // Windows can reject simultaneous replacements of the same destination even
  // with different temporary names. Serialize writes without poisoning retries.
  const pending = (writes.get(key) ?? Promise.resolve()).catch(() => undefined)
    .then(() => writePrivateTextNow(file, value));
  writes.set(key, pending);
  return pending.finally(() => { if (writes.get(key) === pending) writes.delete(key); });
}

async function writePrivateTextNow(file: string, value: string): Promise<void> {
  await preparePrivateDirectory(path.dirname(file));
  const temporary = `${file}.${randomUUID()}.tmp`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(value, "utf8");
    await handle.close();
    await rename(temporary, file);
    files.set(fileKey(file), Promise.resolve());
  } finally {
    await handle.close();
    await rm(temporary, { force: true });
  }
}

async function restrictWindowsAccess(file: string, directory: boolean): Promise<void> {
  // Pass only a path, never file contents or credentials, to the child process.
  const encodedPath = Buffer.from(path.resolve(file), "utf8").toString("base64");
  const script = `
$ErrorActionPreference = 'Stop'
$target = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encodedPath}'))
$item = Get-Item -LiteralPath $target -Force
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = $item.GetAccessControl([Security.AccessControl.AccessControlSections]::Access)
$acl.SetAccessRuleProtection($true, $false)
foreach ($rule in @($acl.Access)) { $acl.RemoveAccessRuleSpecific($rule) }
$inherit = [Security.AccessControl.InheritanceFlags]::${directory ? "ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit" : "None"}
$rule = New-Object Security.AccessControl.FileSystemAccessRule($sid, [Security.AccessControl.FileSystemRights]::FullControl, $inherit, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
$acl.AddAccessRule($rule)
$item.SetAccessControl($acl)
$verified = $item.GetAccessControl([Security.AccessControl.AccessControlSections]::Access)
$rules = @($verified.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
if (!$verified.AreAccessRulesProtected -or $rules.Count -ne 1 -or $rules[0].IdentityReference.Value -ne $sid.Value -or $rules[0].AccessControlType -ne 'Allow' -or ($rules[0].FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -ne [Security.AccessControl.FileSystemRights]::FullControl) { throw 'Private file ACL verification failed' }
`;
  await execFileAsync(path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")],
    { windowsHide: true, timeout: 15_000, maxBuffer: 64 * 1024 });
}
