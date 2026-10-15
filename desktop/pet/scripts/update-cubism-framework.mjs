import { execFileSync } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const petRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destination = path.join(petRoot, "vendor", "cubism-web-framework");
const repository = "https://github.com/Live2D/CubismWebFramework.git";
const tag = process.env.CUBISM_FRAMEWORK_REF || "5-r.5";
const temporary = await mkdtemp(path.join(os.tmpdir(), "cubism-web-framework-"));
const localPatches = [
  path.join(petRoot, "vendor", "patches", "official-r5-shared-offscreen-lifecycle.patch"),
];

try {
  execFileSync("git", ["clone", "--depth", "1", "--branch", tag, repository, temporary], {
    stdio: "inherit",
  });
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: temporary,
    encoding: "utf8",
  }).trim();
  await rm(path.join(destination, "src"), { recursive: true, force: true });
  await rm(path.join(destination, "Shaders"), { recursive: true, force: true });
  await cp(path.join(temporary, "src"), path.join(destination, "src"), { recursive: true });
  await cp(path.join(temporary, "Shaders", "WebGL"), path.join(destination, "Shaders", "WebGL"), {
    recursive: true,
  });
  for (const patchFile of localPatches) {
    execFileSync("git", ["apply", "--whitespace=nowarn", patchFile], {
      cwd: petRoot,
      stdio: "inherit",
    });
  }
  await cp(path.join(temporary, "LICENSE.md"), path.join(destination, "LICENSE.md"));
  for (const [source, target] of [
    ["README.md", "README.upstream.md"],
    ["CHANGELOG.md", "CHANGELOG.upstream.md"],
  ]) {
    await cp(path.join(temporary, source), path.join(destination, target));
  }
  const metadata = JSON.parse(await readFile(path.join(destination, "UPSTREAM.json"), "utf8"));
  await writeFile(
    path.join(destination, "UPSTREAM.json"),
    `${JSON.stringify({
      ...metadata,
      repository,
      tag,
      commit,
      updatedAt: new Date().toISOString().slice(0, 10),
    }, null, 2)}\n`,
    "utf8",
  );
  console.log(`Cubism Web Framework 已更新到 ${tag} (${commit})`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
