import { cp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const petRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(
  petRoot,
  "vendor",
  "cubism-web-framework",
  "Shaders",
  "WebGL",
);
const target = path.join(petRoot, "public-runtime", "cubism-shaders");

await rm(target, { recursive: true, force: true });
await mkdir(path.dirname(target), { recursive: true });
await cp(source, target, { recursive: true });
