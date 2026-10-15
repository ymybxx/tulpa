import { defineConfig, type Plugin } from "vitest/config";
import path from "node:path";

/**
 * The dev server injects CSS as inline <style> tags and talks to the page over
 * a WebSocket, both of which the strict pet-chat CSP forbids. Relax exactly
 * those two directives while serving; built pages keep the original policy.
 */
function devContentSecurityPolicy(): Plugin {
  return {
    name: "dev-content-security-policy",
    apply: "serve",
    transformIndexHtml: (html) => html
      .replace("style-src 'self'", "style-src 'self' 'unsafe-inline'")
      .replace("connect-src 'none'", "connect-src ws://127.0.0.1:5173"),
  };
}

const inputs: Record<string, string> = {
  pet: path.resolve(__dirname, "index.html"),
  "pet-chat": path.resolve(__dirname, "pet-chat.html"),
  main: path.resolve(__dirname, "main.html"),
  "emotion-sampler": path.resolve(__dirname, "emotion-sampler.html"),
};
if (process.env.TEXTURE_PATCH_HARNESS === "1") {
  inputs["texture-patch-harness"] = path.resolve(__dirname, "texture-patch-harness.html");
}

export default defineConfig({
  base: "./",
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  plugins: [devContentSecurityPolicy()],
  publicDir: "public-runtime",
  build: {
    outDir: "dist-renderer",
    emptyOutDir: true,
    target: "chrome126",
    rollupOptions: {
      input: inputs,
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  resolve: {
    alias: {
      "@cubism": path.resolve(__dirname, "vendor/cubism-web-framework/dist"),
    },
  },
  test: {
    exclude: ["node_modules/**", "dist-renderer/**", "dist-electron/**", "release/**"],
  },
});
