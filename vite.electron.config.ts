import tailwindcss from "@tailwindcss/postcss";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.resolve(projectRoot, "renderer"),
  publicDir: path.resolve(projectRoot, "public"),
  base: "/",
  plugins: [react()],
  css: { postcss: { plugins: [tailwindcss()] } },
  build: {
    outDir: path.resolve(projectRoot, "dist-renderer"),
    emptyOutDir: true,
    target: "chrome142",
  },
});
