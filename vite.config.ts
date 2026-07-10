import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    strictPort: true,
    watch: {
      // Don't watch heavy non-source trees. The Python `.venv` (torch/transformers/
      // sympy from the semantic-embedding tooling) holds ~38k files and blows past the
      // inotify watcher limit (ENOSPC) if Vite tries to watch it. NOTE: Vite 7's watcher
      // takes a predicate/regex here, NOT glob strings — a function is version-robust.
      ignored: (p: string) =>
        /[\\/](\.venv|\.git|node_modules|target|models?|embeddings)([\\/]|$)/.test(p),
    },
  },
  build: {
    outDir: "build",
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, "index.html"),
        broadcast: path.resolve(__dirname, "broadcast-output.html"),
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
