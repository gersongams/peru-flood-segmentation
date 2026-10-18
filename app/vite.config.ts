import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (floodseg.system.server) runs on port 8000.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
