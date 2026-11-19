import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, Vite forwards /api to the backend, like nginx does on the server.
// Inside Docker the backend is http://api:8000, on your laptop http://localhost:8000.
const apiTarget = process.env.API_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': apiTarget,
    },
  },
})
