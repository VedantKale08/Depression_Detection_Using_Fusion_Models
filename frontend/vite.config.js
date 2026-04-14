import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Needed for Docker port mapping
    port: 3000,
    proxy: {
      '/api': {
        // Use host.docker.internal to allow the Docker container to hit Flask running on the Windows host
        target: 'http://host.docker.internal:5000',
        changeOrigin: true,
      }
    }
  }
})
