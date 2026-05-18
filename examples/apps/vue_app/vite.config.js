import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  preview: {
    port: 8080,
    host: true,
    allowedHosts: ['.unionai.cloud'],
  },
})
