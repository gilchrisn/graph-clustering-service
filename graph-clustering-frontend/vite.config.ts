// vite.config.ts - Clean version without testing
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  
  // Development server configuration
  server: {
    port: 3000,
    open: true,
    cors: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,

        // rewrite: (path) => path.replace(/^\/api/, '/api')
      }
    }
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'esbuild',
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          store: ['zustand'],
          utils: ['./src/utils/drillDownEngine', './src/utils/formatters']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  },

  // Path resolution
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@hooks': resolve(__dirname, './src/hooks'),
      '@services': resolve(__dirname, './src/services'),
      '@store': resolve(__dirname, './src/store'),
      '@types': resolve(__dirname, './src/types'),
      '@utils': resolve(__dirname, './src/utils'),
      '@styles': resolve(__dirname, './src/styles')
    }
  },

  // Environment variables
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version)
  },

  // CSS configuration
  css: {
    postcss: './postcss.config.js',
    devSourcemap: true
  },

  // Optimization
  optimizeDeps: {
    include: ['react', 'react-dom', 'zustand']
  },

  // Preview server configuration
  preview: {
    port: 4173,
    open: true
  }
});