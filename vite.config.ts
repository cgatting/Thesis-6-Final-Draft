import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
          '/api/semanticscholar': {
            target: 'https://api.semanticscholar.org/graph/v1',
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/semanticscholar/, ''),
          },
          '/refine': {
            target: 'http://localhost:8000',
            changeOrigin: true,
            secure: false,
          },
          '/ws': {
            target: 'ws://localhost:8000',
            ws: true,
            changeOrigin: true,
            secure: false,
          }
        },
      },
      build: {
        outDir: 'dist',
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    charts: ['recharts'],
                    pdf: ['jspdf', 'html2canvas']
                }
            }
        }
      },
      plugins: [react()],
      test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: './tests/setup.ts',
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, './src'),
        }
      }
    };
});
