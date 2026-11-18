/// <reference types="vitest" />
/**
 * @file vite.config.ts
 * 1. What this file does:
 *    Configures Vite build tool, React plugin, development server, path aliases, and Vitest test runner.
 * 2. Overall design:
 *    Lightweight ESM configuration defining '@' alias resolving to 'src' directory,
 *    setting standard dev server port 5173, and jsdom environment for unit tests.
 * 3. Elements in the file:
 *    - react plugin configuration
 *    - resolve alias definition for '@'
 *    - dev server configuration
 *    - vitest configuration (globals, environment: 'jsdom')
 * 4. Final output of the file:
 *    Vite/Vitest configuration object consumed by Vite CLI and test runner.
 */

import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: true,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/test/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['e2e/**', 'node_modules/**'],
  },
});

