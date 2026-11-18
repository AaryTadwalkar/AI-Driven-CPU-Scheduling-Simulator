/**
 * @file frontend/playwright.config.ts
 * 1. What this file does:
 *    Configures Playwright End-to-End (E2E) testing framework, test directories,
 *    browser targets, viewport dimensions, webServer auto-spawning, and timeout limits.
 * 2. Overall design:
 *    Targets Chromium browser at canonical 1440x900 desktop resolution. Connects to the
 *    Vite development/preview server at http://localhost:5173 with auto-start and reuse.
 * 3. Elements in the file:
 *    - `config`: Playwright TestConfig definition object
 * 4. Final output of the file:
 *    Exported Playwright configuration utilized by `npx playwright test` and E2E runners.
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 45000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  use: {
    baseURL: 'http://localhost:5173',
    viewport: { width: 1440, height: 900 },
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 30000,
  },
});
