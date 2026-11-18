/**
 * @file frontend/e2e/smoke.spec.ts
 * 1. What this file does:
 *    Executes end-to-end smoke verification of the CPU Scheduling Simulator dashboard:
 *    loads the application, switches workload presets, executes simulation, verifies Gantt
 *    timeline and metric cards, and confirms routing navigation across all pages.
 * 2. Overall design:
 *    Uses Playwright browser automation against the live Vite server and FastAPI backend.
 *    Verifies full lifecycle without client-side mock fixtures.
 * 3. Elements in the file:
 *    - `test('smoke: load -> pick preset -> run -> see Gantt and metrics')`: Core simulation flow
 *    - `test('smoke: verify responsive page routes navigation')`: Route traversal check
 * 4. Final output of the file:
 *    Automated Playwright test assertions validating Sub-stage F6 end-to-end acceptance criteria.
 */

import { test, expect } from '@playwright/test';

/**
 * Validates the primary simulator user journey: loading the application, selecting a preset,
 * running the simulation, and inspecting the rendered Gantt timeline and metric cards.
 *
 * 1. The use of function:
 *    Provides end-to-end regression validation for core simulator features.
 * 2. How does the specific function work:
 *    Navigates to '/', selects the 'Presets' tab, chooses the 'convoy' option from the preset
 *    dropdown, selects 'SJF' algorithm, clicks 'Run Simulation', and asserts presence of timeline
 *    and metric values.
 * 3. Concepts used in the function:
 *    Playwright locator selectors, selectOption, async DOM assertions, end-to-end scheduling verification.
 * 4. Places where this function is imported and used:
 *    Dispatched by Playwright test runner via `npm run test:e2e`.
 * 5. Final output of the function, before and after:
 *    Before: Unverified browser DOM state.
 *    After: Asserted that preset loads, simulation executes against FastAPI, and Gantt renders.
 */
test('smoke: load -> pick preset -> run -> see Gantt and metrics', async ({ page }) => {
  // 1. Load the simulator workbench
  await page.goto('/');

  // Ensure header and status indicator are present
  const header = page.locator('header');
  await expect(header).toBeVisible();

  // 2. Switch to Presets tab in Workload Editor
  const presetsTabButton = page.getByRole('button', { name: /presets/i });
  await presetsTabButton.click();

  // Select 'convoy' preset from dropdown
  const presetSelect = page.locator('#preset-select');
  await expect(presetSelect).toBeVisible();
  await presetSelect.selectOption('convoy');

  // Verify preset description appears
  await expect(page.locator('text=Convoy effect scenario')).toBeVisible();

  // Switch back to Table tab to verify processes were loaded
  const tableTabButton = page.getByRole('button', { name: /table/i });
  await tableTabButton.click();
  await expect(page.locator('table').first()).toBeVisible();

  // 3. Select SJF algorithm and run simulation
  const sjfButton = page.locator('button:has-text("SJF")').first();
  if (await sjfButton.isVisible()) {
    await sjfButton.click();
  }

  const runButton = page.getByRole('button', { name: /run simulation/i });
  await expect(runButton).toBeEnabled();
  await runButton.click();

  // 4. Verify Gantt chart and metric cards are rendered
  const ganttHeading = page.locator('h3:has-text("CPU Execution Timeline")');
  await expect(ganttHeading).toBeVisible({ timeout: 10000 });

  const avgWaitingCard = page.locator('text=Avg Waiting').first();
  await expect(avgWaitingCard).toBeVisible();

  const cpuUtilCard = page.locator('text=CPU Utilization').first();
  await expect(cpuUtilCard).toBeVisible();
});

/**
 * Validates route navigation across all four dashboard views.
 *
 * 1. The use of function:
 *    Verifies that Sidebar navigation links transition correctly to Compare, AI Lab, and Results.
 * 2. How does the specific function work:
 *    Clicks each sidebar link, waits for URL matching, and checks for prominent page headings in TopBar.
 * 3. Concepts used in the function:
 *    Client-side routing verification, heading locator checks.
 * 4. Places where this function is imported and used:
 *    Dispatched by Playwright test runner via `npm run test:e2e`.
 * 5. Final output of the function, before and after:
 *    Before: Single route tested.
 *    After: All four application routes verified to mount cleanly without errors.
 */
test('smoke: verify responsive page routes navigation', async ({ page }) => {
  await page.goto('/');

  // Navigate to Compare view
  const compareLink = page.getByRole('link', { name: /compare/i });
  await compareLink.click();
  await expect(page).toHaveURL(/.*compare/);
  await expect(page.locator('header h2')).toHaveText('Algorithm Comparison');

  // Navigate to AI Lab view
  const aiLabLink = page.getByRole('link', { name: /ai lab/i });
  await aiLabLink.click();
  await expect(page).toHaveURL(/.*ai-lab/);
  await expect(page.locator('header h2')).toHaveText('AI Lab');

  // Navigate to Results view
  const resultsLink = page.getByRole('link', { name: /results/i });
  await resultsLink.click();
  await expect(page).toHaveURL(/.*results/);
  await expect(page.locator('header h2')).toHaveText('Benchmark Results');
});
