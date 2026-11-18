/**
 * @file frontend/scripts/capture_screenshots.ts
 * 1. What this file does:
 *    Automates capture of the 14 mandatory report screenshots specified in Section 9 of
 *    FRONTEND_GUIDE.md at 1440x900 resolution using Playwright browser automation.
 * 2. Overall design:
 *    Connects to the running dashboard application, sets viewport to 1440x900, navigates
 *    across Simulator, Compare, AI Lab, and Results views, exercises interactive controls
 *    (presets, generator, algorithm execution, CS cost 0/1, tab switches, dark theme toggle),
 *    and saves verified, non-empty PNG files into the root `screenshots/` directory.
 * 3. Elements in the file:
 *    - `captureAllScreenshots`: Master orchestration routine capturing all 14 views
 *    - Helper functions for page settling and element scrolling
 *    - Verification assertions checking file sizes and presence
 * 4. Final output of the file:
 *    14 high-resolution 1440x900 PNG screenshots stored in root `screenshots/` directory.
 */

import { chromium, type Page } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Target output directory in repository root (c:\...\scheduling_simulator_rl\screenshots)
const OUTPUT_DIR = path.resolve(__dirname, '../../screenshots');
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

/**
 * Ensures a directory exists, creating parent paths recursively if necessary.
 *
 * 1. The use of function:
 *    Initializes target screenshot output storage directory.
 * 2. How does the specific function work:
 *    Checks if target directory exists on disk via fs.existsSync, invokes fs.mkdirSync with recursive true.
 * 3. Concepts used in the function:
 *    Filesystem idempotent directory creation.
 * 4. Places where this function is imported and used:
 *    Internal to capture_screenshots script.
 * 5. Final output of the function, before and after:
 *    Before: Missing or existing folder.
 *    After: Confirmed existing directory ready for image writes.
 *
 * @param dir - Absolute directory path.
 */
function ensureDirectory(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Sets application color theme state in localStorage and HTML document element.
 *
 * 1. The use of function:
 *    Enforces explicit light or dark theme mode for consistent screenshot captures.
 * 2. How does the specific function work:
 *    Executes client-side script mutating document.documentElement classList and localStorage key.
 * 3. Concepts used in the function:
 *    Client-side DOM manipulation, Tailwind dark mode class orchestration.
 * 4. Places where this function is imported and used:
 *    Internal helper used before taking light-mode and dark-mode snapshots.
 * 5. Final output of the function, before and after:
 *    Before: Ambiguous or inherited theme state.
 *    After: Explicit 'light' or 'dark' theme applied with refreshed styles.
 *
 * @param page - Playwright page instance.
 * @param theme - 'light' or 'dark'.
 */
async function setTheme(page: Page, theme: 'light' | 'dark'): Promise<void> {
  await page.evaluate((targetTheme) => {
    localStorage.setItem('schedsim-theme', targetTheme);
    if (targetTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, theme);
  await page.waitForTimeout(400);
}

/**
 * Captures a named screenshot at the canonical 1440x900 viewport resolution.
 *
 * 1. The use of function:
 *    Saves rendered browser viewport to disk with progress logging.
 * 2. How does the specific function work:
 *    Constructs destination path, invokes page.screenshot, and logs generated file size.
 * 3. Concepts used in the function:
 *    Playwright viewport capture, filesystem verification.
 * 4. Places where this function is imported and used:
 *    Called 14 times for each required checklist deliverable.
 * 5. Final output of the function, before and after:
 *    Before: Uncaptured DOM state.
 *    After: Valid PNG image file on disk.
 *
 * @param page - Playwright page instance.
 * @param filename - Target filename (e.g. '01_input_workload_editor.png').
 */
async function snap(page: Page, filename: string): Promise<void> {
  const targetPath = path.join(OUTPUT_DIR, filename);
  await page.waitForTimeout(500); // Allow animations and charts to settle
  await page.screenshot({ path: targetPath, fullPage: false });
  const stats = fs.statSync(targetPath);
  console.log(`[Captured] ${filename} (${Math.round(stats.size / 1024)} KB) -> ${targetPath}`);
}

/**
 * Main automation entry point capturing all 14 mandatory report screenshots.
 *
 * 1. The use of function:
 *    Coordinates browser launch, view navigation, user interaction simulation, and image capture.
 * 2. How does the specific function work:
 *    Spawns Playwright Chromium at 1440x900, iterates sequentially through all 14 checklist steps,
 *    asserts DOM readiness before each capture, and validates generated images.
 * 3. Concepts used in the function:
 *    Browser automation orchestration, interactive UI synthesis, robust timing synchronization.
 * 4. Places where this function is imported and used:
 *    Executed via `npx tsx scripts/capture_screenshots.ts` or `npm run screenshots`.
 * 5. Final output of the function, before and after:
 *    Before: Empty screenshots directory.
 *    After: 14 verified PNG files matching Section 9 checklist.
 */
async function captureAllScreenshots(): Promise<void> {
  console.log('================================================================');
  console.log(' Starting Automated Screenshot Capture for Report (1440x900)   ');
  console.log(` Target Directory: ${OUTPUT_DIR}`);
  console.log(` Dashboard URL:    ${BASE_URL}`);
  console.log('================================================================\n');

  ensureDirectory(OUTPUT_DIR);

  const browser = await chromium.launch({
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });

  const page = await context.newPage();

  try {
    // --------------------------------------------------------------------------
    // 01_input_workload_editor.png: workload table with a preset loaded
    // --------------------------------------------------------------------------
    console.log('[1/14] Capturing 01_input_workload_editor.png...');
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });
    await setTheme(page, 'light');

    // Switch to Presets tab
    await page.getByRole('button', { name: /presets/i }).click();
    await page.locator('#preset-select').selectOption('convoy');
    await page.waitForTimeout(400);

    // Switch back to Table tab to display the full workload table
    await page.getByRole('button', { name: /table/i }).click();
    await page.waitForSelector('table');
    await page.locator('[data-testid="workload-editor"], .rounded-card').first().scrollIntoViewIfNeeded();
    await snap(page, '01_input_workload_editor.png');

    // --------------------------------------------------------------------------
    // 02_generator_seeded.png: generator form showing scenario and seed
    // --------------------------------------------------------------------------
    console.log('[2/14] Capturing 02_generator_seeded.png...');
    await page.getByRole('button', { name: /generator/i }).click();
    await page.waitForSelector('#scenario-select');
    await page.locator('#scenario-select').selectOption('bimodal');

    // Fill seed 42
    const seedInput = page.locator('#seed-input');
    await seedInput.fill('42');

    // Click Generate Workload to trigger synthesis and show confirmation alert
    await page.getByRole('button', { name: /generate workload/i }).click();
    await page.waitForTimeout(600);
    await snap(page, '02_generator_seeded.png');

    // --------------------------------------------------------------------------
    // 03_gantt_fcfs.png: Gantt with FCFS and metric cards visible
    // --------------------------------------------------------------------------
    console.log('[3/14] Capturing 03_gantt_fcfs.png...');
    await page.getByRole('button', { name: /table/i }).click();

    // Select FCFS algorithm
    const fcfsBtn = page.locator('button:has-text("FCFS")').first();
    await fcfsBtn.click();
    await page.getByRole('button', { name: /run simulation/i }).click();
    await page.waitForSelector('h3:has-text("CPU Execution Timeline")');
    await page.waitForTimeout(600);

    // Scroll to position Gantt chart and MetricCards prominently
    await page.locator('h3:has-text("CPU Execution Timeline")').scrollIntoViewIfNeeded();
    await snap(page, '03_gantt_fcfs.png');

    // --------------------------------------------------------------------------
    // 04_gantt_sjf.png: Gantt with SJF and metric cards visible
    // --------------------------------------------------------------------------
    console.log('[4/14] Capturing 04_gantt_sjf.png...');
    const sjfBtn = page.locator('button:has-text("SJF")').first();
    await sjfBtn.click();
    await page.getByRole('button', { name: /run simulation/i }).click();
    await page.waitForTimeout(600);
    await page.locator('h3:has-text("CPU Execution Timeline")').scrollIntoViewIfNeeded();
    await snap(page, '04_gantt_sjf.png');

    // --------------------------------------------------------------------------
    // 05_gantt_rr.png: Gantt with RR and metric cards visible
    // --------------------------------------------------------------------------
    console.log('[5/14] Capturing 05_gantt_rr.png...');
    const rrBtn = page.locator('button:has-text("Round Robin")').first();
    await rrBtn.click();
    await page.getByRole('button', { name: /run simulation/i }).click();
    await page.waitForTimeout(600);
    await page.locator('h3:has-text("CPU Execution Timeline")').scrollIntoViewIfNeeded();
    await snap(page, '05_gantt_rr.png');

    // --------------------------------------------------------------------------
    // 06_gantt_qlearning_trace.png: Gantt plus Decision Trace
    // --------------------------------------------------------------------------
    console.log('[6/14] Capturing 06_gantt_qlearning_trace.png...');
    const rlBtn = page.locator('button:has-text("Q-Learning")').first();
    await rlBtn.click();
    await page.getByRole('button', { name: /run simulation/i }).click();
    await page.waitForSelector('h3:has-text("Decision Trace")', { timeout: 15000 });
    await page.waitForTimeout(600);

    // Position Gantt chart and Decision Trace cleanly in viewport
    await page.locator('h3:has-text("Decision Trace")').scrollIntoViewIfNeeded();
    await snap(page, '06_gantt_qlearning_trace.png');

    // --------------------------------------------------------------------------
    // 07_compare_all.png: comparison table, bars and stacked Gantt
    // --------------------------------------------------------------------------
    console.log('[7/14] Capturing 07_compare_all.png...');
    await page.goto(`${BASE_URL}/compare`, { waitUntil: 'networkidle' });
    await setTheme(page, 'light');

    // Ensure CS cost = 0 (0 ticks)
    const cs0Btn = page.locator('button:has-text("0 ticks")');
    if (await cs0Btn.isVisible()) {
      await cs0Btn.click();
    }

    // Trigger run comparison if button is present
    const runCompareBtn = page.getByRole('button', { name: /run all comparison/i });
    if (await runCompareBtn.isVisible()) {
      await runCompareBtn.click();
    }
    await page.waitForSelector('table', { timeout: 15000 });
    await page.waitForTimeout(800);

    // Scroll to comparison table and bars
    await page.locator('h3:has-text("Performance Benchmark Metrics")').scrollIntoViewIfNeeded();
    await snap(page, '07_compare_all.png');

    // --------------------------------------------------------------------------
    // 08_compare_cs1.png: comparison with context-switch cost 1
    // --------------------------------------------------------------------------
    console.log('[8/14] Capturing 08_compare_cs1.png...');
    const cs1Btn = page.locator('button:has-text("1 tick")');
    if (await cs1Btn.isVisible()) {
      await cs1Btn.click();
    }
    if (await runCompareBtn.isVisible()) {
      await runCompareBtn.click();
    }
    await page.waitForTimeout(800);
    await page.locator('h3:has-text("Performance Benchmark Metrics")').scrollIntoViewIfNeeded();
    await snap(page, '08_compare_cs1.png');

    // --------------------------------------------------------------------------
    // 09_dt_recommendation.png: recommendation card with probabilities
    // --------------------------------------------------------------------------
    console.log('[9/14] Capturing 09_dt_recommendation.png...');
    await page.goto(`${BASE_URL}/ai-lab`, { waitUntil: 'networkidle' });
    await setTheme(page, 'light');

    // Switch to Decision Tree tab
    const dtTabBtn = page.getByRole('button', { name: /decision tree recommender/i });
    if (await dtTabBtn.isVisible()) {
      await dtTabBtn.click();
    }
    await page.waitForSelector('[data-testid="live-recommender"]', { timeout: 15000 });
    await page.waitForTimeout(800);

    // Scroll live recommender into view
    await page.locator('[data-testid="live-recommender"]').scrollIntoViewIfNeeded();
    await snap(page, '09_dt_recommendation.png');

    // --------------------------------------------------------------------------
    // 10_dt_confusion_importance.png: confusion matrix and feature importances
    // --------------------------------------------------------------------------
    console.log('[10/14] Capturing 10_dt_confusion_importance.png...');
    const confusionHeading = page.locator('h3:has-text("Decision Tree Test Confusion Matrix")');
    await confusionHeading.scrollIntoViewIfNeeded();
    await page.waitForTimeout(600);
    await snap(page, '10_dt_confusion_importance.png');

    // --------------------------------------------------------------------------
    // 11_rl_learning_curve.png: learning curve with baseline lines
    // --------------------------------------------------------------------------
    console.log('[11/14] Capturing 11_rl_learning_curve.png...');
    const rlTabBtn = page.getByRole('button', { name: /q-learning tabular agent/i });
    await rlTabBtn.click();
    await page.waitForSelector('[data-testid="learning-curve"]', { timeout: 15000 });
    await page.waitForTimeout(800);

    await page.locator('[data-testid="learning-curve"]').scrollIntoViewIfNeeded();
    await snap(page, '11_rl_learning_curve.png');

    // --------------------------------------------------------------------------
    // 12_rl_policy_table.png: policy table
    // --------------------------------------------------------------------------
    console.log('[12/14] Capturing 12_rl_policy_table.png...');
    await page.waitForSelector('[data-testid="policy-table"]', { timeout: 15000 });
    await page.locator('[data-testid="policy-table"]').scrollIntoViewIfNeeded();
    await page.waitForTimeout(600);
    await snap(page, '12_rl_policy_table.png');

    // --------------------------------------------------------------------------
    // 13_results_summary.png: final test results with CIs
    // --------------------------------------------------------------------------
    console.log('[13/14] Capturing 13_results_summary.png...');
    await page.goto(`${BASE_URL}/results`, { waitUntil: 'networkidle' });
    await setTheme(page, 'light');
    await page.waitForSelector('table', { timeout: 15000 });
    await page.waitForTimeout(800);

    // Scroll so executive summary and canonical summary table are prominent
    await page.locator('[data-testid="summary-table"]').scrollIntoViewIfNeeded();
    await snap(page, '13_results_summary.png');

    // --------------------------------------------------------------------------
    // 14_dark_mode_simulator.png: dark mode capture of Simulator page
    // --------------------------------------------------------------------------
    console.log('[14/14] Capturing 14_dark_mode_simulator.png...');
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });
    await setTheme(page, 'dark');

    // Ensure simulation has executed so Gantt and metrics are populated
    await page.waitForSelector('h3:has-text("CPU Execution Timeline")');
    await page.waitForTimeout(600);
    await page.locator('h3:has-text("CPU Execution Timeline")').scrollIntoViewIfNeeded();
    await snap(page, '14_dark_mode_simulator.png');

    console.log('\n================================================================');
    console.log(' All 14 Checklist Screenshots Successfully Captured!          ');
    console.log('================================================================\n');
  } finally {
    await context.close();
    await browser.close();
  }
}

// Execute runner
captureAllScreenshots()
  .then(() => {
    // Verify all 14 files exist and have non-zero size
    const EXPECTED_FILES = [
      '01_input_workload_editor.png',
      '02_generator_seeded.png',
      '03_gantt_fcfs.png',
      '04_gantt_sjf.png',
      '05_gantt_rr.png',
      '06_gantt_qlearning_trace.png',
      '07_compare_all.png',
      '08_compare_cs1.png',
      '09_dt_recommendation.png',
      '10_dt_confusion_importance.png',
      '11_rl_learning_curve.png',
      '12_rl_policy_table.png',
      '13_results_summary.png',
      '14_dark_mode_simulator.png',
    ];

    let allValid = true;
    for (const f of EXPECTED_FILES) {
      const p = path.join(OUTPUT_DIR, f);
      if (!fs.existsSync(p)) {
        console.error(`[Error] Missing screenshot file: ${f}`);
        allValid = false;
      } else {
        const stats = fs.statSync(p);
        if (stats.size === 0) {
          console.error(`[Error] Screenshot file is empty: ${f}`);
          allValid = false;
        }
      }
    }

    if (allValid) {
      console.log('Verification Complete: All 14 screenshots are valid non-empty PNG files.');
      process.exit(0);
    } else {
      console.error('Verification Failed: Some screenshot files are missing or empty.');
      process.exit(1);
    }
  })
  .catch((err) => {
    console.error('Fatal error during screenshot capture:', err);
    process.exit(1);
  });
