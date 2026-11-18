/**
 * @file tailwind.config.ts
 * 1. What this file does:
 *    Defines the Tailwind CSS theme configuration, token mappings, and content paths.
 * 2. Overall design:
 *    Integrates semantic CSS variables (--background, --surface, --primary, etc.)
 *    for automatic light and dark mode switching via the 'class' strategy.
 * 3. Elements in the file:
 *    - content path matchers for TSX/HTML
 *    - darkMode configuration set to 'class'
 *    - color token mappings referencing CSS variables
 *    - typography families (Inter for sans, JetBrains Mono for mono)
 *    - border radius tokens (10px cards, 8px controls)
 * 4. Final output of the file:
 *    Tailwind CSS configuration object consumed by the Vite build pipeline.
 */

import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        surface: {
          DEFAULT: 'var(--surface)',
          muted: 'var(--surface-muted)',
        },
        border: 'var(--border)',
        foreground: {
          DEFAULT: 'var(--text)',
          muted: 'var(--text-muted)',
        },
        primary: {
          DEFAULT: 'var(--primary)',
          hover: 'var(--primary-hover)',
        },
        success: 'var(--success)',
        warning: 'var(--warning)',
        danger: 'var(--danger)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        card: '10px',
        control: '8px',
      },
      boxShadow: {
        subtle: '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [],
};

export default config;
