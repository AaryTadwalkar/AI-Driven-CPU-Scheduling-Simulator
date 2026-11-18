/**
 * @file postcss.config.js
 * 1. What this file does: Configures PostCSS plugins for Tailwind CSS and Autoprefixer.
 * 2. Overall design: Standard PostCSS pipeline passing CSS AST through Tailwind engine and vendor prefixer.
 * 3. Elements in the file: plugins object with tailwindcss and autoprefixer configurations.
 * 4. Final output: CSS transformation pipeline for Vite build and development runtime.
 */

export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
