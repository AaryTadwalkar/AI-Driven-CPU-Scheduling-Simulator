/**
 * @file src/main.tsx
 * 1. What this file does:
 *    Main entrypoint that bootstraps the React application and mounts it to the DOM root element.
 * 2. Overall design:
 *    Imports global CSS stylesheets and mounts the App component inside React.StrictMode.
 * 3. Elements in the file:
 *    - DOM root mount invocation
 * 4. Final output of the file:
 *    Browser DOM hydration and application execution startup.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element with id "root"');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
