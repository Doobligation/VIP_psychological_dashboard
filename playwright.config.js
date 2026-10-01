import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:8000',
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined },
  },
  webServer: {
    command: `${process.env.PYTHON || 'python3'} -u -c 'from serve_dashboard import Handler, ThreadingHTTPServer; ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()'`,
    url: 'http://127.0.0.1:8000',
    reuseExistingServer: !process.env.CI,
  },
});
