import { defineConfig } from '@playwright/test';

const port = Number(process.env.POLAR_TEST_PORT || 8000);

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined },
  },
  webServer: {
    command: `${process.env.PYTHON || 'python3'} -u tests/browser/server.py`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
  },
});
