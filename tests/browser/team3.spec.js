import { test, expect } from '@playwright/test';

// A worn Empatica batch session posted to the real local API, followed from the session picker.
async function seedBatchSession(request, id, seconds = 330, contact = undefined) {
  const start = Math.floor(Date.now() / 1000) - 7200;
  for (let i = 0; i < seconds; i++) {
    const raw = { source: 'empatica-avro', eda_us: 2 + Math.sin(i / 7) / 10, temp_c: 33, hr: 75, rr_intervals_ms: [800 + (i % 3)] };
    if (contact !== undefined) raw.contact = contact;
    const response = await request.post('/samples', { data: {
      session_id: id, participant: 'empatica-test', mode: 'batch', sample_id: String(i), timestamp_ms: (start + i) * 1000, raw } });
    expect(response.ok()).toBeTruthy();
  }
}

test('any saved session can be picked and followed without typing a URL', async ({ page, request }) => {
  await seedBatchSession(request, 'team3-picker-worn');
  await page.goto('/serving/dashboard.html');
  await expect(page.locator('#follow')).toHaveText('Follow session');
  const picker = page.locator('#session-picker');
  await picker.focus();
  await expect(picker.locator('option[value="team3-picker-worn"]')).toHaveCount(1);
  await picker.selectOption('team3-picker-worn');
  await page.locator('#follow').click();
  await expect(page.locator('#mode-badge')).toHaveText('DELAYED · EMPATICA BATCH');
  await expect(page.locator('#session-label')).toContainText('team3-picker-worn');
  await expect(page).toHaveURL(/\?session=team3-picker-worn$/);
  // One timeline cell per prediction (every 30 s), not one per reading.
  const cells = page.locator('#timeline .tl-cells[role=list]').first().locator('.cell:not(.empty)');
  await expect(cells.first()).toBeVisible();
  expect(await cells.count()).toBeLessThanOrEqual(12);
  await expect(page.locator('#inputs')).toContainText('of beats clean');
  await page.locator('#follow').click();
  await expect(page.locator('#follow')).toHaveText('Follow session');
  await expect(page).toHaveURL(/dashboard\.html$/);
});

test('a session link opens directly and an unworn device explains itself', async ({ page, request }) => {
  await seedBatchSession(request, 'team3-picker-unworn', 80, false);
  await page.goto('/serving/dashboard.html?session=team3-picker-unworn');
  await expect(page.locator('#phase-note')).toContainText('not worn');
  await expect(page.locator('#inputs')).toContainText('Device not worn');
  await expect(page.locator('#card-frustration')).toContainText('not worn');
  await expect(page.locator('#card-frustration .pill')).toContainText('UNAVAILABLE');
  await expect(page.locator('#alert-list')).toContainText('None yet');
});

test('following with nothing picked asks for a session instead of assuming Polar', async ({ page }) => {
  await page.goto('/serving/dashboard.html');
  await expect(page.locator('#follow')).toBeEnabled();
  await page.locator('#follow').click();
  await expect(page.locator('#phase-title')).toHaveText('Choose a session');
});

test('the model card leads with accuracy on new people', async ({ page }) => {
  await page.goto('/serving/dashboard.html');
  await page.locator('#ml-details summary').click();
  await expect(page.locator('#ml-report strong').first()).toContainText('new people');
});

test('a High prediction asks a Quick check; answers and self-reports are saved and scored', async ({ page }) => {
  await page.goto('/serving/dashboard.html');
  await expect(page.locator('#feel')).toBeDisabled();
  await page.locator('#speed').selectOption('500');
  await page.locator('#run').click();
  await expect(page.locator('#checkin-title')).toHaveText('Quick check', { timeout: 30000 });
  await page.locator('#pause').click();
  await expect(page.locator('#checkin-intro')).toContainText('Frustration predicted High');
  await expect(page.locator('#checkin-clock')).toContainText('Closes in');
  await expect(page.locator('#checkin-cause option:checked')).toContainText('(predicted)');
  const send = page.locator('#checkin-send');
  await expect(send).toBeDisabled();
  await page.locator('#checkin [data-state=frustration][data-value=yes]').click();
  await expect(send).toBeDisabled();  // both questions are required
  await page.locator('#checkin [data-state=fatigue][data-value=no]').click();
  await page.locator('#checkin-note').fill('Deadline moved up');
  await send.click();
  await expect(page.locator('#checkin')).toContainText('Your answer is saved');
  await expect(page.locator('#checkin-table')).toContainText('Deadline moved up');
  await expect(page.locator('#checkin-table')).toContainText('Agrees');
  await expect(page.locator('#timeline .mark', { hasText: '?' })).toHaveCount(1);

  await page.locator('#feel').click();
  await expect(page.locator('#checkin-title')).toHaveText('How do you feel?');
  await page.locator('#checkin [data-state=frustration][data-value=no]').click();
  await page.locator('#checkin [data-state=fatigue][data-value=a-little]').click();
  await page.locator('#checkin-send').click();
  await expect(page.locator('#checkin-table')).toContainText('Self-report');
  const stats = page.locator('#checkin-stats .stat');
  await expect(stats.nth(1)).toContainText('1');  // answered
  await expect(stats.nth(4)).toContainText('1');  // self-reports
  await expect(page.locator('#agreement-table')).toContainText('Scripted demo · DBR rules (demo)');
  await expect(page.locator('#checkin-files')).toContainText('checkins.csv');
});
