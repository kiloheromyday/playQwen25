const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = 'http://127.0.0.1:8765';
const OUT_DIR = path.resolve(__dirname, '../artifacts/ui_smoke');
const results = [];

function record(name, ok, detail) {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

async function waitReady(page) {
  await page.waitForFunction(
    () => document.getElementById('connection')?.textContent === '模型已就绪',
    null, { timeout: 15000 }
  );
}

async function chatFlow(page) {
  await page.waitForSelector('#chat-empty', { state: 'visible' });
  await page.click('button[data-prompt="预训练通常包括"]');
  await page.click('#send-button');
  await page.waitForSelector('.message.assistant', { timeout: 20000 });
  await page.waitForFunction(
    () => [...document.querySelectorAll('.message.assistant')].some(
      (node) => !node.classList.contains('pending') && node.querySelector('.message-text')?.textContent.trim()
    ),
    null, { timeout: 20000 }
  );
  const text = await page.textContent('.message.assistant .message-text');
  const meta = await page.textContent('.message-meta') || '';
  record('chat-send-and-reply', text.trim().length > 0, `${text.trim().length} chars, ${meta.trim()}`);
  record('chat-metrics-filled', /token\/s/.test(meta), 'latency/throughput shown after reply');
}

async function evaluationFlow(page) {
  await page.click('[data-view="evaluation"]');
  await page.waitForSelector('#metric-grid .metric', { state: 'visible', timeout: 15000 });
  const metrics = await page.$$eval('#metric-grid .metric', (nodes) =>
    nodes.map((n) => n.querySelector('.metric-label')?.textContent.trim())
  );
  record('evaluation-metrics-rendered', metrics.length === 4, metrics.join(' | '));
  const rows = await page.$$eval('#score-table tr', (nodes) => nodes.length);
  record('evaluation-table-rendered', rows >= 4, `${rows} rows`);
  const probes = await page.$$eval('#probe-results .probe', (nodes) => nodes.length);
  record('evaluation-probes-rendered', probes >= 2, `${probes} probes`);
}

async function trainingFlow(page) {
  await page.click('[data-view="training"]');
  await page.waitForSelector('#training-canvas', { state: 'visible', timeout: 15000 });
  const painted = await page.waitForFunction(() => {
    const canvas = document.getElementById('training-canvas');
    if (!canvas || canvas.width < 1 || canvas.height < 1) return false;
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted += 1;
    return painted > 500 ? painted : false;
  }, null, { timeout: 5000 }).then((handle) => handle.jsonValue());
  record('training-chart-drawn', painted > 500, `${painted} painted pixels`);
  const bestRow = await page.textContent('.training-finding strong') || '';
  record('training-finding-rendered', /过拟合/.test(bestRow), bestRow);
}

async function runViewport(browser, name, width, height) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(20000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await waitReady(page);
  record(`${name}-page-load`, errors.length === 0, errors.join('; ') || 'no console/page errors');
  await chatFlow(page);
  await evaluationFlow(page);
  await trainingFlow(page);
  const shot = path.join(OUT_DIR, `${name}.png`);
  await page.screenshot({ path: shot, fullPage: true });
  record(`${name}-screenshot`, fs.existsSync(shot), path.relative(process.cwd(), shot));
  await context.close();
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  await runViewport(browser, 'desktop', 1440, 900);
  await runViewport(browser, 'mobile', 390, 844);
  await browser.close();
  const failed = results.filter((item) => !item.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`FATAL  ${error.message}`);
  process.exitCode = 1;
});
