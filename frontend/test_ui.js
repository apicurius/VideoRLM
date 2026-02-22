const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto('http://localhost:4000', { waitUntil: 'networkidle0' });
  await page.click('header button:first-of-type'); // Settings button
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: 'settings_ui.png' });
  await browser.close();
})();
