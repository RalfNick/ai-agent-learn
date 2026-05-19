import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import puppeteer from "/Users/bytedance/ClaudeCode-Projects/wechat-article/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js";

const inputPath = "docs/phase-3/00-langchain-to-langgraph-foundations-wechat.md";
const outDir = "docs/phase-3/diagram/wechat-mermaid";
const mermaidJsPath = "/Users/bytedance/ClaudeCode-Projects/wechat-article/node_modules/mermaid/dist/mermaid.min.js";

const markdown = readFileSync(inputPath, "utf8");
const mermaidBlocks = [...markdown.matchAll(/```mermaid\n([\s\S]*?)\n```/g)].map((match) =>
  match[1].trim(),
);

if (mermaidBlocks.length === 0) {
  console.log("No mermaid blocks found.");
  process.exit(0);
}

mkdirSync(outDir, { recursive: true });

const mermaidJs = readFileSync(mermaidJsPath, "utf8");
const browser = await puppeteer.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
const page = await browser.newPage();

for (let index = 0; index < mermaidBlocks.length; index += 1) {
  const id = String(index + 1).padStart(2, "0");
  const source = mermaidBlocks[index];
  const html = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body {
        margin: 0;
        padding: 24px;
        background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .wrap {
        display: inline-block;
        background: #ffffff;
      }
    </style>
  </head>
  <body>
    <div class="wrap"><div class="mermaid">${source.replaceAll("&", "&amp;").replaceAll("<", "&lt;")}</div></div>
    <script>${mermaidJs}</script>
    <script>
      mermaid.initialize({ startOnLoad: true, theme: "default", securityLevel: "loose" });
    </script>
  </body>
</html>`;

  await page.setContent(html, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => document.querySelectorAll(".mermaid svg").length === 1, {
    timeout: 30000,
  });

  const element = await page.$(".wrap");
  const outputPath = join(outDir, `flow-${id}.png`);
  await element.screenshot({ path: outputPath, omitBackground: false });
  console.log(outputPath);
}

await browser.close();
