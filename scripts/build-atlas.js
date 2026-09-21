#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const os = require('os');
const sharp = require('sharp');

const SRC_DIR = path.join(os.homedir(), 'Downloads', 'gens');
const OUT_DIR = path.join(os.homedir(), 'Downloads', 'gens-view');
const TILE = 64;
const COLS = 32;
const ROWS = 32;
const PER_SHEET = COLS * ROWS;

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const files = fs.readdirSync(SRC_DIR)
    .filter((f) => f.toLowerCase().endsWith('.jpg'))
    .sort();

  if (!files.length) {
    console.error(`No .jpg files in ${SRC_DIR}`);
    process.exit(1);
  }

  const sheets = [];
  for (let i = 0; i < files.length; i += PER_SHEET) {
    const chunk = files.slice(i, i + PER_SHEET);
    const sheetIdx = sheets.length;
    const sheetName = `sheet-${String(sheetIdx).padStart(2, '0')}.jpg`;
    const rowsNeeded = Math.ceil(chunk.length / COLS);
    const width = COLS * TILE;
    const height = rowsNeeded * TILE;

    const composites = chunk.map((name, idx) => ({
      input: path.join(SRC_DIR, name),
      left: (idx % COLS) * TILE,
      top: Math.floor(idx / COLS) * TILE,
    }));

    await sharp({
      create: {
        width,
        height,
        channels: 3,
        background: { r: 17, g: 17, b: 17 },
      },
    })
      .composite(composites)
      .jpeg({ quality: 85, mozjpeg: true })
      .toFile(path.join(OUT_DIR, sheetName));

    sheets.push({
      file: sheetName,
      width,
      height,
      cols: COLS,
      rows: rowsNeeded,
      tiles: chunk,
    });
    console.log(`  wrote ${sheetName}  (${chunk.length} tiles, ${width}x${height})`);
  }

  const manifest = {
    tileSize: TILE,
    cols: COLS,
    rows: ROWS,
    total: files.length,
    sheets,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));

  const html = renderHtml(manifest);
  fs.writeFileSync(path.join(OUT_DIR, 'index.html'), html);

  console.log(`\nDone. ${files.length} tiles across ${sheets.length} sheet(s).`);
  console.log(`Open: ${path.join(OUT_DIR, 'index.html')}`);
}

function renderHtml(m) {
  const sheetsJson = JSON.stringify(m.sheets.map((s) => ({
    file: s.file,
    width: s.width,
    height: s.height,
    cols: s.cols,
    rows: s.rows,
    tiles: s.tiles,
  })));
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Logo atlas (${m.total})</title>
<style>
  :root { --tile: ${m.tileSize}px; --scale: 1.5; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; background: #0b0b0c; color: #ddd; }
  header { position: sticky; top: 0; background: #111; padding: 10px 14px; border-bottom: 1px solid #222; display: flex; gap: 14px; align-items: center; z-index: 10; }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  header .meta { color: #888; }
  header label { color: #aaa; display: inline-flex; gap: 6px; align-items: center; }
  input[type=range] { width: 180px; }
  input[type=search] { background: #1a1a1c; border: 1px solid #2a2a2d; color: #ddd; padding: 5px 8px; border-radius: 4px; min-width: 240px; }
  main { padding: 14px; }
  .sheet { margin-bottom: 22px; }
  .sheet h2 { font-size: 12px; color: #888; font-weight: 500; margin: 0 0 6px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), calc(var(--tile) * var(--scale)));
    gap: 1px;
    background: #1a1a1c;
    padding: 1px;
    width: max-content;
  }
  .tile {
    width: calc(var(--tile) * var(--scale));
    height: calc(var(--tile) * var(--scale));
    background-image: var(--bg);
    background-size: calc(var(--sw) * var(--scale)) calc(var(--sh) * var(--scale));
    background-repeat: no-repeat;
    background-position: calc(var(--x) * var(--scale) * -1) calc(var(--y) * var(--scale) * -1);
    cursor: pointer;
    image-rendering: pixelated;
  }
  .tile:hover { outline: 2px solid #4af; outline-offset: -1px; z-index: 2; position: relative; }
  .tile.dim { opacity: 0.15; }
  #tooltip { position: fixed; pointer-events: none; background: #000; color: #fff; padding: 4px 8px; border: 1px solid #333; border-radius: 4px; font-size: 11px; max-width: 420px; word-break: break-all; display: none; z-index: 100; }
</style>
</head>
<body>
<header>
  <h1>Logo atlas</h1>
  <span class="meta">${m.total} tiles · ${m.sheets.length} sheet(s) · ${m.tileSize}px</span>
  <label>zoom <input type="range" id="zoom" min="1" max="5" step="0.25" value="1.5"><span id="zoomv">1.5x</span></label>
  <input type="search" id="filter" placeholder="filter by filename substring...">
</header>
<main id="main"></main>
<div id="tooltip"></div>
<script>
const SHEETS = ${sheetsJson};
const main = document.getElementById('main');
const tooltip = document.getElementById('tooltip');

for (const s of SHEETS) {
  const wrap = document.createElement('section');
  wrap.className = 'sheet';
  wrap.innerHTML = '<h2>' + s.file + ' · ' + s.tiles.length + ' tiles</h2>';
  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.setProperty('--cols', s.cols);
  grid.style.setProperty('--sw', s.width + 'px');
  grid.style.setProperty('--sh', s.height + 'px');
  grid.style.setProperty('--bg', 'url("' + s.file + '")');
  for (let i = 0; i < s.tiles.length; i++) {
    const x = (i % s.cols) * ${m.tileSize};
    const y = Math.floor(i / s.cols) * ${m.tileSize};
    const t = document.createElement('div');
    t.className = 'tile';
    t.style.setProperty('--x', x + 'px');
    t.style.setProperty('--y', y + 'px');
    t.dataset.name = s.tiles[i];
    grid.appendChild(t);
  }
  wrap.appendChild(grid);
  main.appendChild(wrap);
}

main.addEventListener('mousemove', (e) => {
  const t = e.target.closest('.tile');
  if (!t) { tooltip.style.display = 'none'; return; }
  tooltip.textContent = t.dataset.name;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
});
main.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
main.addEventListener('click', (e) => {
  const t = e.target.closest('.tile');
  if (!t) return;
  navigator.clipboard?.writeText(t.dataset.name);
});

const zoom = document.getElementById('zoom');
const zoomv = document.getElementById('zoomv');
zoom.addEventListener('input', () => {
  document.documentElement.style.setProperty('--scale', zoom.value);
  zoomv.textContent = zoom.value + 'x';
});

const filter = document.getElementById('filter');
filter.addEventListener('input', () => {
  const q = filter.value.trim().toLowerCase();
  for (const t of document.querySelectorAll('.tile')) {
    t.classList.toggle('dim', q && !t.dataset.name.toLowerCase().includes(q));
  }
});
</script>
</body>
</html>
`;
}

main().catch((err) => { console.error(err); process.exit(1); });
