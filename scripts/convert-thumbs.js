#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const os = require('os');
const sharp = require('sharp');

const SRC_DIR = path.join(__dirname, '..', 'output', 'generated');
const DEST_DIR = path.join(os.homedir(), 'Downloads', 'gens');
const SIZE = 64;
const QUALITY = 80;
const CONCURRENCY = 8;
const PAD = 5;
const PREFIX_RE = /^(\d+)_(.+)\.jpg$/i;

async function main() {
  fs.mkdirSync(DEST_DIR, { recursive: true });

  const existingBases = new Set();
  let maxNum = 0;
  for (const f of fs.readdirSync(DEST_DIR)) {
    if (!f.toLowerCase().endsWith('.jpg')) continue;
    const m = f.match(PREFIX_RE);
    if (m) {
      maxNum = Math.max(maxNum, parseInt(m[1], 10));
      existingBases.add(m[2]);
    } else {
      existingBases.add(f.replace(/\.jpg$/i, ''));
    }
  }

  const pending = [];
  for (const name of fs.readdirSync(SRC_DIR)) {
    if (!/\.(png|jpe?g|webp)$/i.test(name)) continue;
    const base = name.replace(/\.[^.]+$/, '');
    if (existingBases.has(base)) continue;
    const stat = fs.statSync(path.join(SRC_DIR, name));
    pending.push({ name, base, ts: stat.birthtimeMs });
  }
  pending.sort((a, b) => a.ts - b.ts || a.base.localeCompare(b.base));

  const pad = Math.max(PAD, String(maxNum + pending.length).length);
  const tasks = pending.map((p, i) => ({
    src: path.join(SRC_DIR, p.name),
    dest: path.join(DEST_DIR, `${String(maxNum + i + 1).padStart(pad, '0')}_${p.base}.jpg`),
  }));

  console.log(`${existingBases.size} already done, ${tasks.length} new to convert`);

  let done = 0, failed = 0;
  const workers = Array.from({ length: CONCURRENCY }, async () => {
    while (tasks.length) {
      const job = tasks.shift();
      if (!job) break;
      try {
        await sharp(job.src)
          .resize(SIZE, SIZE, { fit: 'cover' })
          .jpeg({ quality: QUALITY, mozjpeg: true })
          .toFile(job.dest);
      } catch (err) {
        failed++;
        console.error(`FAIL ${path.basename(job.src)}: ${err.message}`);
      }
      done++;
      if (done % 100 === 0) console.log(`  ...${done}`);
    }
  });
  await Promise.all(workers);
  console.log(`Done. Converted ${done - failed}, failed ${failed}.`);
}

main().catch((err) => { console.error(err); process.exit(1); });
