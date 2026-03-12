#!/usr/bin/env node
/**
 * gen-graph-triples.mjs — Extract entity-relationship triples from knowledge chunks.
 *
 * Implements DRAG Step 3: Graph RAG Generation.
 * For each chunk, prompts Groq (Llama 3.3 70B) to extract (subject, relation, object)
 * triples. Output is stored as a sidecar JSON file used by:
 *   1. gen-drag-data.mjs  — includes triples in GKD training examples
 *   2. chat-engine.ts     — injects relevant triples into the RAG system prompt
 *
 * Usage:
 *   GROQ_API_KEY=xxx node scripts/gen-graph-triples.mjs
 *   GROQ_API_KEY=xxx node scripts/gen-graph-triples.mjs --dry-run
 *
 * Output: scripts/finetune/knowledge_triples.json
 *   { "chunk-0": [{ subject, relation, object }, ...], ... }
 */

import { createDecipheriv } from 'crypto';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const TEACHER_MODEL = 'llama-3.3-70b-versatile';
const OUTPUT_FILE = join(ROOT, 'scripts', 'finetune', 'knowledge_triples.json');
const MAX_TRIPLES_PER_CHUNK = 12;
const MAX_RETRIES = 3;

const args = process.argv.slice(2);
const isDryRun = args.includes('--dry-run');

// ── Load .env.local ──────────────────────────────────────────────────────────
const envPath = join(ROOT, '.env.local');
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, 'utf8').split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('#')) continue;
    const eq = t.indexOf('=');
    if (eq < 0) continue;
    const k = t.slice(0, eq).trim(), v = t.slice(eq + 1).trim();
    if (!process.env[k]) process.env[k] = v;
  }
}

const colors = {
  reset: '\x1b[0m', green: '\x1b[32m', yellow: '\x1b[33m',
  red: '\x1b[31m', cyan: '\x1b[36m', bold: '\x1b[1m',
};
const log = (msg, c = colors.reset) => console.log(`${c}${msg}${colors.reset}`);

// ── Decrypt helper ───────────────────────────────────────────────────────────
function decryptFile(filePath) {
  const KEY_HEX = process.env.VITE_CHUNKS_KEY;
  if (!KEY_HEX) throw new Error('VITE_CHUNKS_KEY not set');
  const buf = Buffer.from(KEY_HEX, 'hex');
  const { iv, data } = JSON.parse(readFileSync(filePath, 'utf8'));
  const raw = Buffer.from(data, 'base64');
  const dec = createDecipheriv('aes-256-gcm', buf, Buffer.from(iv, 'base64'));
  dec.setAuthTag(raw.slice(raw.length - 16));
  return dec.update(raw.slice(0, raw.length - 16), '', 'utf8') + dec.final('utf8');
}

// ── Load all chunks ──────────────────────────────────────────────────────────
function loadAllChunks() {
  const kcRaw = JSON.parse(decryptFile(join(ROOT, 'public/data/knowledge_chunks.enc.json')));
  const ccRaw = JSON.parse(decryptFile(join(ROOT, 'public/data/company_chunks.enc.json')));
  const knowledge = Array.isArray(kcRaw) ? kcRaw : [];
  const company   = Array.isArray(ccRaw.chunks) ? ccRaw.chunks : (Array.isArray(ccRaw) ? ccRaw : []);
  return [...knowledge, ...company];
}

// ── Triple extraction prompt ─────────────────────────────────────────────────
function buildPrompt(chunk) {
  return `You are a knowledge graph extractor. Extract structured entity-relationship triples from the following text about Kham.

Text:
"""
${chunk.text}
"""

Rules:
- Extract up to ${MAX_TRIPLES_PER_CHUNK} distinct triples
- Subject and object should be specific entities (people, organizations, technologies, projects, skills, numbers)
- Relation should be a concise verb phrase (e.g., "developed", "employed_at", "improved_by", "has_skill", "achieved", "used_technology")
- Every triple must be directly supported by the text — no inference
- "Kham" refers to Kham

Respond with JSON only (no markdown):
{
  "triples": [
    { "subject": "Kham", "relation": "developed", "object": "power-grid digital twin" },
    ...
  ]
}`;
}

// ── Groq API call ────────────────────────────────────────────────────────────
async function callGroq(prompt, retries = MAX_RETRIES) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(GROQ_API_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: TEACHER_MODEL,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.2, // Low temp for factual extraction
          response_format: { type: 'json_object' },
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const data = await resp.json();
      return data.choices[0].message.content;
    } catch (err) {
      if (attempt === retries - 1) throw err;
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
    }
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  log(`\n${'─'.repeat(60)}`, colors.cyan);
  log(`  DRAG Graph Triple Extractor`, colors.bold);
  log(`${'─'.repeat(60)}\n`, colors.cyan);

  if (!process.env.GROQ_API_KEY && !isDryRun) {
    log('❌  GROQ_API_KEY not set', colors.red);
    process.exit(1);
  }

  const chunks = loadAllChunks();
  log(`Loaded ${chunks.length} chunks (knowledge + company)`, colors.green);

  // Load existing output to support re-runs without re-processing
  const existing = existsSync(OUTPUT_FILE) ? JSON.parse(readFileSync(OUTPUT_FILE, 'utf8')) : {};
  const alreadyDone = new Set(Object.keys(existing));
  const toProcess = chunks.filter(c => !alreadyDone.has(c.id));
  log(`Already processed: ${alreadyDone.size} | Remaining: ${toProcess.length}\n`, colors.yellow);

  if (isDryRun) {
    const sample = chunks[0];
    log(`[DRY RUN] Sample prompt for chunk: ${sample.id}`, colors.yellow);
    console.log(buildPrompt(sample).substring(0, 400) + '...');
    log('\nMock triples:', colors.yellow);
    console.log(JSON.stringify({
      triples: [
        { subject: 'Kham', relation: 'has_skill', object: '<extracted from chunk>' },
        { subject: 'Kham', relation: 'worked_on', object: '<extracted from chunk>' },
      ]
    }, null, 2));
    return;
  }

  const results = { ...existing };
  let success = 0, failed = 0;

  for (let i = 0; i < toProcess.length; i++) {
    const chunk = toProcess[i];
    process.stdout.write(`  [${i + 1}/${toProcess.length}] ${chunk.id}... `);

    try {
      const raw = await callGroq(buildPrompt(chunk));
      const parsed = JSON.parse(raw.match(/\{[\s\S]*\}/)?.[0] ?? raw);
      const triples = (parsed.triples ?? [])
        .filter(t => t.subject && t.relation && t.object)
        .slice(0, MAX_TRIPLES_PER_CHUNK);

      results[chunk.id] = triples;
      // Save after each chunk — safe to interrupt
      writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));

      process.stdout.write(`${colors.green}${triples.length} triples saved${colors.reset}\n`);
      success++;
    } catch (err) {
      process.stdout.write(`${colors.red}FAILED: ${err.message}${colors.reset}\n`);
      failed++;
    }

    // Rate-limit: 300ms between requests
    await new Promise(r => setTimeout(r, 300));
  }

  log(`\n✅  Done: ${success} chunks processed, ${failed} failed`, colors.green);
  log(`📄  Output: ${OUTPUT_FILE}`, colors.cyan);

  // Summary stats
  const totalTriples = Object.values(results).reduce((s, t) => s + t.length, 0);
  log(`📊  Total triples: ${totalTriples} across ${Object.keys(results).length} chunks`, colors.cyan);
}

main().catch(err => { console.error(err); process.exit(1); });
