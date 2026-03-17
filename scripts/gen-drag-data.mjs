#!/usr/bin/env node
/**
 * gen-drag-data.mjs — Generate DRAG-format GKD training examples.
 *
 * Implements the full DRAG training data pipeline:
 *   1. For each chunk, retrieve related chunks (cosine similarity)
 *   2. Load pre-extracted graph triples for those chunks
 *   3. Build a DRAG-style system prompt: evidence paragraphs + relationship triples
 *   4. Prompt Groq (teacher) to answer a diverse question given that context
 *   5. Save ChatML example with system/user/assistant messages
 *
 * The resulting training examples teach the student to:
 *   - Read and reason over structured evidence + graph triples
 *   - Give grounded, concise answers — not hallucinate
 *
 * This format matches exactly what chat-engine.ts will inject at inference time,
 * so the student learns the same prompt structure it will see in production.
 *
 * Usage:
 *   GROQ_API_KEY=xxx node scripts/gen-drag-data.mjs
 *   GROQ_API_KEY=xxx node scripts/gen-drag-data.mjs --dry-run
 *   GROQ_API_KEY=xxx node scripts/gen-drag-data.mjs --resume
 *
 * Output: scripts/finetune/training_data_drag.jsonl
 *   ~400 new examples (16 per chunk × 25 chunks)
 *   Combined with training_data.jsonl → ~500 total for GKD
 *
 * Prerequisites:
 *   Run gen-graph-triples.mjs first to produce knowledge_triples.json
 */

import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const TEACHER_MODEL = 'llama-3.3-70b-versatile';
const OUTPUT_DIR = join(ROOT, 'scripts', 'finetune');
const OUTPUT_FILE = join(OUTPUT_DIR, 'training_data_drag.jsonl');
const TRIPLES_FILE = join(OUTPUT_DIR, 'knowledge_triples.json');
const GROUNDING_THRESHOLD = 0.85;
const PAIRS_PER_CHUNK = 16;  // 25 chunks × 16 = 400 examples
const TOP_K_EVIDENCE = 3;   // How many chunks to include as evidence context
const TOP_K_TRIPLES = 10;  // Max triples to include in system prompt
const MAX_RETRIES = 3;

const args = process.argv.slice(2);
const isDryRun = args.includes('--dry-run');
const isResume = args.includes('--resume');

const colors = {
  reset: '\x1b[0m', green: '\x1b[32m', yellow: '\x1b[33m',
  red: '\x1b[31m', cyan: '\x1b[36m', bold: '\x1b[1m',
};
const log = (msg, c = colors.reset) => console.log(`${c}${msg}${colors.reset}`);

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

// ── Load all chunks ─────────────────────────────────────────────────────────
// NOTE: In this public version, chunks are expected as plain JSON files.
// Provide knowledge_chunks.json and company_chunks.json in public/data/.
function loadAllChunks() {
  const kcRaw = JSON.parse(readFileSync(join(ROOT, 'public/data/knowledge_chunks.json'), 'utf8'));
  const ccRaw = JSON.parse(readFileSync(join(ROOT, 'public/data/company_chunks.json'), 'utf8'));
  const knowledge = Array.isArray(kcRaw) ? kcRaw : [];
  const company   = Array.isArray(ccRaw.chunks) ? ccRaw.chunks : (Array.isArray(ccRaw) ? ccRaw : []);
  return [...knowledge, ...company];
}

// ── Cosine similarity for chunk selection ────────────────────────────────────
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

/** Return the topK most similar chunks to the anchor (excluding itself). */
function getRelatedChunks(anchor, allChunks, k = TOP_K_EVIDENCE - 1) {
  if (!anchor.embedding) return [];
  return allChunks
    .filter(c => c.id !== anchor.id && c.embedding)
    .map(c => ({ chunk: c, score: cosine(anchor.embedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(x => x.chunk);
}

// ── Build DRAG system prompt ─────────────────────────────────────────────────
function buildDragSystemPrompt(evidenceChunks, triples) {
  const evidenceLines = evidenceChunks
    .map((c, i) => `${i + 1}. ${c.text.trim()}`)
    .join('\n\n');

  const tripleLines = triples
    .slice(0, TOP_K_TRIPLES)
    .map(t => `- ${t.subject} → ${t.relation} → ${t.object}`)
    .join('\n');

  return [
    'You are an AI assistant for Kham\'s professional portfolio.',
    '',
    '## Evidence',
    evidenceLines,
    '',
    tripleLines ? '## Key Relationships' : '',
    tripleLines || '',
    '',
    'Answer questions about Kham based strictly on the evidence above.',
    'Be concise, factual, and specific. Do not invent information.',
  ].filter(l => l !== undefined).join('\n').trim();
}

// ── Question type instructions ───────────────────────────────────────────────
const QUESTION_TYPES = [
  'factual',
  'behavioral',
  'technical',
  'metrics',
  'motivation',
  'comparison',
  'depth',
  'scenario',
  'skills',
  'impact',
  'multi_chunk',   // Synthesizes across multiple chunks
  'elaboration',   // Provide a rich, detailed explanation with evidence
  'recruiter',     // Recruiter / hiring manager framing
  'fit',           // Job fit / candidate evaluation
  'gap',           // What Kham might lack or how he'd grow
  'followup',      // Follows up on a previous answer
];

const TYPE_INSTRUCTIONS = {
  factual: 'Ask a specific factual question answerable from the evidence (e.g., "What did Kham do at X?" or "What technology did Kham use for Y?").',
  behavioral: 'Write a behavioral interview question starting with "Tell me about a time..." or "Describe a situation where..."',
  technical: 'Ask a technical deep-dive question about the methodology, algorithm, architecture, or engineering approach in the evidence.',
  metrics: 'Ask about measurable outcomes, numbers, percentages, timelines, or quantified impact mentioned in the evidence.',
  motivation: 'Ask why a particular approach, technology, or decision was made — probe reasoning and judgment.',
  comparison: 'Ask how one approach, tool, result, or skill in the evidence compares to another or to industry standards.',
  depth: 'Ask for a detailed explanation of a specific concept, project, or technique mentioned in the evidence.',
  scenario: 'Pose a realistic work scenario and ask how Kham would approach it based on his demonstrated experience.',
  skills: 'Ask what specific technical skills, tools, languages, or domain knowledge were required or demonstrated.',
  impact: 'Ask about the business value, strategic importance, or measurable real-world impact of the work described.',
  multi_chunk: 'Ask a question that requires synthesizing information across MULTIPLE evidence sections (e.g., linking skills to a specific project outcome).',
  elaboration: 'Ask for a thorough, detailed explanation of a project or accomplishment. Expect a rich, evidence-backed answer with specific names, metrics, and technologies.',
  recruiter: 'Frame the question from a recruiter\'s perspective evaluating Kham for a senior role (e.g., "Would Kham be a good fit for a team that needs X?").',
  fit: 'Ask how Kham\'s background fits a specific context, role, or requirement implied by the evidence.',
  gap: 'Ask a thoughtful question about where Kham might grow further or what adjacent skills he could develop.',
  followup: 'Write a realistic follow-up question someone might ask after an initial answer about the topic in the evidence.',
};

// ── Generate one DRAG-format training example ────────────────────────────────
async function generateExample(primaryChunk, relatedChunks, allTriples, questionType, mock) {
  const evidenceChunks = [primaryChunk, ...relatedChunks];
  const chunkIds = evidenceChunks.map(c => c.id);
  const triples = chunkIds.flatMap(id => allTriples[id] ?? []);

  const systemContent = buildDragSystemPrompt(evidenceChunks, triples);
  const instruction = TYPE_INSTRUCTIONS[questionType] ?? TYPE_INSTRUCTIONS.factual;

  if (mock) {
    return {
      messages: [
        { role: 'system', content: systemContent.substring(0, 200) + '...' },
        { role: 'user', content: `[MOCK ${questionType}] What did Kham accomplish?` },
        { role: 'assistant', content: 'Mock answer based on context.' },
      ],
      meta: { sourceChunkIds: chunkIds, groundingScore: 0.95, generatedBy: 'mock', questionType },
    };
  }

  const prompt = `You are creating high-quality training data for a personal AI assistant about Kham.

System context that will be provided to the AI (evidence + graph triples):
${systemContent}

Task:
1. Write a "${questionType.toUpperCase()}" question: ${instruction}
2. The question must be answerable using ONLY the system context above.
3. Write a thorough, specific answer grounded strictly in the context. Do not invent facts.
4. Write a detailed, substantive answer of 4–8 sentences. Always cite specific evidence from the context (names, projects, metrics, technologies). Explain WHY the facts matter — connect them to qualifications, impact, and capabilities. The answer should sound like a knowledgeable colleague enthusiastically describing Kham's work, not a bullet-point summary.
5. Rate how well the answer is grounded (0.0–1.0).

Respond with JSON only (no markdown):
{
  "question": "...",
  "answer": "...",
  "grounding_score": 0.0
}`;

  const raw = await callGroq(prompt);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) return null;
  const parsed = JSON.parse(match[0]);
  if (!parsed?.question || !parsed?.answer) return null;

  return {
    messages: [
      { role: 'system', content: systemContent },
      { role: 'user', content: parsed.question },
      { role: 'assistant', content: parsed.answer },
    ],
    meta: {
      sourceChunkIds: chunkIds,
      groundingScore: parsed.grounding_score ?? 0,
      generatedBy: TEACHER_MODEL,
      questionType,
      dragFormat: true,
    },
  };
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
          temperature: 0.85,   // higher diversity → less repetitive training labels
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
  log(`  DRAG Training Data Generator`, colors.bold);
  log(`${'─'.repeat(60)}\n`, colors.cyan);

  if (!process.env.GROQ_API_KEY && !isDryRun) {
    log('❌  GROQ_API_KEY not set', colors.red);
    process.exit(1);
  }

  if (!existsSync(TRIPLES_FILE) && !isDryRun) {
    log('❌  knowledge_triples.json not found. Run gen-graph-triples.mjs first.', colors.red);
    process.exit(1);
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const allChunks = loadAllChunks();
  const allTriples = existsSync(TRIPLES_FILE)
    ? JSON.parse(readFileSync(TRIPLES_FILE, 'utf8'))
    : {};

  log(`Chunks loaded: ${allChunks.length}`, colors.green);
  log(`Chunks with triples: ${Object.keys(allTriples).length}`, colors.green);
  log(`Target examples: ${allChunks.length} chunks × ${PAIRS_PER_CHUNK} pairs = ${allChunks.length * PAIRS_PER_CHUNK}`, colors.cyan);

  // Resume: skip chunks already fully processed
  const processedChunkIds = new Set();
  const processedCounts = {};
  if (isResume && existsSync(OUTPUT_FILE)) {
    for (const line of readFileSync(OUTPUT_FILE, 'utf8').split('\n').filter(Boolean)) {
      try {
        const entry = JSON.parse(line);
        const primaryId = entry.meta?.sourceChunkIds?.[0];
        if (primaryId) {
          processedCounts[primaryId] = (processedCounts[primaryId] ?? 0) + 1;
        }
      } catch { /* skip */ }
    }
    for (const [id, count] of Object.entries(processedCounts)) {
      if (count >= PAIRS_PER_CHUNK) processedChunkIds.add(id);
    }
    log(`Resuming: ${processedChunkIds.size} chunks fully done, ${Object.keys(processedCounts).length - processedChunkIds.size} partially done`, colors.yellow);
  }

  const chunksToProcess = allChunks.filter(c => !processedChunkIds.has(c.id));
  log(`Processing: ${chunksToProcess.length} chunks\n`, colors.cyan);

  if (isDryRun) {
    log(`[DRY RUN] Sample for chunk: ${allChunks[0].id}`, colors.yellow);
    const related = getRelatedChunks(allChunks[0], allChunks);
    const example = await generateExample(allChunks[0], related, allTriples, 'factual', true);
    console.log(JSON.stringify(example, null, 2));
    log('\nSystem prompt preview:', colors.yellow);
    const sys = example.messages[0].content;
    console.log(sys.substring(0, 600) + (sys.length > 600 ? '...' : ''));
    return;
  }

  let success = 0, skipped = 0, failed = 0;

  for (let i = 0; i < chunksToProcess.length; i++) {
    const chunk = chunksToProcess[i];
    const alreadyCount = processedCounts[chunk.id] ?? 0;
    const remaining = PAIRS_PER_CHUNK - alreadyCount;
    log(`\nChunk ${i + 1}/${chunksToProcess.length}: ${chunk.id} (${remaining} pairs needed)`, colors.cyan);

    const relatedChunks = getRelatedChunks(chunk, allChunks);
    const typesNeeded = QUESTION_TYPES.slice(0, remaining);

    for (let p = 0; p < typesNeeded.length; p++) {
      const qType = typesNeeded[p];
      process.stdout.write(`  pair ${alreadyCount + p + 1}/${PAIRS_PER_CHUNK} [${qType}]... `);

      try {
        const result = await generateExample(chunk, relatedChunks, allTriples, qType, false);
        if (!result) {
          process.stdout.write(`${colors.red}null response${colors.reset}\n`);
          failed++;
          continue;
        }
        if (result.meta.groundingScore < GROUNDING_THRESHOLD) {
          process.stdout.write(`${colors.yellow}skipped (grounding: ${result.meta.groundingScore})${colors.reset}\n`);
          skipped++;
          continue;
        }
        appendFileSync(OUTPUT_FILE, JSON.stringify(result) + '\n');
        process.stdout.write(`${colors.green}saved${colors.reset}\n`);
        success++;
      } catch (err) {
        process.stdout.write(`${colors.red}error: ${err.message}${colors.reset}\n`);
        failed++;
      }

      await new Promise(r => setTimeout(r, 350));
    }

    await new Promise(r => setTimeout(r, 500));
  }

  log(`\n${'─'.repeat(60)}`, colors.cyan);
  log(`✅  Done!`, colors.green);
  log(`   Saved:   ${success}`, colors.green);
  log(`   Skipped: ${skipped} (low grounding)`, colors.yellow);
  log(`   Failed:  ${failed}`, colors.red);
  log(`   Output:  ${OUTPUT_FILE}`, colors.cyan);

  // Show combined dataset size
  const existingCount = existsSync(join(OUTPUT_DIR, 'training_data.jsonl'))
    ? readFileSync(join(OUTPUT_DIR, 'training_data.jsonl'), 'utf8').split('\n').filter(Boolean).length
    : 0;
  const newCount = readFileSync(OUTPUT_FILE, 'utf8').split('\n').filter(Boolean).length;
  log(`\n📊  Dataset summary:`, colors.bold);
  log(`   Original (SFT):    ${existingCount} examples`, colors.cyan);
  log(`   DRAG-format (new): ${newCount} examples`, colors.cyan);
  log(`   Combined total:    ${existingCount + newCount} examples`, colors.green);
  log(`\n   ✨ Combine both files in colab_kd.ipynb: DATA_PATH array should include both.\n`, colors.yellow);
}

main().catch(err => { console.error(err); process.exit(1); });
