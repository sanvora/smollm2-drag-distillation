#!/usr/bin/env node

/**
 * Generates ChatML training data from knowledge chunks using a teacher model (Groq).
 * Usage: node scripts/gen-training-data.mjs [--dry-run] [--resume]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Configuration
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const TEACHER_MODEL = 'llama-3.3-70b-versatile'; // High quality teacher model
const GROUNDING_THRESHOLD = 0.85;
const MAX_RETRIES = 3;
const PAIRS_PER_CHUNK = 10; // Generate N diverse Q&A pairs per chunk
const EVIDENCE_CHARS = 400;  // Max chars of source chunk text to include in training system prompt

const QUESTION_TYPES = [
  'factual',       // What did Kham do / what is X?
  'behavioral',    // Tell me about a time when...
  'technical',     // How does X work technically?
  'metrics',       // What measurable results were achieved?
  'motivation',    // Why did Kham choose X approach?
  'comparison',    // How does X compare to Y?
  'depth',         // Explain X in more detail
  'scenario',      // If Kham were to solve X problem, how would he approach it?
  'skills',        // What skills were required for X?
  'impact',        // What was the business or technical impact of X?
];

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const OUTPUT_DIR = path.join(ROOT, 'scripts', 'finetune');

// CLI Arguments
const args = process.argv.slice(2);
const isDryRun = args.includes('--dry-run');
const isResume = args.includes('--resume');
// --rag: include a compact system prompt with source evidence in each training example.
// This teaches the model to condition on retrieved context, matching inference-time behavior.
const isRag = args.includes('--rag');

// --rag writes to training_data_rag.jsonl to preserve the bare training_data.jsonl
const OUTPUT_FILE = path.join(OUTPUT_DIR, isRag ? 'training_data_rag.jsonl' : 'training_data.jsonl');

// Chunk sources - expand as needed
const CHUNK_FILES = [
  'public/data/knowledge_chunks.json',
  'public/data/company_chunks.json'
];

// Colors for console
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m'
};

function log(msg, color = colors.reset) {
  console.log(`${color}${msg}${colors.reset}`);
}

async function main() {
  log(`Starting training data generation...`, colors.cyan);

  if (!process.env.GROQ_API_KEY && !isDryRun) {
    log(`Error: GROQ_API_KEY environment variable is missing.`, colors.red);
    process.exit(1);
  }

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Load all chunks
  let allChunks = [];
  for (const relativePath of CHUNK_FILES) {
    const fullPath = path.join(ROOT, relativePath);
    if (fs.existsSync(fullPath)) {
      try {
        const fileContent = fs.readFileSync(fullPath, 'utf-8');
        const parsed = JSON.parse(fileContent);
        // knowledge_chunks.json is a flat array; company_chunks.json is { companyIndex, jobIndex, chunks: [...] }
        const chunkList = Array.isArray(parsed) ? parsed : (parsed.chunks ?? []);
        if (chunkList.length > 0) {
          allChunks = allChunks.concat(chunkList);
          log(`Loaded ${chunkList.length} chunks from ${relativePath}`, colors.green);
        } else {
          log(`Warning: No chunks found in ${relativePath}`, colors.yellow);
        }
      } catch (err) {
        log(`Warning: Failed to parse ${relativePath}: ${err.message}`, colors.yellow);
      }
    } else {
      log(`Warning: File not found: ${relativePath}`, colors.yellow);
    }
  }

  if (allChunks.length === 0) {
    log(`Error: No chunks loaded. Ensure plaintext chunks are available (try: 'npm run embed:all').`, colors.red);
    process.exit(1);
  }

  // Determine which chunks to process
  let chunksToProcess = allChunks;
  const processedChunkIds = new Set();

  if (isResume && fs.existsSync(OUTPUT_FILE)) {
    const existingLines = fs.readFileSync(OUTPUT_FILE, 'utf-8').split('\n').filter(Boolean);
    for (const line of existingLines) {
      try {
        const entry = JSON.parse(line);
        if (entry.meta && entry.meta.sourceChunkIds) {
          entry.meta.sourceChunkIds.forEach(id => processedChunkIds.add(id));
        }
      } catch (e) {
        // ignore malformed lines
      }
    }
    chunksToProcess = allChunks.filter(c => !processedChunkIds.has(c.id));
    log(`Resuming: Skipped ${processedChunkIds.size} already processed chunks.`, colors.yellow);
  }

  log(`Processing ${chunksToProcess.length} chunks...`, colors.cyan);

  if (isDryRun) {
    log(`\n--- DRY RUN MODE (Printing ${PAIRS_PER_CHUNK} mock examples from first chunk) ---\n`, colors.yellow);
    log(`RAG mode: ${isRag ? 'ON (system prompt + evidence)' : 'OFF (bare user/assistant)'}`, colors.yellow);
    const sampleChunk = chunksToProcess[0];
    for (let i = 0; i < Math.min(PAIRS_PER_CHUNK, 3); i++) {
      const example = await generateExample(sampleChunk, QUESTION_TYPES[i], true, isRag);
      if (example) {
        console.log(JSON.stringify(example, null, 2));
        console.log('---');
      }
    }
    log(`\nDry run complete. No files written.`, colors.green);
    return;
  }

  log(`RAG mode: ${isRag ? 'ON — adding compact system prompt + evidence to each example' : 'OFF — bare user/assistant pairs'}`, colors.cyan);

  // Main processing loop
  let successCount = 0;
  let failCount = 0;
  let skippedCount = 0;

  for (let i = 0; i < chunksToProcess.length; i++) {
    const chunk = chunksToProcess[i];
    log(`\nChunk ${i + 1}/${chunksToProcess.length} (${chunk.id})`, colors.cyan);

    for (let p = 0; p < PAIRS_PER_CHUNK; p++) {
      const qType = QUESTION_TYPES[p % QUESTION_TYPES.length];
      process.stdout.write(`  pair ${p + 1}/${PAIRS_PER_CHUNK} [${qType}]... `);

      try {
        const result = await generateExample(chunk, qType, false, isRag);

        if (!result) {
          process.stdout.write(`${colors.red}Failed${colors.reset}\n`);
          failCount++;
          continue;
        }

        if (result.meta.groundingScore < GROUNDING_THRESHOLD) {
          process.stdout.write(`${colors.yellow}Skipped (Low Grounding: ${result.meta.groundingScore})${colors.reset}\n`);
          skippedCount++;
          continue;
        }

        fs.appendFileSync(OUTPUT_FILE, JSON.stringify(result) + '\n');
        process.stdout.write(`${colors.green}Saved${colors.reset}\n`);
        successCount++;

      } catch (err) {
        process.stdout.write(`${colors.red}Error: ${err.message}${colors.reset}\n`);
        failCount++;
      }

      // Rate limiting delay
      await new Promise(resolve => setTimeout(resolve, 300));
    }

    // Rate limiting between chunks
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  log(`\nGeneration complete!`, colors.cyan);
  log(`Success: ${successCount}`, colors.green);
  log(`Skipped (Low Grounding): ${skippedCount}`, colors.yellow);
  log(`Failed: ${failCount}`, colors.red);
  log(`Output: ${OUTPUT_FILE}`, colors.reset);
}

// Generate a single Q&A pair of a given question type
async function generateExample(chunk, questionType = 'factual', mock = false, ragMode = false) {
  // Build system message for RAG-aware training (matches inference compact prompt format)
  const systemMessage = ragMode
    ? {
        role: 'system',
        content: [
          `You are a helpful assistant for Kham's portfolio. Be concise and accurate.`,
          `Answer using ONLY these facts:`,
          `1. ${chunk.text.slice(0, EVIDENCE_CHARS)}`,
        ].join('\n'),
      }
    : null;

  if (mock) {
    const messages = [
      ...(systemMessage ? [systemMessage] : []),
      { role: 'user', content: `[${questionType}] What is interesting about ${chunk.id}?` },
      { role: 'assistant', content: `This chunk discusses... ${chunk.text.substring(0, 20)}...` }
    ];
    return {
      messages,
      meta: {
        sourceChunkIds: [chunk.id],
        groundingScore: 0.95,
        generatedBy: 'mock-teacher',
        questionType,
        ragMode,
      }
    };
  }

  const typeInstructions = {
    factual:    'Ask a specific factual question about what is described (e.g., "What did Kham do at X?" or "What is X?").',
    behavioral: 'Ask a behavioral interview question starting with "Tell me about a time..." or "Describe a situation where..."',
    technical:  'Ask a technical deep-dive question about the methodology, algorithm, or engineering approach described.',
    metrics:    'Ask about measurable outcomes, numbers, percentages, or quantified impact described in the context.',
    motivation: 'Ask why a particular approach, technology, or decision was made.',
    comparison: 'Ask how one approach, tool, or result compares to another or to industry standards.',
    depth:      'Ask for a more detailed explanation of a specific concept, project, or technique mentioned.',
    scenario:   'Pose a realistic work scenario and ask how Kham would approach it based on his experience.',
    skills:     'Ask what specific technical skills, tools, or domain knowledge were required or demonstrated.',
    impact:     'Ask about the business value, strategic importance, or real-world impact of the work described.',
  };

  const instruction = typeInstructions[questionType] || typeInstructions.factual;

  const prompt = `You are an expert technical interviewer creating high-quality training data for a personal AI assistant.

Context about Kham:
"""
${chunk.text}
"""

Task:
1. Question type: ${questionType.toUpperCase()} — ${instruction}
2. Write a question that can be answered using ONLY the context above.
3. Write a thorough, specific answer grounded strictly in the context. Do not invent facts.
4. Rate how well the answer is grounded in the context (0.0–1.0).

Response Format (JSON only, no markdown):
{
  "question": "...",
  "answer": "...",
  "grounding_score": 0.0
}`;

  try {
    const response = await callGroq(prompt);
    let parsed;
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      parsed = JSON.parse(jsonMatch ? jsonMatch[0] : response);
    } catch (e) {
      console.error("Failed to parse JSON response", response.substring(0, 100));
      return null;
    }

    if (!parsed || !parsed.question || !parsed.answer) return null;

    const messages = [
      ...(systemMessage ? [systemMessage] : []),
      { role: 'user', content: parsed.question },
      { role: 'assistant', content: parsed.answer }
    ];

    return {
      messages,
      meta: {
        sourceChunkIds: [chunk.id],
        groundingScore: parsed.grounding_score || 0,
        generatedBy: TEACHER_MODEL,
        questionType,
        ragMode,
      }
    };
  } catch (error) {
    console.error(`API Call failed: ${error.message}`);
    return null;
  }
}

async function callGroq(prompt, retries = MAX_RETRIES) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(GROQ_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: TEACHER_MODEL,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.85,   // higher diversity → less repetitive training labels
          response_format: { type: "json_object" } // Force JSON
        })
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      }

      const data = await resp.json();
      return data.choices[0].message.content;
    } catch (err) {
      if (attempt === retries - 1) throw err;
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt))); // Exponential backoff
    }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
