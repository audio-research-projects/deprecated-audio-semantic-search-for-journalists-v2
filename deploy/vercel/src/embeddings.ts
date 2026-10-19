/**
 * API-based query embedding & translation utilities for Vercel Serverless Functions.
 *
 * Because CLAP (~1.7 GB PyTorch) exceeds Vercel Function bundle limits, query
 * embeddings are generated via lightweight HTTP APIs (HuggingFace Inference API
 * for 384d MiniLM, Gemini Embedding 2 for multimodal audio, OpenAI embeddings,
 * or a custom HTTP endpoint) and compared against Neon pgvector.
 */

export type EmbeddingProvider = "huggingface" | "gemini" | "openai" | "custom" | "embeddinggemma";

import { gateway, generateText } from "ai";

function l2Normalize(values: number[]): number[] {
  let normSq = 0;
  for (const v of values) normSq += v * v;
  const norm = Math.sqrt(normSq);
  if (norm === 0 || !Number.isFinite(norm)) return values;
  return values.map((v) => v / norm);
}

export function toPgVectorLiteral(values: number[]): string {
  return `[${values.map((v) => Number(v).toFixed(8)).join(",")}]`;
}

export async function translateToEnglish(text: string): Promise<string> {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const queryLang = (process.env.QUERY_LANGUAGE ?? "es").toLowerCase();
  if (queryLang === "en" || queryLang === "english") return trimmed;

  const model = process.env.TEXT_GENERATION_MODEL?.trim() || "openai/gpt-4o-mini";
  const response = await generateText({
    model: gateway(model),
    temperature: 0,
    system:
      "Translate the audio search query into English for CLAP, EmbeddingGemma, or AudioSet retrieval. Preserve its meaning, sound events, proper nouns, quoted phrases, and numbers. If already English, return it unchanged. Output only the English query.",
    prompt: trimmed,
  });
  const translated = response.text.trim();
  if (!translated) throw new Error("Gateway returned an empty English translation.");
  return translated;
}

async function embedWithHuggingFace(text: string, model: string): Promise<number[]> {
  const token = process.env.HF_API_TOKEN?.trim();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(
    `https://router.huggingface.co/hf-inference/models/${model}/pipeline/feature-extraction`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
    },
  );
  if (!response.ok) {
    throw new Error(`HuggingFace embedding error (${response.status}): ${await response.text()}`);
  }
  const payload = (await response.json()) as unknown;
  const vector = Array.isArray(payload) && Array.isArray(payload[0])
    ? (payload[0] as number[])
    : (payload as number[]);
  return l2Normalize(vector);
}

async function embedWithOpenAI(text: string, model: string): Promise<number[]> {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) throw new Error("OPENAI_API_KEY is required for OpenAI embeddings.");
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input: text }),
  });
  if (!response.ok) {
    throw new Error(`OpenAI embedding error (${response.status}): ${await response.text()}`);
  }
  const data = (await response.json()) as {
    data?: Array<{ embedding: number[] }>;
  };
  const vector = data.data?.[0]?.embedding;
  if (!vector) throw new Error("OpenAI returned no embedding.");
  return l2Normalize(vector);
}

async function embedWithGemini(text: string, model: string): Promise<number[]> {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) throw new Error("GEMINI_API_KEY is required for Gemini embeddings.");
  const outputDimensionality = Number(process.env.GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY ?? 1536);
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: `models/${model}`,
        content: { parts: [{ text: `task: search result | query: ${text}` }] },
        outputDimensionality,
      }),
    },
  );
  if (!response.ok) {
    throw new Error(`Gemini embedding error (${response.status}): ${await response.text()}`);
  }
  const data = (await response.json()) as {
    embedding?: { values?: number[] };
  };
  const values = data.embedding?.values;
  if (!values?.length) throw new Error("Gemini returned no embedding.");
  return l2Normalize(values);
}

async function embedWithCustomEndpoint(text: string, modality: "text" | "audio"): Promise<number[]> {
  const baseUrl = process.env.CUSTOM_EMBEDDING_API_URL?.trim().replace(/\/$/, "");
  if (!baseUrl) throw new Error("CUSTOM_EMBEDDING_API_URL is required for custom embeddings.");
  const response = await fetch(`${baseUrl}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, modality }),
  });
  if (!response.ok) {
    throw new Error(`Custom embedding error (${response.status}): ${await response.text()}`);
  }
  const data = (await response.json()) as { embedding?: number[] };
  if (!data.embedding?.length) throw new Error("Custom endpoint returned no embedding.");
  return l2Normalize(data.embedding);
}

/**
 * EmbeddingGemma 2 served over HTTP (e.g. deploy/vercel/embeddinggemma-service
 * on Cloud Run). Same wire contract as the custom provider: POST /embed with
 * { text, modality } → { embedding }. The service applies the SearchQuery
 * prompt and MRL truncation, so the vector matches the 768/512/256/128-dim
 * embeddings ingested into Neon.
 */
async function embedWithEmbeddingGemma(text: string): Promise<number[]> {
  const baseUrl = (
    process.env.EMBEDDINGGEMMA_API_URL ??
    process.env.EMBEDDINGGEMMA_SERVICE_URL ??
    process.env.CUSTOM_EMBEDDING_API_URL
  )?.trim().replace(/\/$/, "");
  if (!baseUrl) {
    throw new Error(
      "EMBEDDINGGEMMA_SERVICE_URL or EMBEDDINGGEMMA_API_URL is required for the embeddinggemma provider."
    );
  }
  const response = await fetch(`${baseUrl}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Both indexes receive text queries. The endpoint's modality describes
    // the input being encoded, rather than the index being searched.
    body: JSON.stringify({ text, modality: "text" }),
  });
  if (!response.ok) {
    throw new Error(`EmbeddingGemma endpoint error (${response.status}): ${await response.text()}`);
  }
  const data = (await response.json()) as { embedding?: number[] };
  if (!data.embedding?.length) throw new Error("EmbeddingGemma endpoint returned no embedding.");
  return l2Normalize(data.embedding);
}

export async function generateTextQueryEmbedding(query: string): Promise<number[]> {
  // The deployed corpus stores 384d MiniLM vectors; keep the query encoder aligned
  // until the complete corpus is re-embedded with another model.
  const provider = (process.env.TEXT_EMBEDDING_PROVIDER ?? "huggingface").toLowerCase() as EmbeddingProvider;
  const model = process.env.TEXT_EMBEDDING_MODEL?.trim() || "sentence-transformers/all-MiniLM-L6-v2";
  if (provider === "openai") return embedWithOpenAI(query, model);
  if (provider === "gemini") return embedWithGemini(query, model);
  if (provider === "custom") return embedWithCustomEndpoint(query, "text");
  if (provider === "embeddinggemma") return embedWithEmbeddingGemma(query);
  return embedWithHuggingFace(query, model);
}

export async function generateAudioQueryEmbedding(englishQuery: string): Promise<number[]> {
  const provider = (process.env.AUDIO_EMBEDDING_PROVIDER ?? "embeddinggemma").toLowerCase() as EmbeddingProvider;
  const model = process.env.AUDIO_EMBEDDING_MODEL?.trim() || "gemini-embedding-2";
  if (provider === "gemini") return embedWithGemini(englishQuery, model);
  if (provider === "openai") return embedWithOpenAI(englishQuery, model);
  if (provider === "custom") return embedWithCustomEndpoint(englishQuery, "audio");
  if (provider === "embeddinggemma") return embedWithEmbeddingGemma(englishQuery);
  return embedWithHuggingFace(englishQuery, model);
}
