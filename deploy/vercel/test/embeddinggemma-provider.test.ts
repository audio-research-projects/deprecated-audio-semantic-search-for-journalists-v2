import assert from "node:assert/strict";
import { test } from "node:test";

import { generateAudioQueryEmbedding } from "../src/embeddings.js";

test("EmbeddingGemma audio retrieval encodes its text query as text", async (t) => {
  const originalProvider = process.env.AUDIO_EMBEDDING_PROVIDER;
  const originalUrl = process.env.EMBEDDINGGEMMA_API_URL;
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
    if (originalProvider === undefined) delete process.env.AUDIO_EMBEDDING_PROVIDER;
    else process.env.AUDIO_EMBEDDING_PROVIDER = originalProvider;
    if (originalUrl === undefined) delete process.env.EMBEDDINGGEMMA_API_URL;
    else process.env.EMBEDDINGGEMMA_API_URL = originalUrl;
  });
  process.env.AUDIO_EMBEDDING_PROVIDER = "embeddinggemma";
  process.env.EMBEDDINGGEMMA_API_URL = "https://embedding.example/";
  globalThis.fetch = async (input, init) => {
    assert.equal(input, "https://embedding.example/embed");
    assert.deepEqual(JSON.parse(String(init?.body)), {
      text: "applause in a presidential debate",
      modality: "text",
    });
    return new Response(JSON.stringify({ embedding: [3, 4] }), {
      headers: { "Content-Type": "application/json" },
    });
  };

  assert.deepEqual(
    await generateAudioQueryEmbedding("applause in a presidential debate"),
    [0.6, 0.8],
  );
});

test("default audio provider uses the internal EmbeddingGemma service binding", async (t) => {
  const keys = [
    "AUDIO_EMBEDDING_PROVIDER",
    "EMBEDDINGGEMMA_API_URL",
    "EMBEDDINGGEMMA_SERVICE_URL",
    "CUSTOM_EMBEDDING_API_URL",
  ];
  const saved = new Map(keys.map((key) => [key, process.env[key]]));
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
    for (const [key, value] of saved) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
  for (const key of keys) delete process.env[key];
  process.env.EMBEDDINGGEMMA_SERVICE_URL = "https://internal-embeddings.example/";
  globalThis.fetch = async (input, init) => {
    assert.equal(input, "https://internal-embeddings.example/embed");
    assert.deepEqual(JSON.parse(String(init?.body)), { text: "inflación", modality: "text" });
    return new Response(JSON.stringify({ embedding: [1, 0] }));
  };
  assert.deepEqual(await generateAudioQueryEmbedding("inflación"), [1, 0]);
});
