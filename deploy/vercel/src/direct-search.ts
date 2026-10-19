/**
 * Direct-search request parser, AI query planner, and Neon executor for Vercel.
 *
 * Compatible with `frontend-direct-search` (`POST /api/search`, `POST /api/search/plan`).
 */

import {
  resolveDatasetMeta,
  searchAudioNeon,
  searchSemanticNeon,
  searchYamnetNeon,
  type SearchIndex,
} from "./db.js";
import { gateway, generateText } from "ai";

export interface SearchPlan {
  original_query: string;
  indexes: SearchIndex[];
  text_query?: string;
  audio_query?: string;
  audio_query_en?: string;
  yamnet_query?: string;
  yamnet_query_en?: string;
  rationale?: string;
}

export interface DirectSearchRequest {
  query: string;
  indexes: SearchIndex[];
  k: number;
  rewrite: boolean;
  dataset_id?: string;
  plan?: SearchPlan;
}

const indexLabels: Record<SearchIndex, string> = {
  text: "Índice de texto (transcripciones)",
  audio: "Índice de audio (CLAP)",
  yamnet: "Clases de audio (YAMNet/AudioSet)",
};

export function parseSearchRequest(value: unknown): DirectSearchRequest {
  if (!value || typeof value !== "object") throw new Error("El body debe ser un objeto JSON.");
  const source = value as Record<string, unknown>;
  const query = typeof source.query === "string" ? source.query.trim() : "";
  if (!query) throw new Error("query es obligatorio.");

  const dataset_id = typeof source.dataset_id === "string" && source.dataset_id.trim()
    ? source.dataset_id.trim()
    : undefined;

  const explicitIndexes = Array.isArray(source.indexes)
    ? source.indexes.filter((idx): idx is SearchIndex => idx === "text" || idx === "audio" || idx === "yamnet")
    : undefined;

  const indexes: SearchIndex[] = explicitIndexes?.length
    ? explicitIndexes
    : [
        ...(source.include_text !== false ? (["text"] as const) : []),
        ...(source.include_clap === true ? (["audio"] as const) : []),
        ...(source.include_yamnet === true ? (["yamnet"] as const) : []),
      ];

  if (!indexes.length) throw new Error("Elegí al menos un índice para buscar.");

  const k = typeof source.k === "number" ? source.k : Number(source.k ?? 10);
  if (!Number.isInteger(k) || k < 1 || k > 50) throw new Error("k debe ser un entero entre 1 y 50.");

  return {
    query,
    indexes,
    k,
    rewrite: source.rewrite !== false,
    dataset_id,
    plan: source.plan && typeof source.plan === "object" ? (source.plan as SearchPlan) : undefined,
  };
}

export function effectivePlan(request: DirectSearchRequest): SearchPlan {
  if (request.plan) return { ...request.plan, original_query: request.query };
  return {
    original_query: request.query,
    indexes: request.indexes,
    text_query: request.indexes.includes("text") ? request.query : undefined,
    audio_query: request.indexes.includes("audio") ? request.query : undefined,
    yamnet_query: request.indexes.includes("yamnet") ? request.query : undefined,
    rationale: "Búsqueda literal.",
  };
}

export async function rewriteSearchPlanWithLLM(request: DirectSearchRequest): Promise<SearchPlan> {
  if (!request.rewrite || request.plan) return effectivePlan(request);

  const prompt = [
    "DIRECT_SEARCH_PLANNER_V1",
    "Generá exclusivamente un plan de búsqueda en formato JSON para el archivo periodístico de audio.",
    `Índices habilitados por el usuario: ${JSON.stringify(request.indexes)}. No agregues ni quites índices.`,
    "Devolvé sólo un objeto JSON con: original_query, indexes, rationale, y las queries correspondientes (text_query, audio_query, yamnet_query).",
    `Consulta original: ${JSON.stringify(request.query)}`,
  ].join("\n");

  const model = process.env.TEXT_GENERATION_MODEL?.trim() || "openai/gpt-4o-mini";
  const response = await generateText({
    model: gateway(model),
    temperature: 0,
    prompt,
  });
  const content = response.text;
  if (!content) throw new Error("El modelo devolvió un plan vacío.");
  const parsed = JSON.parse(content.replace(/^```(?:json)?\s*|\s*```$/g, "")) as Partial<SearchPlan>;
  return {
    original_query: request.query,
    indexes: request.indexes,
    text_query: request.indexes.includes("text") ? parsed.text_query || request.query : undefined,
    audio_query: request.indexes.includes("audio") ? parsed.audio_query || request.query : undefined,
    yamnet_query: request.indexes.includes("yamnet") ? parsed.yamnet_query || request.query : undefined,
    rationale: parsed.rationale || "Plan reformulado con IA.",
  };
}

export async function executeDirectSearchNeon(
  request: DirectSearchRequest,
): Promise<Record<string, unknown>> {
  const start = Date.now();
  const plan = await rewriteSearchPlanWithLLM(request);
  const meta = await resolveDatasetMeta(request.dataset_id);

  const searches = await Promise.all(
    plan.indexes.map(async (index) => {
      const effectiveQuery =
        index === "text"
          ? plan.text_query || request.query
          : index === "audio"
            ? plan.audio_query || request.query
            : plan.yamnet_query || request.query;

      try {
        if (index === "text") {
          const { results } = await searchSemanticNeon(effectiveQuery, request.k, meta.dataset_id);
          return {
            index,
            translatedQuery: undefined,
            bucket: {
              available: true,
              effective_query: effectiveQuery,
              results: results.map((hit, pos) => ({
                ...hit.segment,
                similarity: hit.similarity,
                rank: pos + 1,
                search_index: index,
                search_index_label: indexLabels[index],
              })),
            },
          };
        }
        if (index === "audio") {
          const { results, translatedQuery } = await searchAudioNeon(effectiveQuery, request.k, {
            queryEn: plan.audio_query_en,
            datasetId: meta.dataset_id,
          });
          return {
            index,
            translatedQuery,
            bucket: {
              available: true,
              effective_query: effectiveQuery,
              translated_query: translatedQuery,
              results: results.map((hit, pos) => ({
                ...hit.segment,
                similarity: hit.similarity,
                rank: pos + 1,
                search_index: index,
                search_index_label: indexLabels[index],
              })),
            },
          };
        }
        const { results, translatedQuery } = await searchYamnetNeon(effectiveQuery, request.k, {
          queryEn: plan.yamnet_query_en,
          datasetId: meta.dataset_id,
        });
        return {
          index,
          translatedQuery,
          bucket: {
            available: true,
            effective_query: effectiveQuery,
            translated_query: translatedQuery,
            results: results.map((hit, pos) => ({
              ...hit.segment,
              similarity: hit.similarity,
              rank: pos + 1,
              search_index: index,
              search_index_label: indexLabels[index],
            })),
          },
        };
      } catch (error) {
        console.error(`Direct ${index} search on Neon failed:`, error);
        return {
          index,
          translatedQuery: undefined,
          bucket: {
            available: false,
            effective_query: effectiveQuery,
            results: [],
            error: "Fuente de búsqueda no disponible temporalmente.",
          },
        };
      }
    }),
  );

  const audioTranslation = searches.find((s) => s.index === "audio")?.translatedQuery;
  const yamnetTranslation = searches.find((s) => s.index === "yamnet")?.translatedQuery;

  return {
    query: request.query,
    dataset_id: meta.dataset_id,
    dataset_label: meta.dataset_label,
    dataset_release: meta.dataset_release,
    plan: {
      ...plan,
      ...(audioTranslation ? { audio_query_en: audioTranslation } : {}),
      ...(yamnetTranslation ? { yamnet_query_en: yamnetTranslation } : {}),
    },
    took_ms: Date.now() - start,
    indexes: Object.fromEntries(searches.map(({ index, bucket }) => [index, bucket])),
  };
}
