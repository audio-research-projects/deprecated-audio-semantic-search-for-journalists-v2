/**
 * Neon PostgreSQL + pgvector retrieval backend for Vercel + Eve Agents.
 *
 * Implements the exact same retrieval semantics as `src/agent_service/search_engine.py`:
 * - Minimum segment duration filter (`MIN_SEGMENT_DURATION_SECONDS`)
 * - Semantic text search over `text_embedding` using cosine similarity (`1 - (vec <=> query)`)
 * - Cross-modal audio search over `audio_embedding`
 * - YAMNet/AudioSet inverted token search ranked by `coverage * mean_classifier_score`
 * - Journalist-facing result serialization (`search_index_label`, `clip_url`, etc.)
 */

import { neon } from "@neondatabase/serverless";
import {
  generateAudioQueryEmbedding,
  generateTextQueryEmbedding,
  toPgVectorLiteral,
  translateToEnglish,
} from "./embeddings.js";

export type SearchIndex = "text" | "audio" | "yamnet";

export interface YamnetClassItem {
  class_id: string;
  class_name: string;
  score: number;
}

export interface SegmentRecord {
  segment_id: number;
  text: string;
  start_time: number;
  end_time: number;
  original_file_name: string;
  language: string;
  confidence: number;
  clip_file_name?: string;
  clip_start_time?: number;
  clip_end_time?: number;
  clip_url?: string;
  yamnet_audio_classes?: YamnetClassItem[];
  yamnet_matched_classes?: YamnetClassItem[];
  sentiment_positive?: number;
  sentiment_negative?: number;
  sentiment_neutral?: number;
  dominant_sentiment?: string;
}

export interface SearchHit {
  segment: SegmentRecord;
  similarity: number;
  distance: number;
}

export interface DatasetMeta {
  dataset_id: string;
  dataset_label: string;
  dataset_release: string | null;
  language: string;
  active_indexes: string[];
  total_segments: number;
}

const STOPWORDS = new Set([
  "a", "an", "and", "at", "audio", "by", "during", "event", "events",
  "from", "in", "near", "of", "or", "sound", "sounds", "the", "to", "while", "with",
]);

export function normalizedAudioClassTokens(value: string): string[] {
  const normalized = value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const matches = normalized.match(/[a-z0-9]+/g) ?? [];
  return [...new Set(matches.filter((token) => !STOPWORDS.has(token)))].sort();
}

export function configuredMinDurationSeconds(): number {
  const raw = process.env.MIN_SEGMENT_DURATION_SECONDS?.trim();
  if (!raw) return 0.0;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0.0;
}

function getSql() {
  const url = process.env.DATABASE_URL?.trim();
  if (!url) {
    throw new Error("DATABASE_URL is not configured. Set your Neon Postgres connection string.");
  }
  return neon(url);
}

function rowToSegment(
  row: Record<string, unknown>,
  includeSentiment = false,
  datasetId?: string,
): SegmentRecord {
  const startTime = Number(row.start_time ?? 0);
  const endTime = Number(row.end_time ?? 0);
  const classesRaw = row.yamnet_top_classes;
  const yamnetClasses: YamnetClassItem[] = Array.isArray(classesRaw)
    ? (classesRaw as YamnetClassItem[])
    : typeof classesRaw === "string"
      ? (JSON.parse(classesRaw) as YamnetClassItem[])
      : [];

  const segment: SegmentRecord = {
    segment_id: Number(row.segment_id ?? 0),
    text: String(row.text ?? ""),
    start_time: startTime,
    end_time: endTime,
    original_file_name: String(row.original_file_name ?? ""),
    language: String(row.language ?? "es"),
    confidence: Number(row.confidence ?? 0),
  };

  if (row.clip_file_name) {
    segment.clip_file_name = String(row.clip_file_name);
    segment.clip_start_time = Number(row.clip_start_time ?? startTime);
    segment.clip_end_time = Number(row.clip_end_time ?? endTime);
  }
  if (row.clip_url) {
    segment.clip_url = datasetId
      ? `/api/clips/${segment.segment_id}?dataset_id=${encodeURIComponent(datasetId)}`
      : undefined;
  }
  if (yamnetClasses.length > 0) {
    segment.yamnet_audio_classes = yamnetClasses;
  }
  if (includeSentiment && row.dominant_sentiment) {
    segment.sentiment_positive = Number(row.sentiment_positive ?? 0);
    segment.sentiment_negative = Number(row.sentiment_negative ?? 0);
    segment.sentiment_neutral = Number(row.sentiment_neutral ?? 0);
    segment.dominant_sentiment = String(row.dominant_sentiment);
  }
  return segment;
}

export async function resolveDatasetMeta(datasetId?: string): Promise<DatasetMeta> {
  const sql = getSql();
  const targetId = datasetId?.trim() || process.env.DEFAULT_DATASET_ID?.trim() || "";
  const rows = targetId
    ? await sql`SELECT * FROM datasets WHERE id = ${targetId} LIMIT 1`
    : await sql`SELECT * FROM datasets ORDER BY is_default DESC, updated_at DESC LIMIT 1`;

  if (!rows.length) {
    throw new Error(targetId ? `Dataset '${targetId}' not found in Neon.` : "No datasets found in Neon.");
  }
  const row = rows[0] as Record<string, unknown>;
  return {
    dataset_id: String(row.id),
    dataset_label: String(row.label ?? row.id),
    dataset_release: row.dataset_version ? String(row.dataset_version) : null,
    language: String(row.language ?? "es"),
    active_indexes: Array.isArray(row.active_indexes) ? (row.active_indexes as string[]) : ["text"],
    total_segments: Number(row.total_segments ?? 0),
  };
}

export async function listDatasets(): Promise<Array<Record<string, unknown>>> {
  const sql = getSql();
  const rows = await sql`SELECT * FROM datasets ORDER BY is_default DESC, id ASC`;
  return rows.map((row) => ({
    id: String(row.id),
    label: String(row.label ?? row.id),
    language: String(row.language ?? "es"),
    active_indexes: Array.isArray(row.active_indexes) ? row.active_indexes : [],
    dataset_version: row.dataset_version ? String(row.dataset_version) : null,
  }));
}

export async function searchSemanticNeon(
  query: string,
  k = 5,
  datasetId?: string,
): Promise<{ results: SearchHit[]; meta: DatasetMeta }> {
  const sql = getSql();
  const meta = await resolveDatasetMeta(datasetId);
  const minDuration = configuredMinDurationSeconds();
  const embedding = await generateTextQueryEmbedding(query);
  const vectorLiteral = toPgVectorLiteral(embedding);

  const rows = await sql`
    SELECT
      segment_id, text, start_time, end_time, original_file_name, language, confidence,
      clip_file_name, clip_start_time, clip_end_time, clip_url, yamnet_top_classes,
      1 - (text_embedding <=> ${vectorLiteral}::vector) AS similarity
    FROM audio_segments
    WHERE dataset_id = ${meta.dataset_id}
      AND text_embedding IS NOT NULL
      AND (end_time - start_time) > ${minDuration}
    ORDER BY text_embedding <=> ${vectorLiteral}::vector ASC
    LIMIT ${k}
  `;

  const results: SearchHit[] = rows.map((row) => {
    const similarity = Number(row.similarity ?? 0);
    return {
      segment: rowToSegment(row as Record<string, unknown>, false, meta.dataset_id),
      similarity,
      distance: 1 - similarity,
    };
  });
  return { results, meta };
}

export async function searchAudioNeon(
  query: string,
  k = 5,
  options?: { queryEn?: string; datasetId?: string },
): Promise<{ results: SearchHit[]; translatedQuery: string; meta: DatasetMeta }> {
  const sql = getSql();
  const meta = await resolveDatasetMeta(options?.datasetId);
  const minDuration = configuredMinDurationSeconds();
  const translatedQuery = options?.queryEn?.trim() || await translateToEnglish(query);
  const embedding = await generateAudioQueryEmbedding(translatedQuery);
  const vectorLiteral = toPgVectorLiteral(embedding);

  const rows = await sql`
    SELECT
      segment_id, text, start_time, end_time, original_file_name, language, confidence,
      clip_file_name, clip_start_time, clip_end_time, clip_url, yamnet_top_classes,
      1 - (audio_embedding <=> ${vectorLiteral}::vector) AS similarity
    FROM audio_segments
    WHERE dataset_id = ${meta.dataset_id}
      AND audio_embedding IS NOT NULL
      AND (end_time - start_time) > ${minDuration}
    ORDER BY audio_embedding <=> ${vectorLiteral}::vector ASC
    LIMIT ${k}
  `;

  const results: SearchHit[] = rows.map((row) => {
    const similarity = Number(row.similarity ?? 0);
    return {
      segment: rowToSegment(row as Record<string, unknown>, false, meta.dataset_id),
      similarity,
      distance: 1 - similarity,
    };
  });
  return { results, translatedQuery, meta };
}

export async function searchYamnetNeon(
  query: string,
  k = 5,
  options?: { queryEn?: string; datasetId?: string },
): Promise<{ results: SearchHit[]; translatedQuery: string; meta: DatasetMeta }> {
  const sql = getSql();
  const meta = await resolveDatasetMeta(options?.datasetId);
  const minDuration = configuredMinDurationSeconds();
  const translatedQuery = options?.queryEn?.trim() || (await translateToEnglish(query));
  const tokens = normalizedAudioClassTokens(translatedQuery);
  if (!tokens.length) {
    return { results: [], translatedQuery, meta };
  }

  const postings = await sql`
    SELECT
      p.token, p.segment_id, p.class_id, p.class_name, p.score, p.class_rank,
      s.text, s.start_time, s.end_time, s.original_file_name, s.language, s.confidence,
      s.clip_file_name, s.clip_start_time, s.clip_end_time, s.clip_url, s.yamnet_top_classes
    FROM segment_yamnet_postings p
    JOIN audio_segments s
      ON s.dataset_id = p.dataset_id AND s.segment_id = p.segment_id
    WHERE p.dataset_id = ${meta.dataset_id}
      AND p.token = ANY(${tokens})
      AND (s.end_time - s.start_time) > ${minDuration}
  `;

  const bySegment = new Map<
    number,
    {
      row: Record<string, unknown>;
      classes: Map<
        string,
        { class_id: string; class_name: string; score: number; class_rank: number; tokens: Set<string> }
      >;
    }
  >();

  for (const raw of postings) {
    const row = raw as Record<string, unknown>;
    const segmentId = Number(row.segment_id);
    const classId = String(row.class_id);
    const token = String(row.token);

    let entry = bySegment.get(segmentId);
    if (!entry) {
      entry = { row, classes: new Map() };
      bySegment.set(segmentId, entry);
    }
    let classMatch = entry.classes.get(classId);
    if (!classMatch) {
      classMatch = {
        class_id: classId,
        class_name: String(row.class_name),
        score: Number(row.score),
        class_rank: Number(row.class_rank),
        tokens: new Set(),
      };
      entry.classes.set(classId, classMatch);
    }
    classMatch.tokens.add(token);
  }

  const ranked: Array<{ rankScore: number; segment: SegmentRecord }> = [];
  for (const [, entry] of bySegment) {
    const orderedMatches = [...entry.classes.values()].sort((a, b) => a.class_rank - b.class_rank);
    const coveredTokens = new Set<string>();
    for (const m of orderedMatches) {
      for (const t of m.tokens) coveredTokens.add(t);
    }
    const coverage = coveredTokens.size / tokens.length;
    const meanScore =
      orderedMatches.reduce((acc, item) => acc + item.score, 0) / orderedMatches.length;
    const rankScore = coverage * meanScore;

    const segment = rowToSegment(entry.row, false, meta.dataset_id);
    segment.yamnet_matched_classes = orderedMatches.map((m) => ({
      class_id: m.class_id,
      class_name: m.class_name,
      score: m.score,
    }));
    ranked.push({ rankScore, segment });
  }

  ranked.sort((a, b) => b.rankScore - a.rankScore);
  const results: SearchHit[] = ranked.slice(0, k).map(({ rankScore, segment }) => ({
    segment,
    similarity: rankScore,
    distance: 1 - rankScore,
  }));

  return { results, translatedQuery, meta };
}

export async function getSegmentInfoNeon(
  segmentId: number,
  datasetId?: string,
): Promise<SegmentRecord | null> {
  const sql = getSql();
  const meta = await resolveDatasetMeta(datasetId);
  const rows = await sql`
    SELECT * FROM audio_segments
    WHERE dataset_id = ${meta.dataset_id} AND segment_id = ${segmentId}
    LIMIT 1
  `;
  if (!rows.length) return null;
  return rowToSegment(rows[0] as Record<string, unknown>, true, meta.dataset_id);
}

export async function getClipBlobReferenceNeon(
  segmentId: number,
  datasetId: string,
): Promise<string | null> {
  const sql = getSql();
  const rows = await sql`
    SELECT clip_url FROM audio_segments
    WHERE dataset_id = ${datasetId} AND segment_id = ${segmentId}
    LIMIT 1
  `;
  const url = rows[0]?.clip_url;
  return typeof url === "string" && url ? url : null;
}

export async function getAudioClassesNeon(
  segmentId: number,
  datasetId?: string,
): Promise<YamnetClassItem[] | null> {
  const segment = await getSegmentInfoNeon(segmentId, datasetId);
  if (!segment) return null;
  return segment.yamnet_audio_classes ?? [];
}

export async function insertFeedbackNeon(payload: {
  messageId: string;
  feedback: "thumbsUp" | "thumbsDown";
  question: string;
  answer: string;
}): Promise<void> {
  const sql = getSql();
  await sql`
    INSERT INTO feedback_logs (message_id, feedback, question, answer)
    VALUES (${payload.messageId}, ${payload.feedback}, ${payload.question}, ${payload.answer})
  `;
}

export function serializeAgentResults(
  results: SearchHit[],
  searchIndex: SearchIndex,
  searchIndexLabel: string,
): Array<Record<string, unknown>> {
  return results.map((result) => {
    const seg = result.segment;
    const item: Record<string, unknown> = {
      segment_id: seg.segment_id,
      search_index: searchIndex,
      search_index_label: searchIndexLabel,
      text: seg.text,
      similarity: Number(result.similarity.toFixed(4)),
      similarity_percent: Number((result.similarity * 100).toFixed(1)),
      start_time: seg.start_time,
      end_time: seg.end_time,
      duration: Number((seg.end_time - seg.start_time).toFixed(1)),
      original_file_name: seg.original_file_name,
      language: seg.language,
      confidence: seg.confidence,
    };
    if (seg.clip_url) item.clip_url = seg.clip_url;
    if (seg.clip_start_time !== undefined) item.clip_start_time = seg.clip_start_time;
    if (seg.clip_end_time !== undefined) item.clip_end_time = seg.clip_end_time;
    if (seg.yamnet_audio_classes) item.yamnet_audio_classes = seg.yamnet_audio_classes;
    if (seg.yamnet_matched_classes) item.yamnet_matched_classes = seg.yamnet_matched_classes;
    return item;
  });
}
