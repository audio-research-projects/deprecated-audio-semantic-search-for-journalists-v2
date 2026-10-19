import { gateway, generateText, stepCountIs, tool } from "ai";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { z } from "zod";
import {
  getAudioClassesNeon,
  getSegmentInfoNeon,
  searchAudioNeon,
  searchSemanticNeon,
  searchYamnetNeon,
  serializeAgentResults,
} from "../src/db.js";
import { withCors } from "../src/cors.js";

const SYSTEM_PROMPT = `Eres un asistente especializado en búsqueda semántica de contenido de audio para periodistas (desplegado en Vercel con Neon pgvector y Eve Agents).

Reglas:
1. Usa buscar_audio para contenido dicho o escrito en las transcripciones.
2. Usa buscar_evento_acustico para sonidos como aplausos, música, gritos o risas.
3. Usa buscar_clase_audio cuando convenga buscar explícitamente etiquetas acústicas estandarizadas de AudioSet/YAMNet, por ejemplo Applause, Speech o Music. Sus scores combinan coincidencia de clase y confianza del clasificador; no son similitud vectorial.
4. Usa obtener_clases_audio para inspeccionar todas las clases AudioSet detectadas por YAMNet en un segmento recuperado. Explica que son etiquetas en inglés y scores del clasificador.
5. Usa obtener_info_segmento cuando se pidan detalles de un segmento identificado.
6. Responde en español, salvo que el usuario pida otro idioma.
7. No inventes información: limita todas las afirmaciones a los resultados de las tools.
8. Para cada hallazgo, cita archivo de origen, timestamp de inicio y fin, e índice consultado.
   Usa exactamente el campo search_index_label devuelto por la tool.
9. Si no hay resultados, dilo claramente y sugiere una reformulación.
10. Si un hallazgo trae clip_url, incluí un enlace Markdown con el texto "▶ Reproducir audio" usando esa URL.`;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }

  const body = (req.body ?? {}) as {
    messages?: ChatMessage[];
    dataset_id?: string;
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (!messages.length) {
    res.status(400).json({ detail: "messages es obligatorio." });
    return;
  }
  const defaultDatasetId = body.dataset_id?.trim() || undefined;

  try {
    const modelName = process.env.TEXT_GENERATION_MODEL?.trim() || "openai/gpt-4o-mini";

    const result = await generateText({
      model: gateway(modelName),
      system: SYSTEM_PROMPT,
      messages: messages.map((m) => ({
        role: m.role === "assistant" ? "assistant" : "user",
        content: String(m.content ?? ""),
      })),
      stopWhen: stepCountIs(5),
      tools: {
        buscar_audio: tool({
          description: "Busca texto semánticamente en las transcripciones del corpus de audio.",
          inputSchema: z.object({
            query: z.string().min(1),
            k: z.number().int().min(1).max(20).default(5),
          }),
          execute: async ({ query, k }) => {
            const { results } = await searchSemanticNeon(query, k, defaultDatasetId);
            return {
              status: "success",
              modality: "text",
              results: serializeAgentResults(
                results,
                "text",
                "Índice de texto (transcripciones)",
              ),
            };
          },
        }),
        buscar_evento_acustico: tool({
          description:
            "Busca eventos acústicos a partir de una descripción textual (aplausos, música, gritos, risas).",
          inputSchema: z.object({
            query: z.string().min(1),
            k: z.number().int().min(1).max(20).default(5),
          }),
          execute: async ({ query, k }) => {
            const { results, translatedQuery } = await searchAudioNeon(query, k, {
              datasetId: defaultDatasetId,
            });
            return {
              status: "success",
              modality: "audio",
              translated_query: translatedQuery,
              results: serializeAgentResults(results, "audio", "Índice de audio (CLAP)"),
            };
          },
        }),
        buscar_clase_audio: tool({
          description: "Busca segmentos por clases acústicas AudioSet detectadas por YAMNet.",
          inputSchema: z.object({
            query: z.string().min(1),
            k: z.number().int().min(1).max(20).default(5),
          }),
          execute: async ({ query, k }) => {
            const { results, translatedQuery } = await searchYamnetNeon(query, k, {
              datasetId: defaultDatasetId,
            });
            return {
              status: "success",
              modality: "yamnet",
              translated_query: translatedQuery,
              results: serializeAgentResults(
                results,
                "yamnet",
                "Clases de audio (YAMNet/AudioSet)",
              ),
            };
          },
        }),
        obtener_info_segmento: tool({
          description: "Obtiene los metadatos completos de un segmento recuperado previamente.",
          inputSchema: z.object({
            segment_id: z.number().int().min(0),
          }),
          execute: async ({ segment_id }) => {
            const segment = await getSegmentInfoNeon(segment_id, defaultDatasetId);
            if (!segment) return { status: "error", error: `Segmento ${segment_id} no encontrado.` };
            return { status: "success", segment };
          },
        }),
        obtener_clases_audio: tool({
          description: "Obtiene las clases de AudioSet detectadas por YAMNet para un segmento.",
          inputSchema: z.object({
            segment_id: z.number().int().min(0),
          }),
          execute: async ({ segment_id }) => {
            const classes = await getAudioClassesNeon(segment_id, defaultDatasetId);
            if (classes === null) {
              return { status: "error", error: `Segmento ${segment_id} no encontrado.` };
            }
            return { status: "success", classifier: "yamnet", classes };
          },
        }),
      },
    });

    res.status(200).json({
      reply: result.text || "No obtuve respuesta del agente.",
    });
  } catch (error) {
    console.error("Agent chat failed:", error);
    res.status(500).json({
      detail: error instanceof Error ? error.message : "Error al consultar el agente.",
    });
  }
}

export default withCors(handler);
