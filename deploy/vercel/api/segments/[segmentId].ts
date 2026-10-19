import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getSegmentInfoNeon } from "../../src/db.js";
import { withCors } from "../../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  const routeParams = (req as VercelRequest & { params?: Record<string, string> }).params;
  const rawId = req.query.segmentId ?? routeParams?.segmentId;
  const segmentId = Number(Array.isArray(rawId) ? rawId[0] : rawId);
  if (!Number.isInteger(segmentId) || segmentId < 0) {
    res.status(400).json({ detail: "segmentId debe ser un entero no negativo." });
    return;
  }
  try {
    const datasetId = typeof req.query.dataset_id === "string" ? req.query.dataset_id : undefined;
    const segment = await getSegmentInfoNeon(segmentId, datasetId);
    if (!segment) {
      res.status(404).json({ detail: `Segmento ${segmentId} no encontrado.` });
      return;
    }
    res.status(200).json({ segment });
  } catch (error) {
    console.error("Failed to fetch segment from Neon:", error);
    res.status(502).json({ detail: "No fue posible consultar el segmento." });
  }
}

export default withCors(handler);
