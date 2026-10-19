import { issueSignedToken, presignUrl } from "@vercel/blob";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getClipBlobReferenceNeon } from "../../src/db.js";

const SIGNED_URL_TTL_MS = 5 * 60 * 1000;
const SIGNED_TOKEN_TTL_MS = 60 * 60 * 1000;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.setHeader("Allow", "GET, HEAD");
    res.status(405).end();
    return;
  }

  // The Vercel handler gets the id in query; the Express service gets it in params.
  const routeParams = (req as VercelRequest & { params?: Record<string, string> }).params;
  const rawSegmentId = req.query.segmentId ?? routeParams?.segmentId;
  const segmentId = Number(Array.isArray(rawSegmentId) ? rawSegmentId[0] : rawSegmentId);
  const datasetId = typeof req.query.dataset_id === "string" ? req.query.dataset_id.trim() : "";
  if (!Number.isInteger(segmentId) || segmentId < 0 || !datasetId) {
    res.status(400).json({ detail: "segmentId y dataset_id son obligatorios." });
    return;
  }

  try {
    const clipReference = await getClipBlobReferenceNeon(segmentId, datasetId);
    if (!clipReference) {
      res.status(404).json({ detail: "Clip no encontrado." });
      return;
    }

    // Neon stores the private Blob pathname, not its host URL. Accept absolute
    // URLs from older releases too, but only from this project's private store.
    const storeId = process.env.BLOB_STORE_ID?.trim();
    let pathname: string;
    if (/^https:\/\//i.test(clipReference)) {
      let parsed: URL;
      try {
        parsed = new URL(clipReference);
      } catch {
        res.status(404).json({ detail: "Clip no encontrado." });
        return;
      }
      const storeHostnameId = storeId?.replace(/^store_/i, "").toLowerCase();
      const validPrivateBlobHost = storeHostnameId
        ? parsed.hostname === `${storeHostnameId}.private.blob.vercel-storage.com`
        : /^[a-z0-9-]+\.private\.blob\.vercel-storage\.com$/i.test(parsed.hostname);
      if (parsed.protocol !== "https:" || !validPrivateBlobHost) {
        res.status(404).json({ detail: "Clip no encontrado." });
        return;
      }
      pathname = decodeURIComponent(parsed.pathname.replace(/^\//, ""));
    } else {
      pathname = decodeURIComponent(clipReference.replace(/^\/+/, ""));
    }

    const prefix = `audio-search/${datasetId}/`;
    const suffix = `/segment_clips/segment_${segmentId}.opus`;
    const release = pathname.startsWith(prefix)
      ? pathname.slice(prefix.length, -suffix.length)
      : "";
    if (
      !release ||
      release.includes("/") ||
      !pathname.endsWith(suffix)
    ) {
      res.status(404).json({ detail: "Clip no encontrado." });
      return;
    }

    const now = Date.now();
    const signedToken = await issueSignedToken({
      pathname,
      operations: ["get", "head"],
      validUntil: now + SIGNED_TOKEN_TTL_MS,
    });
    const { presignedUrl } = await presignUrl(signedToken, {
      operation: req.method === "HEAD" ? "head" : "get",
      pathname,
      access: "private",
      validUntil: now + SIGNED_URL_TTL_MS,
    });

    res.setHeader("Cache-Control", "private, no-store");
    res.redirect(307, presignedUrl);
  } catch (error) {
    console.error("Private clip signing failed:", error);
    res.status(500).json({ detail: "No se pudo preparar el enlace temporal al clip." });
  }
}
