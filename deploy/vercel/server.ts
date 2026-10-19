/**
 * API service entrypoint for the Vercel services graph.
 *
 * Vercel does not build the zero-config `api/` directory inside a service
 * ("The api/ directory will not be built because services are configured"),
 * so this Express app mounts the same handler functions on /api/* and is
 * deployed as its own `api` service — see services.api in vercel.ts.
 * Handlers keep the plain VercelRequest/VercelResponse shape, which Express
 * requests/responses already satisfy at runtime (status/json/setHeader/end,
 * parsed query and body via express.json()).
 */

import express from "express";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import agentChat from "./api/agent-chat.js";
import clip from "./api/clips/[segmentId].js";
import corpus from "./api/corpus.js";
import datasets from "./api/datasets.js";
import feedback from "./api/feedback.js";
import health from "./api/health.js";
import search from "./api/search.js";
import searchPlan from "./api/search/plan.js";
import segment from "./api/segments/[segmentId].js";

type ApiHandler = (req: VercelRequest, res: VercelResponse) => unknown | Promise<unknown>;

const mount =
  (handler: ApiHandler): express.RequestHandler =>
  (req, res, next) => {
    Promise.resolve(
      handler(req as unknown as VercelRequest, res as unknown as VercelResponse),
    ).catch(next);
  };

const app = express();
app.use(express.json({ limit: "1mb" }));

app.all("/api/health", mount(health));
app.all("/api/datasets", mount(datasets));
app.all("/api/corpus", mount(corpus));
app.all("/api/search", mount(search));
app.all("/api/search/plan", mount(searchPlan));
app.all("/api/agent-chat", mount(agentChat));
app.all("/api/feedback", mount(feedback));
app.all("/api/clips/:segmentId", mount(clip));
app.all("/api/segments/:segmentId", mount(segment));

// Standalone local run (`node server.ts` / `tsx server.ts`). Inside `vercel dev`
// and production builds VERCEL is set and the runtime wraps the exported app.
if (!process.env.VERCEL) {
  const port = Number(process.env.PORT ?? 3001);
  app.listen(port, () => {
    console.log(`API service listening on http://localhost:${port}`);
  });
}

export default app;
