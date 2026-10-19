import { defineAgent } from "eve";

export default defineAgent({
  model: process.env.TEXT_GENERATION_MODEL?.trim() || "openai/gpt-4o-mini",
});
