# Despliegue en Vercel con Neon Postgres (pgvector) y Eve Agents

Este directorio contiene el segundo destino de despliegue del proyecto (**Vercel + Neon + Eve Agents**), paralelo al despliegue en Google Cloud Platform (`deploy/gcp/`).

## Arquitectura en Vercel

```text
web/ — SPA React + Vite (búsqueda directa + drawer del Eve Agent)
       │
       ├──► /eve/v1/* (Eve Agent Runtime — servicio generado por withEve en vercel.ts)
       │        ├─ agent/agent.ts (defineAgent)
       │        ├─ agent/instructions.md
       │        ├─ agent/channels/eve.ts (vercelOidc + localDev + bearer/anon)
       │        └─ agent/tools/*.ts (buscar_audio, buscar_evento_acustico, buscar_clase_audio, ...)
       │
       └──► /api/* (Direct Search API — servicio "api", Express en server.ts)
                ├─ POST /api/search y /api/search/plan
                ├─ POST /api/agent-chat (fallback same-origin para el drawer, mismas tools)
                ├─ GET  /api/datasets, /api/corpus, /api/segments/:segmentId
                ├─ POST /api/feedback
                └─ GET  /api/health
                        │
                        ▼
       ┌────────────────────────────────────────────────────────────┐
       │ Neon Serverless Postgres (pgvector) + Vercel Blob (.opus)  │
       │  • datasets, audio_segments, segment_yamnet_postings       │
       │  • Embeddings de consulta vía API (HF / Gemini / OpenAI)   │
       └────────────────────────────────────────────────────────────┘
```

`vercel.ts` compone el grafo de servicios con `withEve` — **3 servicios**: `web` (framework `vite`, estáticos de `dist-web/`), `api` (framework `express`, `server.ts` monta los handlers de `api/` en una sola Function — en modo services Vercel no construye el directorio `api/` suelto) y `eve` (generado automáticamente, montado en `/eve/v1/*`).

---

## Paso 1: Procesar el corpus localmente (con embeddings compatibles con API)

En Vercel Serverless no se carga PyTorch (~1.7 GB de CLAP), por lo que las consultas generan embeddings por API y buscan en **Neon `pgvector`**.

Para usar `all-MiniLM-L6-v2` (texto, 384d vía HuggingFace Inference API) y `gemini-embedding-2` (audio multimodal nativo, 1536d vía Gemini API):

```bash
# En la raíz del repositorio (.env):
EMBEDDINGS_ACTIVE=text,gemini
CLASSIFIERS_ACTIVE=yamnet
GEMINI_API_KEY=tu-clave-gemini

uv run python -m src.simple_dataset_pipeline \
  --input data/ \
  --output ./dataset
```

*(Si ya tenés un dataset con `text` y `clap`, `publish_dataset_neon.py` también soporta exportar `clap` y usar un endpoint externo en `CUSTOM_EMBEDDING_API_URL`).*

### Alternativa: EmbeddingGemma 2 en lugar de CLAP

`EMBEDDINGS_ACTIVE` también acepta `embeddinggemma`: [EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) (`google/embeddinggemma-2`, Apache 2.0) mapea **texto y audio al mismo espacio de 768d** vía `sentence-transformers>=6.1`, sin dependencia de `laion-clap`. Corre local (CPU, ~570M params cargando solo los encoders de texto+audio) y sus vectores son truncables por Matryoshka (`EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY=768|512|256|128`).

```bash
EMBEDDINGS_ACTIVE=text,embeddinggemma \
EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY=768 \
uv run python -m src.simple_dataset_pipeline --input data/ --output ./dataset
```

En el lado de consulta (Vercel no puede correr el modelo), seteá `AUDIO_EMBEDDING_PROVIDER=embeddinggemma` + `EMBEDDINGGEMMA_API_URL` apuntando al servicio de referencia en `deploy/vercel/embeddinggemma-service/` (FastAPI, `POST /embed {text, modality} → {embedding}` — Dockerfile incluido para Cloud Run). La dimensión del servicio debe coincidir con `EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY` de la ingesta.

### Ingesta directa desde YouTube

`scripts/ingest_youtube_neon.py` orquesta todo el flujo (descarga `yt-dlp` → pipeline de ingesta → Neon + Vercel Blob) a partir de uno o más links:

```bash
EMBEDDINGS_ACTIVE=text,embeddinggemma \
DATABASE_URL="postgresql://..." \
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_..." \
uv run python scripts/ingest_youtube_neon.py \
  --youtube-url "https://www.youtube.com/watch?v=DVt_mzCOGtA" \
  --dataset ./dataset-youtube \
  --dataset-id youtube \
  --label "YouTube ingest"
```

- Las descargas quedan en `./youtube-audio/` (override: `--download-dir`) y se **reutilizan entre corridas** — el pipeline es incremental sobre ese directorio, así que agregar un video nuevo no reprocesa los anteriores.
- Flags útiles: `--skip-publish` (solo procesar local), `--skip-pipeline` (solo republicar un dataset ya procesado), `--dry-run`, `--audio-embedding-source embeddinggemma`, `--whisper-model`, `--mock-audio`.
- Requiere `ffmpeg` en PATH (extracción `yt-dlp` + conversión a WAV) y el grupo `ml` de uv.

---

## Paso 2: Provisionar Neon Postgres y Vercel Blob

Desde `deploy/vercel/`:

```bash
cd deploy/vercel
npm install

# Vincular con tu proyecto en Vercel
vercel link

# Agregar las integraciones de Neon Postgres y Vercel Blob
vercel integration add neon
vercel storage add blob

# Descargar variables de entorno (.env.local)
vercel env pull .env.local
```

---

## Paso 3: Publicar el dataset local en Neon + Vercel Blob

Desde la **raíz del repositorio**, ejecutá `scripts/publish_dataset_neon.py` pasando `DATABASE_URL` y `BLOB_READ_WRITE_TOKEN`:

```bash
# Vista previa sin escribir en la base ni subir archivos (--dry-run)
uv run python scripts/publish_dataset_neon.py \
  --dataset ./dataset \
  --dataset-id default \
  --label "Archivo Periodístico" \
  --dry-run

# Publicación real en Neon Postgres + subida de clips .opus a Vercel Blob
DATABASE_URL="postgresql://..." \
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_..." \
uv run --with 'psycopg[binary]' python scripts/publish_dataset_neon.py \
  --dataset ./dataset \
  --dataset-id default \
  --label "Archivo Periodístico"
```

El script aplica automáticamente `deploy/vercel/db/schema.sql` (`CREATE EXTENSION IF NOT EXISTS vector`, tablas `datasets`, `audio_segments`, `segment_yamnet_postings` y `feedback_logs`), sube los clips `.opus` a Vercel Blob y guarda la URL en `audio_segments.clip_url`.

---

## Paso 4: Variables de entorno

Ver `.env.example` para la lista completa y comentada. Mínimo funcional:

```bash
DATABASE_URL=postgresql://...   # Neon Postgres + pgvector (obligatoria)
OPENAI_API_KEY=sk-...           # Eve agent, planner y traducción (obligatoria)
GEMINI_API_KEY=...              # solo si usás el índice acústico (default: gemini)
HF_API_TOKEN=hf_...             # recomendada para embeddings de texto vía HF
```

Reglas importantes:

- **Los providers/dimensiones de embeddings deben coincidir con lo indexado** en `audio_segments.text_embedding` / `audio_embedding` por `publish_dataset_neon.py` (`TEXT_EMBEDDING_*` y `AUDIO_EMBEDDING_*`, default: HF MiniLM 384d + Gemini 1536d).
- `ALLOWED_ORIGINS` habilita CORS real en `/api/*` (`src/cors.ts`): solo hace falta si consumís la API desde otro origen (p. ej. `frontend-direct-search` en GitHub Pages). Sin definir, no se emiten headers CORS.
- `EVE_PUBLIC_BEARER_TOKEN` opcional: protege `/eve/v1/*` con bearer token; sin definir, el canal queda público anónimo.
- `BLOB_READ_WRITE_TOKEN` solo lo usa el publisher, no el runtime.
- `VITE_API_BASE_URL` vacío = same-origin (default para esta SPA).

Para producción, definilas en los settings del proyecto Vercel (`vercel env add` o dashboard), y localmente corré `vercel env pull .env.local`.

## Paso 5: Ejecutar localmente y desplegar en Vercel

```bash
cd deploy/vercel

# Desarrollo local del agente Eve + rutas API
npm run dev

# Desarrollo local del frontend (proxy /api y /eve → localhost:3000)
npm run dev:web

# Stack completo con el grafo de servicios (requiere Vercel CLI >= 59.16.0)
vercel dev --local

# Typecheck, tests del backend y build del frontend
npm run typecheck
npm test
npm run build:web

# Desplegar a producción en Vercel
eve deploy
```

El frontend de `web/` se sirve desde el mismo origen (`VITE_API_BASE_URL` vacío por defecto). También podés conectar `frontend-direct-search` compilándolo con `VITE_API_BASE_URL` apuntando al dominio de tu proyecto Vercel.

### Frontend (`web/`)

Adaptación del `frontend-direct-search` a las rutas de Vercel (detalle completo en [FRONTEND.md](./FRONTEND.md)):

- Mismo contrato de API: `POST /api/search`, `GET /api/datasets`, filtros por archivo/tiempo, plan de búsqueda editable y columnas por índice (texto / audio / YAMNet).
- Reproducción de clips `.opus` desde **Vercel Blob** (URLs públicas, sin firma GCS).
- UI bilingüe (ES/EN), atajos `J`/`K`/`Espacio`, estado en la URL (`q`, `dataset`, `text`, `clap`, `yamnet`, `k`, `lang`).
- **Drawer del Eve Agent**: botón flotante que abre un chat same-origin contra `POST /api/agent-chat`, que ejecuta las mismas tools del agente (`buscar_audio`, `buscar_evento_acustico`, `buscar_clase_audio`, `obtener_info_segmento`, `obtener_clases_audio`) con el AI SDK. Incluye feedback thumbs-up/down hacia `/api/feedback`.
  - Cuando quieras el canal Eve completo (streaming, sesiones durables), podés reemplazar el drawer por `useEveAgent()` de `eve/react` apuntando a `/eve/v1` sin cambiar el backend.
