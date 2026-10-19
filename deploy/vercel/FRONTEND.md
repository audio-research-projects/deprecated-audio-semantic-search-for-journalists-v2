# Frontend de búsqueda en Vercel (`web/`)

Documento de implementación del frontend de búsqueda para **Vercel + Neon (pgvector) + Eve Agents**. La reproducción usa clips en Vercel Blob privado con URLs firmadas de corta duración.

## Estado de despliegue

La Preview actual está **Ready** en <https://audio-semantic-search-olisyswq4-hordiales-projects.vercel.app> con Deployment Protection habilitada. Neon y el store privado Blob están conectados y el dataset Massa–Milei está publicado. No se ha probado una solicitud de búsqueda o reproducción contra el deployment; ese es el siguiente paso. Production no se ha configurado en este flujo.

Para re-deployar cambios:

```bash
cd deploy/vercel
vercel deploy --yes         # Preview; Production requiere una decisión aparte
```

## Arquitectura

```text
Browser (SPA React en dist-web/)
   │
   ├──► POST /api/agent-chat   (drawer del asistente, mismas tools del Eve agent)
   ├──► POST /api/feedback
   ├──► POST /api/search       ┐
   ├──► POST /api/search/plan  │  Direct Search API
   ├──► GET  /api/datasets     │  (servicio "api" — server.ts Express)
   ├──► GET  /api/corpus       │
   ├──► GET  /api/segments/:id │
   ├──► GET  /api/clips/:id    │  URL firmada de cinco minutos → Blob privado
   ├──► GET  /api/health       ┘
   └──► /eve/v1/*              Eve Agent Runtime (servicio generado por withEve)
            │
            ▼
   Neon Postgres (pgvector) + Vercel Blob privado + servicio EmbeddingGemma 2 + AI Gateway
```

`vercel.ts` compone el grafo con `withEve` — **3 servicios**:

- `web` (framework `vite`, root `.`): `npm run build:web` → `dist-web/`. Solo estáticos.
- `api` (framework `express`, entrypoint `server.ts`): los handlers de `api/` montados en `/api/*`, empaquetados como una sola Vercel Function. **Importante**: en modo `services`, Vercel NO construye el directorio `api/` como functions sueltas ("The api/ directory will not be built because services are configured") — por eso existe `server.ts`.
- `eve` (generado por `withEve`, root aislado `.eve/vercel-services/eve`): runtime completo del agente en `/eve/v1/*` (health, info, session, callbacks, workflows).

Rutas: `/eve/v1/*` → eve, `/api/*` → api, resto → web. Los handlers de `api/` usan paths nativos (sin rewrites):

| Ruta | Archivo |
|---|---|
| `POST /api/search` | `api/search.ts` |
| `POST /api/search/plan` | `api/search/plan.ts` |
| `POST /api/agent-chat` | `api/agent-chat.ts` |
| `GET /api/datasets` | `api/datasets.ts` |
| `GET /api/corpus` | `api/corpus.ts` |
| `GET /api/segments/:segmentId` | `api/segments/[segmentId].ts` |
| `POST /api/feedback` | `api/feedback.ts` |
| `GET /api/health` | `api/health.ts` |

## Archivos del frontend

| Archivo | Rol |
|---|---|
| `web/App.tsx` | UI completa: toggles de índice, datasets, plan editable, columnas por índice, filtros, player de audio, reproducción continua, atajos de teclado, estado en URL |
| `web/AgentDrawer.tsx` | Drawer flotante del asistente: chat contra `/api/agent-chat`, feedback 👍/👎 a `/api/feedback` |
| `web/api.ts` | Cliente fetch (`VITE_API_BASE_URL`, vacío = same-origin) |
| `web/types.ts` | Contrato `SearchRequest`/`SearchResponse`/`SearchPlan`/`IndexResults` |
| `web/i18n.ts` | Textos ES/EN, `detectLocale()` por URL/localStorage/navegador |
| `web/styles.css` | Sistema visual completo + estilos del drawer |
| `web/main.tsx` | Bootstrap React |
| `index.html` | Shell de la SPA (entrada `/web/main.tsx`) |
| `vite.config.ts` | `base: "/"`, outDir `dist-web`, proxy local `/api` y `/eve` → `localhost:3000` (solo en `npm run dev:web`; desactivado bajo `VERCEL=1` porque en `vercel dev` las funciones van al servicio `api`) |
| `server.ts` | Entrypoint Express del servicio `api`: monta los handlers de `api/` en rutas `/api/*`; `app.listen` solo fuera de Vercel |
| `tsconfig.web.json` | Typecheck separado del frontend (DOM, `jsx: react-jsx`, bundler resolution) |

## Comportamiento preservado del frontend original

- Índice de **texto activo por defecto**; CLAP y YAMNet desactivados; al menos un índice debe quedar seleccionado.
- Resultados agrupados por índice (scores nunca se mezclan), con `effective_query` y `translated_query` visibles.
- **Plan de búsqueda** visible, copiable y editable; editar descripciones re-ejecuta el retrieval.
- **Playback**: usa `clip_start_time`/`clip_end_time` y timestamps del segmento; renovación de URLs expiradas repitiendo el plan guardado (`clip_expires_at`).
- Atajos `J`/`K`/`Espacio`; filtros por archivo y rango temporal; copiar cita; descargar `.opus`.
- **Diferencia clave**: los clips vienen de Vercel Blob (URL pública) — no se aplica la validación de URLs firmadas GCS del widget CopilotKit original.

## Eve Agent: dos puntos de entrada

1. **Drawer incluido** (`AgentDrawer` → `POST /api/agent-chat`): endpoint same-origin que ejecuta las mismas cinco tools del agente (`buscar_audio`, `buscar_evento_acustico`, `buscar_clase_audio`, `obtener_info_segmento`, `obtener_clases_audio`) con `generateText` del AI SDK y `stopWhen: stepCountIs(5)`. Funciona sin canal Eve completo.
2. **Canal Eve real** (`/eve/v1/*`): `agent/channels/eve.ts` monta `eveChannel` con auth `vercelOidc()` + `localDev()` + bearer opcional (`EVE_PUBLIC_BEARER_TOKEN`) + fallback anónimo `none()`. Para usarlo desde el frontend se puede reemplazar el drawer por `useEveAgent()` de `eve/react` sin tocar el backend.

## Notas de configuración

- `agent/sandbox.ts` fija `JustBashSandbox` (sandbox JS puro): las tools solo consultan Neon, no necesitan contenedor. Evita la dependencia de microsandbox/Docker en local y de Vercel Sandbox en producción.
- Peer deps de `eve@0.80`: `ai@^7`, `@ai-sdk/openai@^4`, `zod@^3.25`.
- Scripts: `dev` (eve dev), `dev:web` (vite), `build` (eve build + tsc), `build:web`, `typecheck`, `test`.
- Variables: ver `.env.example` (comentado por secciones: obligatorias, providers de embedding, opcionales).
- **CORS**: `src/cors.ts` implementa `withCors` en todos los handlers de `api/` respetando `ALLOWED_ORIGINS` (lista separada por comas o `*`, con preflight `OPTIONS`). Necesario solo si el frontend vive en otro origen.
- CLAP/PyTorch **no** está en el bundle del frontend ni en las Functions — los embeddings de consulta se generan por API (ver `plan.md` y `src/embeddings.ts`).

## Verificación

```text
npm run typecheck   → tsc --noEmit (backend) + tsc -p tsconfig.web.json (web)  ✓
npm test            → 16 tests (search logic + CORS)  ✓
npm run build:web   → vite build: dist-web/ (≈174 kB JS / 56 kB gzip)  ✓
npm run build       → eve build (bundle del agente) + tsc --noEmit  ✓
vercel build        → services/web estáticos + services/api index.func + services/eve protocolo completo  ✓
vercel dev --local  → / 200, /api/health 200, /eve/v1/health 200, /eve/v1/info 200  ✓
vercel deploy --prod → https://audio-semantic-search.vercel.app (3 servicios vivos)  ✓
```

Preview ya tiene Neon, Blob privado y el dataset publicados. Pendiente: validar búsqueda y reproducción desde la interfaz protegida y luego decidir si configurar Production. Opcionalmente migrar el drawer a `useEveAgent()` para streaming/sesiones durables del canal Eve.
