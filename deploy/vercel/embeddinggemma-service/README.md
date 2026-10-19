# Inferencia de consultas de texto

`POST /embed` recibe una consulta escrita y devuelve un embedding de 768
componentes compatible con el índice de audio generado con EmbeddingGemma 2.
Solo se carga el encoder de texto (271M parámetros); los encoders de audio y
visión están deshabilitados. No hace falta volver a procesar el corpus.

```json
{"text": "inflación y economía argentina"}
```

La respuesta tiene la forma `{"embedding": [...]}`. Se aplica `SearchQuery` y
normalización L2. Se acepta `modality: "text"` para mantener el contrato del
cliente existente. Se rechazan audio, URLs de audio, texto vacío y consultas
que superen 2.048 caracteres con HTTP 422.

## Vercel

`../vercel.ts` registra el servicio privado `embeddings`, con FastAPI, memoria
máxima de 2.048 MB y duración máxima de 300 segundos. La API y el agente Eve
reciben `EMBEDDINGGEMMA_SERVICE_URL` mediante bindings internos; no hay una
ruta pública directa al modelo ni hace falta configurar un dominio externo.
El navegador sigue utilizando `/api/search`.

Configurar `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` en el entorno de despliegue y
habilitar Fluid Compute con Active CPU. El paquete incluye aproximadamente
1 GiB de pesos y excede el límite estándar de Python. Ver
[Large Functions](https://vercel.com/docs/functions/limitations) y
[bindings entre servicios](https://vercel.com/docs/services/bindings).

El servicio usa Python 3.12 en Vercel, con dependencias propias en
`requirements.txt`. La versión local del pipeline continúa siendo Python 3.11
administrado con pyenv.

Durante el build, `python prepare_model.py` guarda en `model/` únicamente los
pesos del encoder de texto. Transformers y el checkpoint están fijados a las
revisiones probadas con el corpus. El runtime carga ese paquete localmente,
sin descargar pesos en cada arranque. La inferencia se serializa por instancia
para evitar cargas simultáneas del modelo y picos de memoria por concurrencia.

La prueba local del modelo empaquetado registró un pico de 1,35 GiB y una
diferencia máxima de 3e-8 respecto de la consulta equivalente con texto+audio.
Es una medición de macOS. El build remoto de Python 3.12 completó la
preparación y el empaquetado con Large Functions el 18 de octubre de 2026;
la [preview](https://audio-semantic-search-pu72bqotk-hordiales-projects.vercel.app)
quedó en estado Ready. Siguen pendientes las mediciones de memoria, arranque
en frío y latencia durante inferencia en Vercel.

## Desarrollo local

Desde la raíz del repositorio, usando el entorno existente:

```bash
.venv/bin/python deploy/vercel/embeddinggemma-service/prepare_model.py
.venv/bin/python deploy/vercel/embeddinggemma-service/main.py
```

`GET /health` informa el estado y si el modelo ya se cargó. El `Dockerfile`
es una alternativa de empaquetado del mismo servicio para contenedores.
Los pesos generados en `model/` están excluidos de Git.
