# Búsqueda Agéntica de Audio para Periodistas

Sistema de **búsqueda agéntica multimodal** (texto y audio) diseñado para periodistas. Un agente de IA con acceso a herramientas realiza búsqueda semántica sobre un corpus de audios previamente transcritos e indexados, combinando embeddings de texto y embeddings acústicos.

## Stack Tecnológico

|Componente|Tecnología|
|---|---|
|Lenguaje|Python 3.11|
|Gestión de dependencias|uv|
|Transcripción|OpenAI Whisper|
|Embeddings de texto|Sentence Transformers (all-MiniLM-L6-v2)|
|Embeddings de audio|CLAP (LAION)|
|Indexación vectorial|FAISS|
|Agente|Google ADK + LiteLLM/OpenAI GPT-4o-mini|
|API REST|FastAPI + Uvicorn|
|Evaluación|RAGAS o DeepEval + métricas IR custom|

## Setup

### Prerequisitos

```bash
# Python 3.11
pyenv install 3.11.13
pyenv local 3.11.13

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# ffmpeg (macOS)
brew install ffmpeg
```

### Instalación

```bash
# Instalar dependencias base
uv sync

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY

# Instalar pre-commit hooks
uv run pre-commit install

# Dependencias opcionales de evaluación
uv sync --group eval          # RAGAS
uv sync --extra eval-deepeval # DeepEval
```

## Uso

Para procesar WAV, OPUS u otros audios reales, seguí la guía operativa
[Crear un dataset con audios reales](docs/guia-dataset-real.md).

### Configurar embeddings y clasificadores activos

La fuente de verdad es `.env`. Antes de cada ejecución de la CLI, el pipeline
genera [config/embeddings.toml](config/embeddings.toml) a partir de
`EMBEDDINGS_ACTIVE`, `CLASSIFIERS_ACTIVE` y las variables de modelo; no edites
el TOML manualmente. Por ejemplo, para
habilitar Gemini junto con los índices actuales:

```dotenv
GEMINI_API_KEY=tu-clave
EMBEDDINGS_ACTIVE=text,clap,gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY=1536
```

- `text`: MiniLM para consulta↔transcripción (`text_index.faiss`).
- `clap`: CLAP para consulta↔audio (`audio_index.faiss`).
- `gemini`: Gemini Embedding 2 nativo para consulta↔audio
  (`gemini_audio_index.faiss`); requiere `GEMINI_API_KEY`.
- `embeddinggemma`: EmbeddingGemma 2 local para consulta↔audio
  (`embeddinggemma_audio_index.faiss`, 768 dimensiones por defecto). Reemplaza
  a CLAP sin llamadas a una API de inferencia. Requiere el extra
  `embeddinggemma`, que fija Transformers con soporte para este modelo.
- `yamnet`: clasificador AudioSet por segmento. Guarda etiquetas y scores de
  eventos acústicos, pero no crea un índice FAISS. Requiere `uv sync --extra yamnet`.

La configuración base habilita YAMNet con `CLASSIFIERS_ACTIVE=yamnet`; volver a
procesar el corpus al cambiar ese valor. Las demás variables disponibles son
`TEXT_EMBEDDING_MODEL`, `CLAP_EMBEDDING_MODEL`, `YAMNET_MODEL` y `YAMNET_TOP_K`;
están documentadas en `.env.example`.
Las clases se consultan mediante `obtener_clases_audio` después de recuperar un
segmento y también se pueden buscar directamente con `buscar_clase_audio` o
`POST /search/yamnet`. YAMNet usa etiquetas AudioSet en inglés, no crea un
espacio vectorial y es complementario a CLAP. Los resultados CLAP incluyen sus
clases YAMNet cuando el dataset las contiene.

`EMBEDDINGS_CONFIG_PATH` define dónde se escribe el TOML generado (por defecto,
`./config/embeddings.toml`); `--embeddings-config` permite cambiar esa ruta por
ejecución. El manifiesto final registra por separado los embeddings y los
clasificadores activos, sus modelos y sus artefactos.

Las ejecuciones posteriores son incrementales: se reutilizan los resultados de
audios sin cambios y sólo se procesan archivos nuevos o modificados. La huella
de cada fuente y la firma de configuración se guardan en
`final/ingestion_state.json`; cambiar chunking, Whisper, ventanas o embeddings
fuerza una reconstrucción completa para conservar consistencia.

### 1. Pipeline de Ingesta

Procesa archivos de audio y genera un dataset indexado:

```bash
uv run python -m src.simple_dataset_pipeline \
    --input data/ \
    --output ./dataset \
    --whisper-model base \
    --batch-size 8 \
    --verbose
```

Para testing sin CLAP (más rápido):

```bash
uv run python -m src.simple_dataset_pipeline \
    --input data/ \
    --output ./dataset \
    --mock-audio \
    --verbose
```

### 2. Servicio del Agente

```bash
# Iniciar API
uv run uvicorn src.fast_api_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

### 3. Consultar al Agente

```bash
# POST request
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Busca segmentos sobre política económica", "max_results": 5}'

# Health check
curl http://localhost:8000/health
```

## Evaluación

### Validar CLAP contra Clotho (sanity check de implementación)

Antes de sacar conclusiones sobre CLAP en el corpus propio, conviene descartar
un bug de integración corriéndolo contra el banco de pruebas público en el que
fue entrenado.

```bash
uv run python -m benchmarks.evaluate_clap_clotho \
    --audio-dir data/clotho/evaluation \
    --captions-csv data/clotho/clotho_captions_evaluation.csv \
    --output evaluation/results/clap_clotho_eval.json \
    --cache evaluation/results/.clap_clotho_audio_embeddings.npy
```

Un buen resultado acá descarta un bug de implementación; no valida que CLAP
funcione en audio periodístico en español, que es un dominio distinto.

### Comparativa: esquema actual vs. Gemini Embedding 2

El benchmark no modifica el servicio de producción. Evalúa por separado `minilm_text`
(consulta↔transcripción), `clap_audio` (consulta↔audio) y
`gemini_embedding_2_audio` (consulta↔WAV) sobre los mismos segmentos y queries
anotadas. Requiere ventanas acústicas y `relevant_segment_ids` no vacíos.

```bash
export GEMINI_API_KEY="..."
uv run python -m benchmarks.compare_retrieval_with_gemini \
    --dataset-path ./dataset \
    --queries evaluation/test_datasets/retrieval_eval_dataset.json \
    --output evaluation/results/gemini_comparison.json
```

### Retrieval (RAG aislado)

```bash
uv run python -m evaluation.retrieval_evaluation \
    --dataset evaluation/test_datasets/retrieval_eval_dataset.json \
    --dataset-path ./dataset \
    --output evaluation/results/retrieval_results.json
```

### Generar un dataset balanceado conceptual/acústico

Para estudiar si CLAP mejora cuando las preguntas sí mencionan eventos sonoros,
se puede generar un dataset pareado: una pregunta de contenido y una acústica
por segmento. Los resultados de este experimento están en la documentación
interna del proyecto.

```bash
# Generar en español (idioma por defecto del proyecto)
uv run python -m evaluation.generate_balanced_questions \
    --max-segments 10 \
    --output evaluation/test_datasets/synthetic/balanced_journalistic_questions_es.json

# Evaluar CLAP sobre ese dataset
uv run python -m benchmarks.compare_clap_by_question_type \
    --dataset-path ./dataset \
    --questions evaluation/test_datasets/synthetic/balanced_journalistic_questions_es.json \
    --output evaluation/results/clap_retrieval_by_question_type_es.json
```

> Nota: el text encoder de CLAP (`laion/clap-htsat-unfused`) está entrenado en
> inglés. Si `QUERY_LANGUAGE` en `.env` no es inglés, el sistema traduce
> automáticamente la query al inglés antes de generar el embedding de CLAP.
> Para medir el límite del modelo sin la barrera del idioma, también podés
> generar el dataset balanceado en inglés (`--language en`).

### Datasets de evaluación

Existen tres fuentes de preguntas, controlables desde `.env`:

- `manual`: preguntas escritas a mano.
- `synthetic`: preguntas generadas por RAGAS a partir del corpus procesado.
- `hybrid`: unión del manual y el sintético, sin duplicados.

Variables relevantes en `.env`:

```dotenv
EVALUATION_DATASET_SOURCE=manual
EVALUATION_DATASET_SIZE=20
EVALUATION_DATASET_MAX_SEGMENTS=0
EVALUATION_MANUAL_DATASET_PATH=./evaluation/test_datasets/ragas_eval_dataset_unlabeled.json
EVALUATION_SYNTHETIC_DATASET_PATH=./evaluation/test_datasets/synthetic/ragas_synthetic_questions.json
EVALUATION_HYBRID_DATASET_PATH=./evaluation/test_datasets/synthetic/ragas_hybrid_questions.json
```

### Filtro de duración de segmentos

Las búsquedas sólo devuelven segmentos cuya duración sea estrictamente mayor a
`MIN_SEGMENT_DURATION_SECONDS`. El valor predeterminado es `5` segundos; se
puede cambiar en Cloud Run al desplegar el search service:

```bash
gcloud run services update audio-search-service \
  --region REGION --project PROJECT \
  --update-env-vars "MIN_SEGMENT_DURATION_SECONDS=8"
```

Usá `0` para aceptar cualquier segmento con duración positiva. Los endpoints
de consulta directa (`/search/*`) aplican este filtro; la consulta puntual
`GET /segments/:id` continúa permitiendo inspeccionar cualquier segmento.

### Traducción para CLAP y YAMNet en Cloud Run

CLAP y las etiquetas AudioSet de YAMNet se consultan en inglés. Si las consultas
del frontend están en español, el `audio-search-service` debe tener
`QUERY_LANGUAGE=es` y recibir `OPENAI_API_KEY` desde Secret Manager. Su service
account necesita `roles/secretmanager.secretAccessor` sobre ese secreto. Por
ejemplo, al actualizar el servicio:

```bash
gcloud run services update audio-search-service \
  --region REGION --project PROJECT \
  --update-env-vars "QUERY_LANGUAGE=es" \
  --update-secrets "OPENAI_API_KEY=audio-search-openai-api-key:latest"
```

Para generar preguntas sintéticas (requiere `OPENAI_API_KEY` y el corpus en `DATASET_PATH`):

```bash
# 20 preguntas sobre todo el corpus
uv run python -m evaluation.generate_synthetic_questions \
    --dataset-path ./dataset \
    --size 20

# También se puede definir cantidad y límite de segmentos por env:
# EVALUATION_DATASET_SIZE=30
# EVALUATION_DATASET_MAX_SEGMENTS=50

# Si hay problemas de memoria/segfault en MPS (Apple Silicon), usar 1 worker
# y embeddings en CPU:
# EVALUATION_GENERATION_MAX_WORKERS=1
# uv run python -m evaluation.generate_synthetic_questions --size 5
```

El resultado se escribe en `EVALUATION_SYNTHETIC_DATASET_PATH` y contiene una
lista de `question` (y opcionalmente `ground_truth`/`ground_truth_contexts`,
generados por RAGAS).

### Dataset híbrido

Para combinar el manual y el sintético en la misma evaluación:

```dotenv
EVALUATION_DATASET_SOURCE=hybrid
```

```bash
# Primero generar las sintéticas si aún no existen
uv run python -m evaluation.generate_synthetic_questions --size 20

# Evaluar con la unión (el runner elimina duplicados, prioriza el manual)
uv run python -m evaluation.run_agent_evaluation \
    --agent-url http://localhost:8000 \
    --output evaluation/results/hybrid_evaluation.json
```

El resultado combinado se guarda en `EVALUATION_HYBRID_DATASET_PATH`. Notá que
las métricas con `ground_truth` (Context Precision, Context Recall y Answer
Correctness) sólo se activan cuando **todas** las muestras del híbrido traen
`ground_truth`.

### Evaluación RAG del agente (RAGAS o DeepEval)

```bash
# Requiere servicio corriendo
# En .env: EVALUATION_FRAMEWORK=ragas (default) o deepeval
# Instalar: uv sync --group eval (RAGAS) o uv sync --extra eval-deepeval (DeepEval)

# Usa el dataset definido por EVALUATION_DATASET_SOURCE
uv run python -m evaluation.run_agent_evaluation \
    --agent-url http://localhost:8000 \
    --output evaluation/results/agent_evaluation.json

# Sobrescribir el dataset por línea de comandos
uv run python -m evaluation.run_agent_evaluation \
    --dataset evaluation/test_datasets/ragas_eval_dataset.json \
    --agent-url http://localhost:8000 \
    --output evaluation/results/agent_evaluation.json
```

### Evaluación aislada del índice textual y revisión humana de CLAP

La evaluación determinista del índice textual usa las preguntas generadas con
RAGAS y alinea sus contextos de referencia con los segmentos actuales:

```bash
UV_CACHE_DIR=.uv-cache HF_HUB_OFFLINE=1 uv run python \
  -m evaluation.text_index_evaluation \
  --dataset evaluation/test_datasets/synthetic/ragas_synthetic_questions.json \
  --dataset-path "$DATASET_PATH" \
  --minimum-segment-duration 0 \
  --output evaluation/results/text_index_ragas_raw.json
```

La muestra acústica preparada se revisa desde una interfaz local:

```bash
UV_CACHE_DIR=.uv-cache uv run uvicorn evaluation.human_review.app:app \
  --host 127.0.0.1 --port 8010
```

Abrir `http://127.0.0.1:8010`. La metodología, resultados y limitaciones de la
revisión humana, así como el protocolo operativo, están en la documentación
interna del proyecto.

## Tests

```bash
uv run pytest tests/ -v
```

## Estructura del Proyecto

```text
├── src/                           # Código fuente
│   ├── agent_service/             # Agente ADK y tools de retrieval
│   │   ├── agent.py               # root_agent ADK + Runner compatible
│   │   ├── search_engine.py       # Motor FAISS
│   │   └── tools.py               # Tools del agente
│   ├── fast_api_app.py            # Superficie ADK de producción
│   ├── embedding_config.py        # Modalidades activas de ingesta
│   ├── audio_conversion.py        # ffmpeg conversion
│   ├── audio_transcription.py     # Whisper
│   ├── text_embeddings.py         # MiniLM / Sentence Transformers
│   ├── clap_audio_embeddings.py   # CLAP
│   ├── gemini_multimodal_embeddings.py # Gemini Embedding 2 opcional
│   ├── yamnet_audio_classifier.py # Clasificador YAMNet opcional
│   ├── sentiment_analysis.py      # Sentiment analysis
│   ├── vector_indexing.py         # FAISS indexing
│   └── simple_dataset_pipeline.py # Pipeline orchestrator
├── evaluation/                    # Framework de evaluación
│   ├── ragas_evaluation.py        # RAGAS evaluation
│   ├── retrieval_evaluation.py    # IR metrics
│   └── test_datasets/             # Evaluation datasets
├── tests/                         # Unit tests
├── data/                          # Audio source (not versioned)
├── dataset/                       # Processed dataset (not versioned)
├── spec/                          # Specification documents
├── pyproject.toml                 # Dependencias del proyecto
├── uv.lock                        # Lockfile reproducible de uv
└── .env.example                   # Environment template
```
