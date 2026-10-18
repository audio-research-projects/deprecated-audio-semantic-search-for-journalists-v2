"""Audio search engine with FAISS-based semantic search."""

import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import faiss
import pandas as pd

from src.clap_audio_embeddings import CLAPEmbedding
from src.query_translation import translate_to_english
from src.segment_clip_storage import SegmentClipStore
from src.text_embeddings import TextEmbeddingModel
from src.vector_indexing import load_faiss_index, search_faiss_index
from src.yamnet_inverted_index import (
    YAMNET_INVERTED_INDEX_FILENAME,
    load_yamnet_inverted_index,
    normalized_audio_class_tokens,
)

logger = logging.getLogger(__name__)

_MIN_SEGMENT_DURATION_ENV = "MIN_SEGMENT_DURATION_SECONDS"
_DEFAULT_MIN_SEGMENT_DURATION_SECONDS = 5.0


def _configured_min_segment_duration() -> float:
    """Read and validate the minimum duration applied to search results."""
    raw_value = os.getenv(_MIN_SEGMENT_DURATION_ENV, "").strip()
    if not raw_value:
        return _DEFAULT_MIN_SEGMENT_DURATION_SECONDS
    try:
        duration = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{_MIN_SEGMENT_DURATION_ENV} must be a non-negative number") from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"{_MIN_SEGMENT_DURATION_ENV} must be a non-negative number")
    return duration


class AudioSearchEngine:
    """Motor de búsqueda semántica sobre el corpus de audio indexado."""

    def __init__(
        self,
        dataset_path: str,
        text_model: TextEmbeddingModel | None = None,
        clap_model: CLAPEmbedding | None = None,
        clips_gcs_uri: str | None = None,
    ):
        """
        Carga dataset, embeddings e índices FAISS.

        Args:
            dataset_path: Path to the processed dataset directory
            text_model: Optional shared text embedding model to avoid loading it twice
            clap_model: Optional shared CLAP model to avoid loading it twice
            clips_gcs_uri: Optional GCS prefix for playback clips of this dataset
        """
        self.dataset_path = Path(dataset_path)
        self._df: pd.DataFrame | None = None
        self._text_index: faiss.IndexFlatIP | None = None
        self._audio_index: faiss.IndexFlatIP | None = None
        self._text_model: TextEmbeddingModel | None = text_model
        self._clap_model: CLAPEmbedding | None = clap_model
        self._active_embeddings: set[str] | None = None
        self._active_classifiers: set[str] | None = None
        self._dataset_version: str | None = None
        self._segment_positions: dict[int, int] = {}
        self._yamnet_inverted_index: dict[str, list[dict[str, object]]] = {}
        self._min_segment_duration_seconds = _configured_min_segment_duration()
        self._clip_store = SegmentClipStore(
            dataset_path=self.dataset_path, clips_gcs_uri=clips_gcs_uri
        )

        self._load_dataset()
        self._load_indices()

    def _load_dataset(self):
        """Load the complete dataset."""
        pkl_path = self.dataset_path / "final" / "complete_dataset.pkl"
        if pkl_path.exists():
            self._df = pd.read_pickle(pkl_path)
            self._segment_positions = {
                int(segment_id): position
                for position, segment_id in enumerate(self._df["segment_id"])
            }
            logger.info("Loaded dataset: %d segments from %s", len(self._df), pkl_path)
            manifest_path = self.dataset_path / "final" / "dataset_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                active_embeddings = manifest.get("active_embeddings")
                if isinstance(active_embeddings, list):
                    self._active_embeddings = {str(name) for name in active_embeddings}
                active_classifiers = manifest.get("active_classifiers")
                if isinstance(active_classifiers, list):
                    self._active_classifiers = {str(name) for name in active_classifiers}
                elif isinstance(manifest.get("classifiers"), dict):
                    self._active_classifiers = set(manifest["classifiers"])
                elif self._active_embeddings is not None:
                    # Releases created before CLASSIFIERS_ACTIVE listed YAMNet
                    # alongside vector embeddings. Preserve their availability.
                    self._active_classifiers = (
                        {"yamnet"} if "yamnet" in self._active_embeddings else set()
                    )
                version = manifest.get("dataset_version") or manifest.get("release")
                if isinstance(version, str):
                    self._dataset_version = version
        else:
            raise FileNotFoundError(
                f"Dataset not found at {pkl_path}. Run the ingestion pipeline first."
            )

    def _load_indices(self):
        """Load FAISS indices."""
        indices_dir = self.dataset_path / "indices"

        text_index_path = indices_dir / "text_index.faiss"
        if self._is_embedding_active("text") and text_index_path.exists():
            self._text_index = load_faiss_index(str(text_index_path))
            logger.info("Loaded text index: %d vectors", self._text_index.ntotal)

        audio_index_path = indices_dir / "audio_index.faiss"
        if self._is_embedding_active("clap") and audio_index_path.exists():
            self._audio_index = load_faiss_index(str(audio_index_path))
            logger.info("Loaded audio index: %d vectors", self._audio_index.ntotal)

        yamnet_index_path = indices_dir / YAMNET_INVERTED_INDEX_FILENAME
        if self._is_classifier_active("yamnet") and yamnet_index_path.exists():
            try:
                self._yamnet_inverted_index = load_yamnet_inverted_index(yamnet_index_path)
            except (OSError, ValueError, json.JSONDecodeError):
                logger.exception("Could not load YAMNet inverted index from %s", yamnet_index_path)
            else:
                logger.info(
                    "Loaded YAMNet inverted index: %d tokens", len(self._yamnet_inverted_index)
                )

    def _is_embedding_active(self, embedding_name: str) -> bool:
        """Respect manifest configuration while retaining legacy dataset support."""
        return self._active_embeddings is None or embedding_name in self._active_embeddings

    def _is_classifier_active(self, classifier_name: str) -> bool:
        """Respect the classifier manifest, including pre-migration releases."""
        if self._active_classifiers is not None:
            return classifier_name in self._active_classifiers
        return self._active_embeddings is None or classifier_name in self._active_embeddings

    @property
    def text_model(self) -> TextEmbeddingModel:
        if self._text_model is None:
            self._text_model = TextEmbeddingModel()
        return self._text_model

    @property
    def clap_model(self) -> CLAPEmbedding:
        if self._clap_model is None:
            self._clap_model = CLAPEmbedding()
        return self._clap_model

    @property
    def total_segments(self) -> int:
        return len(self._df) if self._df is not None else 0

    @property
    def active_indexes(self) -> list[str]:
        """Search sources currently available and therefore safe to offer."""
        indexes = [
            name
            for name, index in (("text", self._text_index), ("audio", self._audio_index))
            if index is not None
        ]
        if self.yamnet_available:
            indexes.append("yamnet")
        return indexes

    @property
    def yamnet_available(self) -> bool:
        """Whether stored YAMNet labels can be searched in this dataset."""
        if (
            self._df is None
            or "yamnet_top_classes" not in self._df.columns
            or not self._is_classifier_active("yamnet")
        ):
            return False
        return bool(self._yamnet_inverted_index)

    @property
    def dataset_version(self) -> str | None:
        return self._dataset_version

    @property
    def min_segment_duration_seconds(self) -> float:
        """Minimum strict duration for segments returned by searches."""
        return getattr(
            self,
            "_min_segment_duration_seconds",
            _DEFAULT_MIN_SEGMENT_DURATION_SECONDS,
        )

    def _segment_is_long_enough(self, row: pd.Series) -> bool:
        """Return whether a row passes the configured search-result duration filter."""
        try:
            duration = float(row.get("end_time", 0.0)) - float(row.get("start_time", 0.0))
        except (TypeError, ValueError):
            return False
        return math.isfinite(duration) and duration > self.min_segment_duration_seconds

    def _candidate_k(self, index: faiss.IndexFlatIP, k: int) -> int:
        """Over-fetch enough FAISS candidates to fill ``k`` after filtering."""
        index_size = getattr(index, "ntotal", None)
        if index_size is None:
            index_size = len(self._df) if self._df is not None else k
        if self._df is None:
            return min(k, index_size)
        short_segments = sum(
            not self._segment_is_long_enough(row) for _, row in self._df.iterrows()
        )
        return min(index_size, k + short_segments)

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Búsqueda semántica por texto (transcripciones).

        Args:
            query_text: Natural language query
            k: Number of results to return

        Returns:
            List of dicts with segment info and similarity scores
        """
        if self._text_index is None:
            raise RuntimeError("Text index not loaded")

        query_embedding = self.text_model.generate_embedding(query_text, normalize=True)
        similarities, indices = search_faiss_index(
            self._text_index,
            query_embedding,
            k=self._candidate_k(self._text_index, k),
        )

        results = []
        for sim, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._df):
                continue
            row = self._df.iloc[idx]
            if not self._segment_is_long_enough(row):
                continue
            results.append(
                {
                    # AudioSet labels describe the same segment independently of
                    # the retrieval source. Keep them in text results too so the
                    # UI can expose acoustic evidence alongside the transcript.
                    "segment": self._row_to_segment_dict(row, include_audio_classes=True),
                    "similarity": float(sim),
                    "distance": float(1.0 - sim),
                }
            )
            if len(results) >= k:
                break

        return results

    def search_audio_by_classes(
        self,
        query_text: str,
        k: int = 5,
        *,
        source_language: str | None = None,
    ) -> list[dict]:
        """Search stored YAMNet/AudioSet labels using an English text query.

        YAMNet is a classifier, not a vector index. It uses the release's
        inverted token index instead of scanning the complete dataset. Results
        are ranked by query-term coverage multiplied by the mean classifier
        score of the matched classes.
        """
        if not self.yamnet_available:
            raise RuntimeError(
                "YAMNet classes not available. Reprocess the dataset with yamnet enabled."
            )

        translated_query = translate_to_english(query_text, source_language=source_language)
        query_tokens = normalized_audio_class_tokens(translated_query)
        if not query_tokens:
            return []

        candidates: dict[int, dict[str, dict[str, object]]] = {}
        for token in sorted(query_tokens):
            for posting in self._yamnet_inverted_index.get(token, []):
                segment_id = int(posting["segment_id"])
                class_id = str(posting["class_id"])
                segment_matches = candidates.setdefault(segment_id, {})
                match = segment_matches.setdefault(
                    class_id,
                    {
                        "class_id": class_id,
                        "class_name": str(posting["class_name"]),
                        "score": float(posting["score"]),
                        "class_rank": int(posting["class_rank"]),
                        "covered_tokens": set(),
                    },
                )
                match["covered_tokens"].add(token)

        ranked: list[tuple[float, int, list[dict]]] = []
        for segment_id, matches_by_class in candidates.items():
            position = self._segment_positions.get(segment_id)
            if position is None:
                continue
            row = self._df.iloc[position]
            if not self._segment_is_long_enough(row):
                continue
            ordered_matches = sorted(
                matches_by_class.values(), key=lambda item: int(item["class_rank"])
            )
            matched_classes = [
                {
                    "class_id": str(match["class_id"]),
                    "class_name": str(match["class_name"]),
                    "score": float(match["score"]),
                }
                for match in ordered_matches
            ]
            covered_tokens = set().union(
                *(match["covered_tokens"] for match in ordered_matches)
            )

            coverage = len(covered_tokens) / len(query_tokens)
            classifier_score = sum(item["score"] for item in matched_classes) / len(
                matched_classes
            )
            rank_score = coverage * classifier_score
            ranked.append((rank_score, position, matched_classes))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[dict] = []
        for rank_score, position, matched_classes in ranked[:k]:
            segment = self._row_to_segment_dict(
                self._df.iloc[position], include_audio_classes=True
            )
            segment["yamnet_matched_classes"] = matched_classes
            results.append(
                {
                    "segment": segment,
                    "similarity": float(rank_score),
                    "distance": float(1.0 - rank_score),
                }
            )
        return results

    @staticmethod
    def _normalized_audio_class_tokens(value: str) -> set[str]:
        """Normalize English AudioSet class names and natural-language queries."""
        return normalized_audio_class_tokens(value)

    def search_audio_by_text(
        self,
        query_text: str,
        k: int = 5,
        *,
        source_language: str | None = None,
    ) -> list[dict]:
        """
        Búsqueda cross-modal texto→audio usando CLAP.

        Args:
            query_text: Text query (e.g., "applause", "music")
            k: Number of results
            source_language: Language of ``query_text``. ``None`` uses the
                configured query language; pass ``en`` for an already translated query.

        Returns:
            List of dicts with segment info and similarity scores
        """
        if self._audio_index is None:
            raise RuntimeError("Audio index not loaded")

        query_embedding = self.clap_model.generate_text_embedding(
            query_text, source_language=source_language
        )
        similarities, indices = search_faiss_index(
            self._audio_index,
            query_embedding,
            k=self._candidate_k(self._audio_index, k),
        )

        results = []
        for sim, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._df):
                continue
            row = self._df.iloc[idx]
            if not self._segment_is_long_enough(row):
                continue
            results.append(
                {
                    "segment": self._row_to_segment_dict(row, include_audio_classes=True),
                    "similarity": float(sim),
                    "distance": float(1.0 - sim),
                }
            )
            if len(results) >= k:
                break

        return results

    def get_segment_info(self, segment_id: int) -> dict | None:
        """
        Get full information for a specific segment.

        Args:
            segment_id: The segment ID

        Returns:
            Dict with segment info or None if not found
        """
        if self._df is None:
            return None

        matches = self._df[self._df["segment_id"] == segment_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        return self._row_to_segment_dict(row, include_sentiment=True, include_audio_classes=True)

    def get_audio_classes(self, segment_id: int) -> list[dict] | None:
        """Return YAMNet AudioSet labels stored for one processed segment.

        ``None`` means the segment does not exist; an empty list means that the
        dataset was created without YAMNet classification enabled.
        """
        if self._df is None:
            return None
        matches = self._df[self._df["segment_id"] == segment_id]
        if matches.empty:
            return None
        return self._parse_audio_classes(matches.iloc[0].get("yamnet_top_classes", []))

    @staticmethod
    def _parse_audio_classes(value: object) -> list[dict]:
        """Read current list values and CSV-compatible JSON values safely."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if not isinstance(value, list):
            return []
        classes: list[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                classes.append(
                    {
                        "class_id": str(item["class_id"]),
                        "class_name": str(item["class_name"]),
                        "score": float(item["score"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return classes

    def _row_to_segment_dict(
        self,
        row: pd.Series,
        include_sentiment: bool = False,
        include_audio_classes: bool = False,
    ) -> dict:
        """Convert a DataFrame row to a segment dict."""
        result = {
            "segment_id": int(row.get("segment_id", 0)),
            "text": str(row.get("text", "")),
            "start_time": float(row.get("start_time", 0.0)),
            "end_time": float(row.get("end_time", 0.0)),
            "original_file_name": str(row.get("original_file_name", "")),
            "language": str(row.get("language", "unknown")),
            "confidence": float(row.get("confidence", 0.0)),
        }

        # Playback clip, present only in datasets ingested with clips enabled.
        # The clip window is wider than the segment: it carries the surrounding
        # context, so consumers need both pairs of timestamps to position the player.
        clip_name = str(row.get("clip_file_name", "") or "")
        if clip_name:
            result.update(
                {
                    "clip_file_name": clip_name,
                    "clip_start_time": float(row.get("clip_start_time", result["start_time"])),
                    "clip_end_time": float(row.get("clip_end_time", result["end_time"])),
                }
            )
            try:
                reference = self._clip_store.reference(int(result["segment_id"]))
            except Exception:  # noqa: BLE001 - playback must not disable search.
                logger.warning(
                    "Could not create playback URL for segment %s",
                    result["segment_id"],
                    exc_info=True,
                )
            else:
                if reference is not None:
                    result["clip_url"] = reference.url
                    expires_at = getattr(reference, "expires_at", None)
                    if expires_at is not None:
                        result["clip_expires_at"] = datetime.fromtimestamp(
                            expires_at, tz=UTC
                        ).isoformat()

        if include_sentiment:
            result.update(
                {
                    "sentiment_positive": float(row.get("sentiment_positive", 0.0)),
                    "sentiment_negative": float(row.get("sentiment_negative", 0.0)),
                    "sentiment_neutral": float(row.get("sentiment_neutral", 0.0)),
                    "dominant_sentiment": str(row.get("dominant_sentiment", "neutral")),
                }
            )

        if include_audio_classes:
            result["yamnet_audio_classes"] = self._parse_audio_classes(
                row.get("yamnet_top_classes", [])
            )

        return result
