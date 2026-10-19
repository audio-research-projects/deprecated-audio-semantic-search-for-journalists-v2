"""YouTube audio download for the ingestion pipeline (via yt-dlp).

Downloads the best audio stream and extracts it to a pipeline-compatible
format (``.m4a`` by default; ``src/audio_conversion.SUPPORTED_FORMATS``).
Requires ``yt-dlp`` (``uv`` ``ml`` group) and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_FORMAT = "m4a"


@dataclass(frozen=True)
class DownloadedAudio:
    url: str
    path: Path
    title: str
    video_id: str
    duration_sec: float | None


def download_youtube_audio(
    url: str,
    output_dir: str | Path,
    *,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> DownloadedAudio:
    """Download a YouTube video's audio track into ``output_dir``.

    Returns the extracted audio file. Existing files are reused when the same
    video was already downloaded (yt-dlp skips the transfer).
    """
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError(
            "yt-dlp is required for YouTube ingestion. "
            "Install the ml group: uv sync --group ml"
        ) from error

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in out_dir.iterdir()}

    # "<title> [<video id>]" keeps readable names while staying unique.
    outtmpl = str(out_dir / "%(title).120B [%(id)s].%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": audio_format}
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = str(info.get("id", ""))
    title = str(info.get("title", video_id))
    duration = info.get("duration")
    downloaded = [
        path
        for path in out_dir.iterdir()
        if path.name not in before and path.suffix.lower() == f".{audio_format}"
    ]
    if not downloaded:
        # Already downloaded in a previous run: locate the file by video id.
        downloaded = sorted(
            path
            for path in out_dir.iterdir()
            if path.is_file() and path.name.endswith(f" [{video_id}].{audio_format}")
        )
    if not downloaded:
        raise RuntimeError(f"yt-dlp finished but no audio file was produced for {url}")

    path = downloaded[0]
    logger.info("Downloaded YouTube audio: %s -> %s", title, path.name)
    return DownloadedAudio(
        url=url,
        path=path,
        title=title,
        video_id=video_id,
        duration_sec=float(duration) if duration is not None else None,
    )
