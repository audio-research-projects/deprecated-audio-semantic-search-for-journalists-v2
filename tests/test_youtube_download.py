"""Regression coverage for resuming YouTube ingestion without new downloads."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.youtube_download import download_youtube_audio


def test_reuses_existing_audio_with_literal_video_id(tmp_path, monkeypatch):
    video_id = "DVt_mzCOGtA"
    existing = tmp_path / f"Debate [{video_id}].m4a"
    existing.write_bytes(b"existing audio")
    (tmp_path / "Other [different-id].m4a").write_bytes(b"unrelated audio")
    ydl = MagicMock()
    ydl.extract_info.return_value = {"id": video_id, "title": "Debate", "duration": 60}
    downloader = MagicMock()
    downloader.return_value.__enter__.return_value = ydl
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=downloader))

    result = download_youtube_audio(f"https://www.youtube.com/watch?v={video_id}", tmp_path)

    assert result.path == existing
    assert result.video_id == video_id
    assert result.duration_sec == 60.0
