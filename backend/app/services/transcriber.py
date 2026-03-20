"""Transcription abstraction with deterministic fallback behavior.

Only OpenAI Whisper API is supported.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.services.audio_processor import AudioChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    backend: str
    status: str
    reason: str = ""


class BaseLocalTranscriber(ABC):
    def __init__(self, backend_name: str, min_audio_quality: float = 0.35):
        self.backend_name = backend_name
        self.min_audio_quality = min_audio_quality

    @abstractmethod
    def transcribe_audio_file(self, audio_file_path: str, language_hint: str = "") -> TranscriptionResult:
        """Backend-specific transcription for a local audio file."""

    def transcribe_chunk(
        self,
        audio_file_path: str,
        audio_metadata: AudioChunkMetadata,
        fallback_text: str,
        language_hint: str = "",
    ) -> TranscriptionResult:
        """Transcribe a chunk with deterministic fallback for poor/missing audio."""
        if audio_metadata.audio_quality_score < self.min_audio_quality:
            return self._fallback(
                fallback_text=fallback_text,
                confidence=0.2,
                status="fallback-low-quality",
                reason=f"audio_quality={audio_metadata.audio_quality_score:.2f}",
            )

        if audio_metadata.silence_ratio >= 0.95:
            return self._fallback(
                fallback_text=fallback_text,
                confidence=0.15,
                status="fallback-silent",
                reason=f"silence_ratio={audio_metadata.silence_ratio:.2f}",
            )

        audio_path = Path(audio_file_path)
        if not audio_path.exists() or not audio_path.is_file():
            return self._fallback(
                fallback_text=fallback_text,
                confidence=0.35,
                status="fallback-no-audio-file",
                reason=f"audio_file_missing={audio_file_path}",
            )

        result = self.transcribe_audio_file(audio_file_path, language_hint=language_hint)
        if result.text.strip():
            return result

        return self._fallback(
            fallback_text=fallback_text,
            confidence=min(result.confidence, 0.3),
            status="fallback-empty-transcript",
            reason=result.reason or "backend returned empty transcript",
        )

    def _fallback(self, fallback_text: str, confidence: float, status: str, reason: str) -> TranscriptionResult:
        text = fallback_text.strip() or "[inaudible chunk]"
        return TranscriptionResult(
            text=text,
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            backend=self.backend_name,
            status=status,
            reason=reason,
        )


class OpenAIWhisperAPITranscriber(BaseLocalTranscriber):
    def __init__(
        self,
        api_key: str,
        model_name: str = "whisper-1",
        min_audio_quality: float = 0.35,
    ):
        super().__init__(backend_name="openai-whisper-api", min_audio_quality=min_audio_quality)
        self.api_key = api_key.strip()
        self.model_name = model_name.strip() or "whisper-1"
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=self.api_key)
            return self._client
        except Exception as exc:
            logger.warning("OpenAI transcriber backend unavailable: %s", exc)
            return None

    def transcribe_audio_file(self, audio_file_path: str, language_hint: str = "") -> TranscriptionResult:
        client = self._get_client()
        if client is None:
            return self._fallback(
                fallback_text="",
                confidence=0.0,
                status="backend-unavailable",
                reason="openai client or API key not available",
            )

        audio_path = Path(audio_file_path)
        if not audio_path.exists() or not audio_path.is_file():
            return self._fallback(
                fallback_text="",
                confidence=0.0,
                status="backend-unavailable",
                reason=f"audio_file_missing={audio_file_path}",
            )

        try:
            normalized_hint = language_hint.strip().lower()
            request_kwargs: dict = {
                "model": self.model_name,
            }
            if normalized_hint in {"en", "ta"}:
                request_kwargs["language"] = normalized_hint

            with audio_path.open("rb") as audio_file:
                output = client.audio.transcriptions.create(file=audio_file, **request_kwargs)

            text = (getattr(output, "text", "") or "").strip()
            return TranscriptionResult(
                text=text,
                confidence=0.85 if text else 0.0,
                backend=self.backend_name,
                status="ok" if text else "empty",
                reason="",
            )
        except Exception as exc:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                backend=self.backend_name,
                status="error",
                reason=str(exc),
            )


def build_local_transcriber(
    min_audio_quality: float = 0.35,
    openai_api_key: str = "",
    openai_model_name: str = "whisper-1",
) -> BaseLocalTranscriber:
    return OpenAIWhisperAPITranscriber(
        api_key=openai_api_key,
        model_name=openai_model_name,
        min_audio_quality=min_audio_quality,
    )
