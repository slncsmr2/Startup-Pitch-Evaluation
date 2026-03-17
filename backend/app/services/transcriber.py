"""Local transcription abstraction with deterministic fallback behavior.

This module intentionally avoids any network behavior. Backends only run when
the required package and local model artifacts are available.
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


class WhisperLocalTranscriber(BaseLocalTranscriber):
    def __init__(
        self,
        model_path: str,
        min_audio_quality: float = 0.35,
    ):
        super().__init__(backend_name="whisper", min_audio_quality=min_audio_quality)
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path or not Path(self.model_path).exists():
            return None
        try:
            import whisper  # type: ignore

            self._model = whisper.load_model(self.model_path)
            return self._model
        except Exception as exc:
            logger.warning("Whisper backend unavailable: %s", exc)
            return None

    def transcribe_audio_file(self, audio_file_path: str, language_hint: str = "") -> TranscriptionResult:
        model = self._load_model()
        if model is None:
            return self._fallback(
                fallback_text="",
                confidence=0.0,
                status="backend-unavailable",
                reason="whisper model/package not available locally",
            )
        try:
            kwargs = {}
            normalized_hint = language_hint.strip().lower()
            if normalized_hint in {"en", "ta"}:
                kwargs["language"] = normalized_hint
            output = model.transcribe(audio_file_path, **kwargs)
            text = (output.get("text") or "").strip()
            confidence = 0.8 if text else 0.0
            return TranscriptionResult(
                text=text,
                confidence=confidence,
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


class FasterWhisperLocalTranscriber(BaseLocalTranscriber):
    def __init__(
        self,
        model_path: str,
        min_audio_quality: float = 0.35,
    ):
        super().__init__(backend_name="faster-whisper", min_audio_quality=min_audio_quality)
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path or not Path(self.model_path).exists():
            return None
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(self.model_path, device="cpu", compute_type="int8")
            return self._model
        except Exception as exc:
            logger.warning("faster-whisper backend unavailable: %s", exc)
            return None

    def transcribe_audio_file(self, audio_file_path: str, language_hint: str = "") -> TranscriptionResult:
        model = self._load_model()
        if model is None:
            return self._fallback(
                fallback_text="",
                confidence=0.0,
                status="backend-unavailable",
                reason="faster-whisper model/package not available locally",
            )
        try:
            normalized_hint = language_hint.strip().lower()
            language = normalized_hint if normalized_hint in {"en", "ta"} else None
            segments, info = model.transcribe(audio_file_path, language=language)
            text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            confidence = float(getattr(info, "language_probability", 0.8)) if text else 0.0
            return TranscriptionResult(
                text=text,
                confidence=round(max(0.0, min(1.0, confidence)), 2),
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
    backend: str,
    model_path: str,
    min_audio_quality: float = 0.35,
) -> BaseLocalTranscriber:
    normalized = backend.strip().lower()
    if normalized in {"faster-whisper", "faster_whisper", "faster"}:
        return FasterWhisperLocalTranscriber(model_path=model_path, min_audio_quality=min_audio_quality)
    if normalized == "whisper":
        return WhisperLocalTranscriber(model_path=model_path, min_audio_quality=min_audio_quality)
    logger.warning("Unknown transcriber backend '%s'; defaulting to faster-whisper", backend)
    return FasterWhisperLocalTranscriber(model_path=model_path, min_audio_quality=min_audio_quality)
