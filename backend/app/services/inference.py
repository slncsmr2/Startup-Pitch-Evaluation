from __future__ import annotations

import logging
import hashlib
from pathlib import Path

from app.core.config import settings
from app.pipeline import StartupPitchPipeline
from app.schemas import BatchEvaluationResponse, EvaluationResponse, PitchInput, PitchVideoInput

logger = logging.getLogger(__name__)


class InferenceService:
    """Shared inference entrypoint for CLI and FastAPI.

    This service keeps video inference behavior consistent across interfaces and
    ensures local transcriber settings flow through preprocessing.
    """

    def __init__(self, window_seconds: int | None = None) -> None:
        self.pipeline = StartupPitchPipeline(
            window_seconds=window_seconds or settings.chunk_window_seconds
        )

    @staticmethod
    def _build_request_id(payload: PitchInput) -> str:
        serialized = payload.model_dump_json()
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
        return f"req-{digest}"

    def evaluate_payload(self, payload: PitchInput, request_id: str | None = None) -> EvaluationResponse:
        inference_id = request_id or self._build_request_id(payload)
        return self.pipeline.evaluate(payload, request_id=inference_id)

    def evaluate_batch(self, pitches: list[PitchInput]) -> BatchEvaluationResponse:
        if not pitches:
            raise ValueError("pitches list cannot be empty")
        evaluations = [self.evaluate_payload(pitch) for pitch in pitches]
        return BatchEvaluationResponse(evaluations=evaluations)

    def infer_video(
        self,
        video_path: str,
        duration_sec: int = 60,
        transcript_text: str = "",
        title: str = "CLI Video Inference",
        language_hint: str = "en-ta",
    ) -> EvaluationResponse:
        safe_title = title.strip() or Path(video_path).stem or "untitled-pitch"
        payload = PitchInput(
            title=safe_title,
            transcript="",
            language_hint=language_hint,
            slide_text=[],
            video=PitchVideoInput(
                file_name=video_path,
                file_format=Path(video_path).suffix.replace(".", "") or "mp4",
                duration_sec=max(5, int(duration_sec)),
                transcript_text=transcript_text,
            ),
        )

        logger.info(
            "Running shared inference | video=%s | use_local_transcriber=%s",
            video_path,
            settings.use_local_transcriber,
        )
        return self.evaluate_payload(payload)
