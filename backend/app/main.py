import logging

from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.services.inference import InferenceService
from app.schemas import BatchEvaluationRequest, BatchEvaluationResponse, EvaluationResponse, PitchInput

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
inference_service = InferenceService(window_seconds=settings.chunk_window_seconds)

# Log startup mode
logger.info(
    f"App started | {settings.app_name} v{settings.app_version} | "
    f"use_heuristic_pipeline={settings.use_heuristic_pipeline} | "
    f"use_local_transcriber={settings.use_local_transcriber}"
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.post("/evaluate")
def evaluate_pitch(payload: PitchInput) -> EvaluationResponse:
    return inference_service.evaluate_payload(payload)


@app.post(
    "/evaluate/batch",
    responses={400: {"description": "Bad Request: pitches list cannot be empty"}},
)
def evaluate_pitch_batch(payload: BatchEvaluationRequest) -> BatchEvaluationResponse:
    try:
        return inference_service.evaluate_batch(payload.pitches)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
