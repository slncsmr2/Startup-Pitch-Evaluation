from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.pipeline import StartupPitchPipeline
from app.schemas import BatchEvaluationRequest, BatchEvaluationResponse, EvaluationResponse, PitchInput

app = FastAPI(title=settings.app_name, version=settings.app_version)
pipeline = StartupPitchPipeline(window_seconds=settings.chunk_window_seconds)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.post("/evaluate")
def evaluate_pitch(payload: PitchInput) -> EvaluationResponse:
    request_id = str(uuid4())
    return pipeline.evaluate(payload, request_id)


@app.post(
    "/evaluate/batch",
    responses={400: {"description": "Bad Request: pitches list cannot be empty"}},
)
def evaluate_pitch_batch(payload: BatchEvaluationRequest) -> BatchEvaluationResponse:
    if not payload.pitches:
        raise HTTPException(status_code=400, detail="pitches list cannot be empty")

    evaluations = [
        pipeline.evaluate(single_pitch, request_id=str(uuid4()))
        for single_pitch in payload.pitches
    ]
    return BatchEvaluationResponse(evaluations=evaluations)
