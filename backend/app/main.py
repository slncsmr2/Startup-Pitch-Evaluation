import logging
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.services.inference import InferenceService
from app.schemas import BatchEvaluationRequest, BatchEvaluationResponse, EvaluationResponse, PitchInput

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
inference_service = InferenceService(window_seconds=settings.chunk_window_seconds)
static_dir = Path(__file__).resolve().parent / "static"
batch_input_dir = Path(__file__).resolve().parent.parent / "outputs" / "batch_input"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Log startup mode
logger.info(
    f"App started | {settings.app_name} v{settings.app_version} | "
    f"use_heuristic_pipeline={settings.use_heuristic_pipeline} | "
    f"use_local_transcriber={settings.use_local_transcriber}"
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(static_dir / "index.html")


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


@app.get("/videos")
def list_videos() -> dict:
    """List all available videos in batch_input directory"""
    if not batch_input_dir.exists():
        return {"videos": []}
    
    video_files = []
    for file in batch_input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.append(file.name)
    
    return {"videos": sorted(video_files)}


@app.get("/videos/{video_name}")
def get_video(video_name: str):
    """Serve a video file from batch_input directory"""
    # Prevent directory traversal
    if ".." in video_name or video_name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid video name")
    
    video_path = batch_input_dir / video_name
    
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    return FileResponse(
        video_path,
        media_type=f"video/{video_path.suffix.lower().strip('.')}"
    )
