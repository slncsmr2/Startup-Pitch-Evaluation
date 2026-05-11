import logging
import shutil
from pathlib import Path
import os
import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.services.inference import InferenceService
from app.schemas import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationResponse,
    PitchInput,
    PitchVideoInput,
)

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
inference_service = InferenceService(window_seconds=settings.chunk_window_seconds)
static_dir = Path(__file__).resolve().parent / "static"
batch_input_dir = Path(__file__).resolve().parent.parent / "outputs" / "batch_input"
upload_dir = Path(__file__).resolve().parent.parent / "outputs" / "uploads"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Log startup mode
logger.info(
    f"App started | {settings.app_name} v{settings.app_version} | "
    f"use_heuristic_pipeline={settings.use_heuristic_pipeline} | "
    f"use_local_transcriber={settings.use_local_transcriber}"
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "scoring_mode": "heuristic" if settings.use_heuristic_pipeline else "neural-network",
    }


@app.get("/scoring-mode")
def scoring_mode() -> dict:
    return {
        "scoring_mode": "heuristic" if settings.use_heuristic_pipeline else "neural-network"
    }


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(static_dir / "index.html")


def _safe_video_name(original_name: str) -> str:
    suffix = Path(original_name).suffix or ".mp4"
    stem = Path(original_name).stem or "uploaded_pitch"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"


def _detect_duration_sec(video_path: Path, default_duration: int = 60) -> int:
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return default_duration
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        capture.release()
        if fps > 0 and frame_count > 0:
            return max(5, int(round(frame_count / fps)))
    except Exception as exc:
        logger.warning("Duration detection failed for %s: %s", video_path, exc)
    return default_duration


@app.post("/evaluate")
def evaluate_pitch(payload: PitchInput) -> EvaluationResponse:
    return inference_service.evaluate_payload(payload)


@app.post(
    "/evaluate/upload",
    responses={400: {"description": "Bad Request: video file is required"}},
)
async def evaluate_pitch_upload(
    video: Annotated[UploadFile, File(...)],
    title: Annotated[str, Form()] = "",
    transcript: Annotated[str, Form()] = "",
    language_hint: Annotated[str, Form()] = "en",
    slide_text: Annotated[str, Form()] = "",
    founder_name: Annotated[str, Form()] = "",
    startup_name: Annotated[str, Form()] = "",
    sector: Annotated[str, Form()] = "",
    stage: Annotated[str, Form()] = "",
) -> EvaluationResponse:
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_video_name(video.filename or "pitch.mp4")
    saved_path = upload_dir / safe_name

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    resolved_title = title.strip() or Path(video.filename or safe_name).stem
    slide_points = [line.strip() for line in slide_text.splitlines() if line.strip()]
    duration_sec = _detect_duration_sec(saved_path)

    payload = PitchInput(
        title=resolved_title,
        transcript=transcript.strip(),
        language_hint=language_hint.strip() or "en",
        presenter_profile={
            "founder_name": founder_name.strip(),
            "startup_name": startup_name.strip(),
            "sector": sector.strip(),
            "stage": stage.strip(),
        },
        slide_text=slide_points,
        video=PitchVideoInput(
            file_name=str(saved_path),
            file_format=safe_name.split(".")[-1] if "." in safe_name else "mp4",
            duration_sec=duration_sec,
            transcript_text=transcript.strip(),
        ),
        slides=[],
        user_details={
            "founder_name": founder_name.strip(),
            "startup_name": startup_name.strip(),
            "sector": sector.strip(),
            "stage": stage.strip(),
        },
    )

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
