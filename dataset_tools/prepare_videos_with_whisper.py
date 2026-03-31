from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return normalized.strip("_").lower() or "video"


def _collect_videos(input_dir: Path) -> list[Path]:
    videos = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _estimate_duration_seconds(video_path: Path, default_duration: int) -> int:
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return default_duration

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        capture.release()

        if fps <= 0:
            return default_duration

        duration = int(round(frame_count / fps))
        return max(5, duration)
    except Exception:
        return default_duration


def _build_whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not installed. Install dependencies with: pip install -r backend/requirements.txt"
        ) from exc

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    cuda_tokens = [
        "cublas",
        "cudnn",
        "cuda",
        "cannot be loaded",
        "not found",
    ]
    return any(token in message for token in cuda_tokens)


def _transcribe_video(model, video_path: Path, language_hint: str = "") -> str:
    language = language_hint.strip().lower() if language_hint.strip() in {"en", "ta"} else None
    segments, _info = model.transcribe(str(video_path), language=language)
    transcript = " ".join((segment.text or "").strip() for segment in segments).strip()
    return transcript


def _pitch_input_record(video_path: Path, title: str, duration_sec: int, transcript: str, language_hint: str) -> dict[str, Any]:
    return {
        "title": title,
        "transcript": "",
        "language_hint": language_hint or "en-ta",
        "presenter_profile": {},
        "slide_text": [],
        "video": {
            "file_name": str(video_path),
            "file_format": video_path.suffix.lower().replace(".", "") or "mp4",
            "duration_sec": duration_sec,
            "transcript_text": transcript,
        },
        "slides": [],
        "user_details": {
            "founder_name": "",
            "startup_name": title,
            "sector": "",
            "stage": "",
        },
    }


def _labeling_record(video_id: str, transcript: str, slide_text: str = "") -> dict[str, Any]:
    return {
        "video_id": video_id,
        "transcript": transcript,
        "slide_text": slide_text,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch transcribe videos locally with faster-whisper and export project-ready JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="backend/outputs/batch_input",
        help="Directory containing pitch videos.",
    )
    parser.add_argument(
        "--output-pitch-inputs",
        type=str,
        default="backend/outputs/prepared/pitch_inputs.jsonl",
        help="Output JSONL path for API/CLI-ready PitchInput rows.",
    )
    parser.add_argument(
        "--output-labeling",
        type=str,
        default="backend/datasets/splits/labeling_input.jsonl",
        help="Output JSONL path for Option A labeling rows (video_id, transcript, slide_text).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="faster-whisper model size (tiny, base, small, medium, large-v3, etc.).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Whisper device: cpu or cuda.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Whisper compute type (int8 for CPU, float16 for CUDA, etc.).",
    )
    parser.add_argument(
        "--language-hint",
        type=str,
        default="",
        help="Optional language hint: en or ta. Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--default-duration-sec",
        type=int,
        default=60,
        help="Fallback duration when metadata cannot be read.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory not found: {input_dir}")

    videos = _collect_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_dir}")

    active_device = str(args.device).strip().lower() or "cpu"
    active_compute_type = str(args.compute_type).strip() or "int8"

    model = _build_whisper_model(
        model_size=args.model_size,
        device=active_device,
        compute_type=active_compute_type,
    )

    pitch_rows: list[dict[str, Any]] = []
    labeling_rows: list[dict[str, Any]] = []

    for idx, video_path in enumerate(videos, start=1):
        title = video_path.stem
        video_id = f"{idx:03d}_{_safe_id(title)}"

        try:
            transcript = _transcribe_video(
                model=model,
                video_path=video_path,
                language_hint=args.language_hint,
            )
        except RuntimeError as exc:
            if active_device == "cuda" and _is_cuda_runtime_error(exc):
                print("CUDA runtime unavailable. Falling back to CPU int8 for whisper transcription.")
                active_device = "cpu"
                active_compute_type = "int8"
                model = _build_whisper_model(
                    model_size=args.model_size,
                    device=active_device,
                    compute_type=active_compute_type,
                )
                transcript = _transcribe_video(
                    model=model,
                    video_path=video_path,
                    language_hint=args.language_hint,
                )
            else:
                raise
        duration_sec = _estimate_duration_seconds(video_path, default_duration=max(5, int(args.default_duration_sec)))

        pitch_rows.append(
            _pitch_input_record(
                video_path=video_path,
                title=title,
                duration_sec=duration_sec,
                transcript=transcript,
                language_hint=args.language_hint,
            )
        )
        labeling_rows.append(_labeling_record(video_id=video_id, transcript=transcript))

        print(f"[{idx}/{len(videos)}] prepared: {video_path.name}")

    output_pitch = Path(args.output_pitch_inputs)
    output_labeling = Path(args.output_labeling)

    _write_jsonl(output_pitch, pitch_rows)
    _write_jsonl(output_labeling, labeling_rows)

    print(f"Done. PitchInput JSONL: {output_pitch}")
    print(f"Done. Labeling JSONL: {output_labeling}")
    print(f"Whisper runtime used: device={active_device}, compute_type={active_compute_type}")
    print("No external API keys were used.")


if __name__ == "__main__":
    main()
