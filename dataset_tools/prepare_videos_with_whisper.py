from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return normalized.strip("_").lower() or "video"


def _collect_videos(input_dir: Path) -> list[Path]:
    videos = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate

    # Allow execution from either project root or dataset_tools/.
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    project_path = (PROJECT_ROOT / candidate).resolve()
    return project_path


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


def _candidate_cuda_bin_dirs(extra_cuda_bin: str = "") -> list[Path]:
    candidates: list[Path] = []
    if extra_cuda_bin:
        candidates.append(Path(extra_cuda_bin))

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        for version_dir in sorted(cuda_root.glob("v*"), reverse=True):
            candidates.append(version_dir / "bin")

    return candidates


def _configure_windows_cuda_dll_loading(device: str, extra_cuda_bin: str = "", debug_runtime: bool = False) -> None:
    if os.name != "nt" or device != "cuda":
        return

    configured_any = False
    for path in _candidate_cuda_bin_dirs(extra_cuda_bin=extra_cuda_bin):
        configured_any = _register_cuda_dir(path=path, debug_runtime=debug_runtime) or configured_any

    if debug_runtime and not configured_any:
        print("warning=no_valid_cuda_bin_dir_found")


def _register_cuda_dir(path: Path, debug_runtime: bool) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    path_str = str(path)
    _prepend_path_if_missing(path_str)

    try:
        os.add_dll_directory(path_str)
        if debug_runtime:
            print(f"added_cuda_dll_dir={path_str}")
    except Exception as exc:
        if debug_runtime:
            print(f"failed_add_cuda_dll_dir={path_str} | reason={exc}")

    return True


def _prepend_path_if_missing(path_str: str) -> None:
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if path_str not in path_entries:
        os.environ["PATH"] = path_str + os.pathsep + current_path


def _find_cublas_dll() -> str:
    if os.name != "nt":
        return "non-windows"

    for path in os.environ.get("PATH", "").split(os.pathsep):
        if not path:
            continue
        dll_path = Path(path) / "cublas64_12.dll"
        if dll_path.exists():
            return str(dll_path)
    return "missing"


def _gpu_diagnostics() -> str:
    try:
        import torch  # type: ignore

        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if available else 0
        names = [torch.cuda.get_device_name(i) for i in range(device_count)] if available else []
        cublas = _find_cublas_dll()
        return (
            f"torch_cuda_available={available} | gpu_count={device_count} | "
            f"gpu_names={names} | cublas64_12={cublas}"
        )
    except Exception as exc:
        return f"torch_cuda_probe_failed={exc}"


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
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Fail instead of falling back to CPU when CUDA transcription is unavailable.",
    )
    parser.add_argument(
        "--debug-runtime",
        action="store_true",
        help="Print detailed runtime diagnostics for CUDA/faster-whisper setup.",
    )
    parser.add_argument(
        "--cuda-bin",
        type=str,
        default="",
        help="Optional explicit CUDA bin directory (Windows), e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = _resolve_path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory not found: {input_dir}")

    videos = _collect_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_dir}")

    active_device = str(args.device).strip().lower() or "cpu"
    active_compute_type = str(args.compute_type).strip() or "int8"

    _configure_windows_cuda_dll_loading(
        device=active_device,
        extra_cuda_bin=str(args.cuda_bin),
        debug_runtime=bool(args.debug_runtime),
    )

    model = _build_whisper_model(
        model_size=args.model_size,
        device=active_device,
        compute_type=active_compute_type,
    )

    if args.debug_runtime:
        print(f"Requested whisper runtime: device={active_device}, compute_type={active_compute_type}")
        print(_gpu_diagnostics())

    pitch_rows, labeling_rows, active_device, active_compute_type = _process_videos(
        args=args,
        videos=videos,
        model=model,
        active_device=active_device,
        active_compute_type=active_compute_type,
    )

    output_pitch = _resolve_path(args.output_pitch_inputs)
    output_labeling = _resolve_path(args.output_labeling)

    _write_jsonl(output_pitch, pitch_rows)
    _write_jsonl(output_labeling, labeling_rows)

    print(f"Done. PitchInput JSONL: {output_pitch}")
    print(f"Done. Labeling JSONL: {output_labeling}")
    print(f"Whisper runtime used: device={active_device}, compute_type={active_compute_type}")
    print("No external API keys were used.")


def _process_videos(args, videos: list[Path], model, active_device: str, active_compute_type: str):
    pitch_rows: list[dict[str, Any]] = []
    labeling_rows: list[dict[str, Any]] = []

    for idx, video_path in enumerate(videos, start=1):
        title = video_path.stem
        video_id = f"{idx:03d}_{_safe_id(title)}"
        transcript, model, active_device, active_compute_type = _transcribe_with_fallback(
            args=args,
            model=model,
            video_path=video_path,
            active_device=active_device,
            active_compute_type=active_compute_type,
        )
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

    return pitch_rows, labeling_rows, active_device, active_compute_type


def _transcribe_with_fallback(args, model, video_path: Path, active_device: str, active_compute_type: str):
    try:
        transcript = _transcribe_video(model=model, video_path=video_path, language_hint=args.language_hint)
        return transcript, model, active_device, active_compute_type
    except RuntimeError as exc:
        can_fallback = active_device == "cuda" and _is_cuda_runtime_error(exc) and not args.strict_gpu
        if not can_fallback:
            if active_device == "cuda" and args.strict_gpu and _is_cuda_runtime_error(exc):
                raise RuntimeError(f"CUDA transcription failed in strict GPU mode: {exc}") from exc
            raise

        print("CUDA runtime unavailable. Falling back to CPU int8 for whisper transcription.")
        if args.debug_runtime:
            print(f"Fallback reason: {exc}")

        active_device = "cpu"
        active_compute_type = "int8"
        model = _build_whisper_model(model_size=args.model_size, device=active_device, compute_type=active_compute_type)
        transcript = _transcribe_video(model=model, video_path=video_path, language_hint=args.language_hint)
        return transcript, model, active_device, active_compute_type


if __name__ == "__main__":
    main()
