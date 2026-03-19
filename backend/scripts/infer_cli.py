from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.inference import InferenceService


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _read_optional_text(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if file_path.exists() and file_path.is_file():
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _collect_batch_videos(batch_dir: str) -> list[Path]:
    root = Path(batch_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    videos = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 CLI inference for startup pitch videos.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", type=str, default="", help="Single video path for inference.")
    mode.add_argument("--batch-dir", type=str, default="", help="Directory containing multiple videos.")

    parser.add_argument("--duration-sec", type=int, default=60, help="Video duration used for chunk timeline.")
    parser.add_argument("--language-hint", type=str, default="en-ta", help="Language hint passed to transcriber.")
    parser.add_argument("--title", type=str, default="", help="Optional title override for single-video mode.")
    parser.add_argument("--transcript-file", type=str, default="", help="Optional transcript .txt file for single mode.")
    parser.add_argument("--output", type=str, default="", help="Optional output JSON file path.")
    parser.add_argument(
        "--batch-output-dir",
        type=str,
        default="",
        help="Optional output directory for per-video JSON files in batch mode.",
    )
    return parser


def _run_single(args: argparse.Namespace, service: InferenceService) -> dict:
    transcript_text = _read_optional_text(args.transcript_file)
    title = args.title.strip() or Path(args.video).stem
    response = service.infer_video(
        video_path=args.video,
        duration_sec=args.duration_sec,
        transcript_text=transcript_text,
        title=title,
        language_hint=args.language_hint,
    )
    return response.model_dump()


def _run_batch(args: argparse.Namespace, service: InferenceService) -> dict:
    videos = _collect_batch_videos(args.batch_dir)
    batch_results: list[dict] = []

    output_dir = Path(args.batch_output_dir) if args.batch_output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        response = service.infer_video(
            video_path=str(video_path),
            duration_sec=args.duration_sec,
            transcript_text="",
            title=video_path.stem,
            language_hint=args.language_hint,
        )
        payload = response.model_dump()
        batch_results.append(payload)

        if output_dir is not None:
            out_path = output_dir / f"{video_path.stem}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "mode": "batch",
        "count": len(batch_results),
        "results": batch_results,
    }


def main() -> None:
    args = build_parser().parse_args()
    service = InferenceService()

    if args.video:
        output_payload = _run_single(args, service)
    else:
        output_payload = _run_batch(args, service)

    result_json = json.dumps(output_payload, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")

    print(result_json)


if __name__ == "__main__":
    main()
