from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.inference import InferenceService


def _run_profile(profile: str, runs: int, duration_sec: int) -> dict:
    service = InferenceService()
    timings_ms: list[float] = []

    for idx in range(runs):
        start = time.perf_counter()
        service.infer_video(
            video_path=f"benchmark_{profile}_{idx}.mp4",
            duration_sec=duration_sec,
            transcript_text="We benchmark local inference latency with deterministic synthetic input.",
            title=f"benchmark-{profile}-{idx}",
            language_hint="en-ta",
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        timings_ms.append(round(elapsed, 3))

    return {
        "profile": profile,
        "runs": runs,
        "avg_ms": round(statistics.mean(timings_ms), 3),
        "p95_ms": round(sorted(timings_ms)[max(0, int(0.95 * len(timings_ms)) - 1)], 3),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "samples_ms": timings_ms,
        "notes": (
            "GPU profile currently uses the same deterministic local pipeline path; "
            "replace with actual GPU-backed model execution in later optimization phase."
            if profile == "gpu"
            else "CPU baseline for current local inference stack."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU runtime baselines for inference pipeline.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per profile.")
    parser.add_argument("--duration-sec", type=int, default=60, help="Synthetic video duration per run.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmark_runtime.json",
        help="Benchmark JSON output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cpu = _run_profile(profile="cpu", runs=max(1, args.runs), duration_sec=max(5, args.duration_sec))
    gpu = _run_profile(profile="gpu", runs=max(1, args.runs), duration_sec=max(5, args.duration_sec))

    benchmark_payload = {
        "benchmark": "phase7_inference_runtime",
        "profiles": [cpu, gpu],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    print(json.dumps(benchmark_payload, indent=2))


if __name__ == "__main__":
    main()
