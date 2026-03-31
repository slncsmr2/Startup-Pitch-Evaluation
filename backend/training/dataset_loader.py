from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import hashlib


METRIC_COUNT = 10
TEXT_DIM = 384
VISUAL_DIM = 256
AUDIO_DIM = 128


@dataclass
class TrainingSample:
    text_features: list[float]
    visual_features: list[float]
    audio_features: list[float]
    metric_targets: list[float]
    overall_target: float
    investment_band: int


def _clamp_10(value: float) -> float:
    return max(0.0, min(10.0, value))


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_file(split: str) -> Path:
    return _backend_root() / "datasets" / "splits" / f"{split}.jsonl"


def _hash_to_vector(value: str, dim: int) -> list[float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    values = [b / 255.0 for b in digest]
    if dim <= len(values):
        return values[:dim]
    repeats = (dim // len(values)) + 1
    return (values * repeats)[:dim]


def _coerce_band(raw: object, overall: float) -> int:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "high-potential":
            return 2
        if normalized == "watchlist":
            return 1
        if normalized == "early-risk":
            return 0
    if isinstance(raw, (int, float)):
        value = int(raw)
        if value in {0, 1, 2}:
            return value

    # Default rubric-based bucket if label is missing.
    if overall >= 7.5:
        return 2
    if overall >= 5.0:
        return 1
    return 0


def _normalize_scores(raw: list[object]) -> list[float]:
    scores = [float(v) for v in raw][:METRIC_COUNT]
    scores = scores + [0.0] * max(0, METRIC_COUNT - len(scores))
    return [round(_clamp_10(v), 4) for v in scores]


def _sample_from_option_a_row(item: dict) -> TrainingSample:
    transcript = str(item.get("transcript", "")).strip()
    slide_text = str(item.get("slide_text", "")).strip()
    video_id = str(item.get("video_id", "unknown"))
    metrics = _normalize_scores(item.get("scores", []))
    overall = float(item.get("overall_target", sum(metrics) / max(1, len(metrics))))
    overall = _clamp_10(overall)
    band = _coerce_band(item.get("investment_band", None), overall)

    text_features = _hash_to_vector(f"text::{video_id}::{transcript}", TEXT_DIM)
    visual_features = _hash_to_vector(f"visual::{video_id}::{slide_text}", VISUAL_DIM)
    audio_features = _hash_to_vector(f"audio::{video_id}::{transcript[:300]}", AUDIO_DIM)

    return TrainingSample(
        text_features=text_features,
        visual_features=visual_features,
        audio_features=audio_features,
        metric_targets=metrics,
        overall_target=round(overall, 4),
        investment_band=band,
    )


def _sample_from_legacy_row(item: dict) -> TrainingSample:
    # Backward compatibility for older synthetic/feature-only rows.
    legacy_features = [float(v) for v in item.get("features", [])][:16]
    legacy_features = legacy_features + [0.0] * max(0, 16 - len(legacy_features))
    metrics = _normalize_scores(item.get("metric_targets", []))
    overall = float(item.get("overall_target", sum(metrics) / max(1, len(metrics))))
    overall = _clamp_10(overall)
    band = _coerce_band(item.get("investment_band", None), overall)

    serialized = "|".join(f"{v:.6f}" for v in legacy_features)
    text_features = _hash_to_vector(f"legacy-text::{serialized}", TEXT_DIM)
    visual_features = _hash_to_vector(f"legacy-visual::{serialized}", VISUAL_DIM)
    audio_features = _hash_to_vector(f"legacy-audio::{serialized}", AUDIO_DIM)

    return TrainingSample(
        text_features=text_features,
        visual_features=visual_features,
        audio_features=audio_features,
        metric_targets=metrics,
        overall_target=round(overall, 4),
        investment_band=band,
    )


def _load_from_jsonl(path: Path) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "scores" in item and "transcript" in item:
            samples.append(_sample_from_option_a_row(item))
        else:
            samples.append(_sample_from_legacy_row(item))
    return samples


def _synthetic_samples(count: int, seed: int) -> list[TrainingSample]:
    rng = random.Random(seed)
    hidden_w = [[rng.uniform(-0.8, 0.8) for _ in range(TEXT_DIM)] for _ in range(METRIC_COUNT)]
    hidden_b = [rng.uniform(-0.7, 0.7) for _ in range(METRIC_COUNT)]

    samples: list[TrainingSample] = []
    for _ in range(count):
        text_features = [rng.uniform(0.0, 1.0) for _ in range(TEXT_DIM)]
        visual_features = [rng.uniform(0.0, 1.0) for _ in range(VISUAL_DIM)]
        audio_features = [rng.uniform(0.0, 1.0) for _ in range(AUDIO_DIM)]
        metric_targets: list[float] = []
        for row, bias in zip(hidden_w, hidden_b):
            raw = sum(w * x for w, x in zip(row, text_features)) + bias + rng.uniform(-0.15, 0.15)
            scaled = 5.0 + raw * 2.2
            metric_targets.append(round(_clamp_10(scaled), 4))

        overall_target = round(sum(metric_targets) / len(metric_targets), 4)
        investment_band = _coerce_band(None, overall_target)
        samples.append(
            TrainingSample(
                text_features=text_features,
                visual_features=visual_features,
                audio_features=audio_features,
                metric_targets=metric_targets,
                overall_target=overall_target,
                investment_band=investment_band,
            )
        )
    return samples


def load_split_samples(split: str, synthetic_count: int = 64, seed: int = 7) -> list[TrainingSample]:
    path = _split_file(split)
    if path.exists():
        loaded = _load_from_jsonl(path)
        if loaded:
            return loaded

    split_offset = {"train": 0, "val": 1000, "test": 2000}.get(split, 3000)
    return _synthetic_samples(count=synthetic_count, seed=seed + split_offset)
