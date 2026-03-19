from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random


FEATURE_DIM = 16
METRIC_COUNT = 10


@dataclass
class TrainingSample:
    features: list[float]
    metric_targets: list[float]
    overall_target: float


def _clamp_10(value: float) -> float:
    return max(0.0, min(10.0, value))


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_file(split: str) -> Path:
    return _backend_root() / "datasets" / "splits" / f"{split}.jsonl"


def _load_from_jsonl(path: Path) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        features = [float(v) for v in item.get("features", [])][:FEATURE_DIM]
        features = features + [0.0] * max(0, FEATURE_DIM - len(features))
        metrics = [float(v) for v in item.get("metric_targets", [])][:METRIC_COUNT]
        metrics = metrics + [0.0] * max(0, METRIC_COUNT - len(metrics))
        overall = float(item.get("overall_target", sum(metrics) / max(1, len(metrics))))
        samples.append(TrainingSample(features=features, metric_targets=metrics, overall_target=overall))
    return samples


def _synthetic_samples(count: int, seed: int) -> list[TrainingSample]:
    rng = random.Random(seed)
    hidden_w = [[rng.uniform(-0.8, 0.8) for _ in range(FEATURE_DIM)] for _ in range(METRIC_COUNT)]
    hidden_b = [rng.uniform(-0.7, 0.7) for _ in range(METRIC_COUNT)]

    samples: list[TrainingSample] = []
    for _ in range(count):
        features = [rng.uniform(0.0, 1.0) for _ in range(FEATURE_DIM)]
        metric_targets: list[float] = []
        for row, bias in zip(hidden_w, hidden_b):
            raw = sum(w * x for w, x in zip(row, features)) + bias + rng.uniform(-0.15, 0.15)
            scaled = 5.0 + raw * 2.2
            metric_targets.append(round(_clamp_10(scaled), 4))

        overall_target = round(sum(metric_targets) / len(metric_targets), 4)
        samples.append(
            TrainingSample(
                features=features,
                metric_targets=metric_targets,
                overall_target=overall_target,
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
