from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from training.dataset_loader import FEATURE_DIM, METRIC_COUNT, TrainingSample, load_split_samples
from training.losses import mean_squared_error, mse_gradient
from training.metrics import mae, rmse, spearman_rank_correlation


def _clamp_10(value: float) -> float:
    return max(0.0, min(10.0, value))


def _coerce_value(raw: str) -> bool | int | float | str:
    value = raw.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_training_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    parsed: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        if ":" not in row:
            continue
        key, value = row.split(":", 1)
        parsed[key.strip()] = _coerce_value(value)

    defaults = {
        "epochs": 8,
        "learning_rate": 0.03,
        "train_samples": 120,
        "val_samples": 48,
        "test_samples": 48,
        "seed": 7,
        "checkpoint_dir": "models/checkpoints",
        "checkpoint_name": "phase5_checkpoint.json",
    }
    defaults.update(parsed)
    return defaults


@dataclass
class LinearMultiOutputModel:
    weights: list[list[float]]
    biases: list[float]

    @classmethod
    def initialize(cls, seed: int) -> "LinearMultiOutputModel":
        import random

        rng = random.Random(seed)
        weights = [[rng.uniform(-0.05, 0.05) for _ in range(FEATURE_DIM)] for _ in range(METRIC_COUNT)]
        biases = [0.0 for _ in range(METRIC_COUNT)]
        return cls(weights=weights, biases=biases)

    def predict_metrics(self, features: list[float]) -> list[float]:
        metrics: list[float] = []
        for row, bias in zip(self.weights, self.biases):
            raw = sum(w * x for w, x in zip(row, features)) + bias
            metrics.append(_clamp_10(5.0 + raw))
        return metrics

    def predict_overall(self, features: list[float]) -> float:
        metric_scores = self.predict_metrics(features)
        return sum(metric_scores) / len(metric_scores)


def _train_one_epoch(model: LinearMultiOutputModel, samples: list[TrainingSample], learning_rate: float) -> float:
    losses: list[float] = []
    for sample in samples:
        prediction = model.predict_metrics(sample.features)
        losses.append(mean_squared_error(prediction, sample.metric_targets))

        for out_idx in range(METRIC_COUNT):
            grad = mse_gradient(prediction[out_idx], sample.metric_targets[out_idx])
            for feat_idx in range(FEATURE_DIM):
                model.weights[out_idx][feat_idx] -= learning_rate * grad * sample.features[feat_idx]
            model.biases[out_idx] -= learning_rate * grad

    return sum(losses) / max(1, len(losses))


def _evaluate_regression(
    samples: list[TrainingSample],
    predictor: Callable[[list[float]], float],
) -> dict:
    preds = [predictor(sample.features) for sample in samples]
    targets = [sample.overall_target for sample in samples]
    return {
        "mae": round(mae(preds, targets), 4),
        "rmse": round(rmse(preds, targets), 4),
        "spearman": round(spearman_rank_correlation(preds, targets), 4),
    }


def _checkpoint_path(config: dict) -> Path:
    backend_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = backend_root / str(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / str(config["checkpoint_name"])


def save_checkpoint(model: LinearMultiOutputModel, config: dict, history: list[dict]) -> Path:
    path = _checkpoint_path(config)
    payload = {
        "weights": model.weights,
        "biases": model.biases,
        "config": config,
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_checkpoint(checkpoint_path: str) -> LinearMultiOutputModel:
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    return LinearMultiOutputModel(weights=payload["weights"], biases=payload["biases"])


def train_from_config(config_path: str) -> dict:
    config = load_training_config(config_path)

    train_samples = load_split_samples(
        "train",
        synthetic_count=int(config["train_samples"]),
        seed=int(config["seed"]),
    )
    val_samples = load_split_samples(
        "val",
        synthetic_count=int(config["val_samples"]),
        seed=int(config["seed"]),
    )

    model = LinearMultiOutputModel.initialize(seed=int(config["seed"]))
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = _train_one_epoch(model, train_samples, float(config["learning_rate"]))
        val_metrics = _evaluate_regression(val_samples, model.predict_overall)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_spearman": val_metrics["spearman"],
            }
        )

    checkpoint = save_checkpoint(model, config, history)

    return {
        "checkpoint_path": str(checkpoint),
        "epochs": int(config["epochs"]),
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_val_mae": history[-1]["val_mae"] if history else 0.0,
        "final_val_rmse": history[-1]["val_rmse"] if history else 0.0,
        "final_val_spearman": history[-1]["val_spearman"] if history else 0.0,
    }


def evaluate_from_config(config_path: str, checkpoint_path: str | None = None) -> dict:
    config = load_training_config(config_path)
    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        default_path = _checkpoint_path(config)
        model = load_checkpoint(str(default_path))

    test_samples = load_split_samples(
        "test",
        synthetic_count=int(config["test_samples"]),
        seed=int(config["seed"]),
    )
    return _evaluate_regression(test_samples, model.predict_overall)
