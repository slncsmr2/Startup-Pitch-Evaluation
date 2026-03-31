from __future__ import annotations

import json
import math
from pathlib import Path
from copy import deepcopy

from training.dataset_loader import AUDIO_DIM, METRIC_COUNT, TEXT_DIM, VISUAL_DIM, TrainingSample, load_split_samples
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
        "data_strategy": "option_a",
        "device": "auto",
        "epochs": 30,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "warm_restart_t0": 10,
        "warm_restart_t_mult": 2,
        "early_stopping_patience": 10,
        "classification_loss_weight": 1.0,
        "spearman_loss_weight": 0.05,
        "mixed_precision": True,
        "train_samples": 120,
        "val_samples": 48,
        "test_samples": 48,
        "seed": 7,
        "checkpoint_dir": "models/checkpoints",
        "checkpoint_name": "phase6_nn_model.pt",
    }
    defaults.update(parsed)
    return defaults


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as functional  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise ImportError("Phase 6 trainer requires torch to be installed") from exc
    return torch, nn, functional, DataLoader, TensorDataset


def _spearman_proxy_loss(functional, overall_pred, overall_target):
    n = int(overall_pred.shape[0])
    if n < 2:
        return overall_pred.new_tensor(0.0)

    pred_diff = overall_pred.unsqueeze(1) - overall_pred.unsqueeze(0)
    target_diff = overall_target.unsqueeze(1) - overall_target.unsqueeze(0)

    mask = target_diff != 0
    signs = target_diff.sign()
    losses = functional.softplus(-(signs * pred_diff))

    if mask.any():
        return losses[mask].mean()
    return overall_pred.new_tensor(0.0)


def _select_device(torch, requested: str):
    option = requested.strip().lower()
    if option in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if option == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tensorize_samples(torch, samples: list[TrainingSample]):
    text = torch.tensor([s.text_features for s in samples], dtype=torch.float32)
    visual = torch.tensor([s.visual_features for s in samples], dtype=torch.float32)
    audio = torch.tensor([s.audio_features for s in samples], dtype=torch.float32)
    metrics = torch.tensor([s.metric_targets for s in samples], dtype=torch.float32)
    overall = torch.tensor([s.overall_target for s in samples], dtype=torch.float32)
    bands = torch.tensor([s.investment_band for s in samples], dtype=torch.long)
    return text, visual, audio, metrics, overall, bands


def _compute_eval_metrics(overall_preds: list[float], overall_targets: list[float], band_preds: list[int], band_targets: list[int]) -> dict:
    correct = sum(1 for p, t in zip(band_preds, band_targets) if p == t)
    return {
        "mae": round(mae(overall_preds, overall_targets), 4),
        "rmse": round(rmse(overall_preds, overall_targets), 4),
        "spearman": round(spearman_rank_correlation(overall_preds, overall_targets), 4),
        "band_accuracy": round(correct / max(1, len(band_targets)), 4),
    }


def _build_model(nn, common_dim: int = 256, dropout: float = 0.3):
    class Phase6MultimodalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = nn.Sequential(nn.Linear(TEXT_DIM, common_dim), nn.ReLU())
            self.visual_encoder = nn.Sequential(nn.Linear(VISUAL_DIM, common_dim), nn.ReLU())
            self.audio_encoder = nn.Sequential(nn.Linear(AUDIO_DIM, common_dim), nn.ReLU())

            self.attention_score = nn.Linear(common_dim, 1)

            self.scoring_backbone = nn.Sequential(
                nn.Linear(common_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.metric_head = nn.Linear(64, METRIC_COUNT)
            self.overall_head = nn.Linear(METRIC_COUNT, 1)
            self.band_head = nn.Linear(common_dim, 3)

        def forward(self, text_x, visual_x, audio_x):
            t = self.text_encoder(text_x)
            v = self.visual_encoder(visual_x)
            a = self.audio_encoder(audio_x)

            stacked = self._stack_modalities(t, v, a)
            scores = self.attention_score(stacked).squeeze(-1)
            attn = scores.softmax(dim=-1)
            fused = (attn.unsqueeze(-1) * stacked).sum(dim=1)

            hidden = self.scoring_backbone(fused)
            metric_scores = self.metric_head(hidden).sigmoid() * 10.0
            overall_score = self.overall_head(metric_scores).sigmoid().squeeze(-1) * 10.0
            band_logits = self.band_head(fused)
            return metric_scores, overall_score, band_logits

        @staticmethod
        def _stack_modalities(t, v, a):
            import torch  # local import keeps class self-contained

            return torch.stack([t, v, a], dim=1)

    return Phase6MultimodalModel()


def _run_epoch(
    *,
    torch,
    functional,
    model,
    loader,
    optimizer,
    scaler,
    device,
    train_mode: bool,
    cls_weight: float,
    rank_weight: float,
    use_autocast: bool,
) -> dict:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metric_loss_total = 0.0
    cls_loss_total = 0.0
    rank_loss_total = 0.0

    overall_preds: list[float] = []
    overall_targets: list[float] = []
    band_preds: list[int] = []
    band_targets: list[int] = []

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for text_x, visual_x, audio_x, metric_y, overall_y, band_y in loader:
            text_x = text_x.to(device)
            visual_x = visual_x.to(device)
            audio_x = audio_x.to(device)
            metric_y = metric_y.to(device)
            overall_y = overall_y.to(device)
            band_y = band_y.to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_autocast):
                metric_pred, overall_pred, band_logits = model(text_x, visual_x, audio_x)
                metric_loss = functional.mse_loss(metric_pred, metric_y)
                cls_loss = functional.cross_entropy(band_logits, band_y)
                rank_loss = _spearman_proxy_loss(functional, overall_pred, overall_y)
                loss = metric_loss + (cls_weight * cls_loss) + (rank_weight * rank_loss)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item())
            metric_loss_total += float(metric_loss.item())
            cls_loss_total += float(cls_loss.item())
            rank_loss_total += float(rank_loss.item())

            overall_preds.extend(overall_pred.detach().cpu().tolist())
            overall_targets.extend(overall_y.detach().cpu().tolist())
            band_preds.extend(band_logits.argmax(dim=1).detach().cpu().tolist())
            band_targets.extend(band_y.detach().cpu().tolist())

    batches = max(1, len(loader))
    return {
        "loss": total_loss / batches,
        "metric_loss": metric_loss_total / batches,
        "classification_loss": cls_loss_total / batches,
        "rank_loss": rank_loss_total / batches,
        "evaluation": _compute_eval_metrics(overall_preds, overall_targets, band_preds, band_targets),
    }


def _checkpoint_path(config: dict) -> Path:
    backend_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = backend_root / str(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / str(config["checkpoint_name"])


def save_checkpoint(model, config: dict, history: list[dict], best_epoch: int) -> Path:
    torch, _, _, _, _ = _import_torch()
    path = _checkpoint_path(config)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(checkpoint_path: str, device):
    torch, nn, _, _, _ = _import_torch()
    payload = torch.load(checkpoint_path, map_location=device)
    model = _build_model(nn)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def train_from_config(config_path: str) -> dict:
    torch, nn, functional, data_loader_cls, tensor_dataset_cls = _import_torch()
    config = load_training_config(config_path)

    if str(config.get("data_strategy", "option_a")).lower() != "option_a":
        raise ValueError("Phase 6 currently supports only data_strategy=option_a")

    torch.manual_seed(int(config["seed"]))
    device = _select_device(torch, str(config["device"]))

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

    train_tensors = _tensorize_samples(torch, train_samples)
    val_tensors = _tensorize_samples(torch, val_samples)
    train_ds = tensor_dataset_cls(*train_tensors)
    val_ds = tensor_dataset_cls(*val_tensors)
    train_loader = data_loader_cls(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = data_loader_cls(val_ds, batch_size=int(config["batch_size"]), shuffle=False)

    model = _build_model(nn)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, int(config["warm_restart_t0"])),
        T_mult=max(1, int(config["warm_restart_t_mult"])),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(config["mixed_precision"])))

    history: list[dict] = []
    best_epoch = 0
    best_mae = math.inf
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    cls_weight = float(config["classification_loss_weight"])
    rank_weight = float(config["spearman_loss_weight"])
    patience = max(1, int(config["early_stopping_patience"]))
    use_autocast = device.type == "cuda" and bool(config["mixed_precision"])

    for epoch in range(1, int(config["epochs"]) + 1):
        train_stats = _run_epoch(
            torch=torch,
            functional=functional,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_mode=True,
            cls_weight=cls_weight,
            rank_weight=rank_weight,
            use_autocast=use_autocast,
        )
        val_stats = _run_epoch(
            torch=torch,
            functional=functional,
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_mode=False,
            cls_weight=cls_weight,
            rank_weight=rank_weight,
            use_autocast=use_autocast,
        )

        scheduler.step(epoch - 1 + 1e-3)

        val_mae = float(val_stats["evaluation"]["mae"])
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "train_loss": round(float(train_stats["loss"]), 4),
                "train_metric_loss": round(float(train_stats["metric_loss"]), 4),
                "train_classification_loss": round(float(train_stats["classification_loss"]), 4),
                "train_rank_loss": round(float(train_stats["rank_loss"]), 4),
                "val_loss": round(float(val_stats["loss"]), 4),
                "val_mae": round(float(val_stats["evaluation"]["mae"]), 4),
                "val_rmse": round(float(val_stats["evaluation"]["rmse"]), 4),
                "val_spearman": round(float(val_stats["evaluation"]["spearman"]), 4),
                "val_band_accuracy": round(float(val_stats["evaluation"]["band_accuracy"]), 4),
            }
        )

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint = save_checkpoint(model, config, history, best_epoch=best_epoch)
    final_row = history[-1] if history else {}

    return {
        "checkpoint_path": str(checkpoint),
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_mae": round(best_mae, 4) if best_mae < math.inf else 0.0,
        "final_train_loss": final_row.get("train_loss", 0.0),
        "final_val_mae": final_row.get("val_mae", 0.0),
        "final_val_rmse": final_row.get("val_rmse", 0.0),
        "final_val_spearman": final_row.get("val_spearman", 0.0),
        "final_val_band_accuracy": final_row.get("val_band_accuracy", 0.0),
    }


def evaluate_from_config(config_path: str, checkpoint_path: str | None = None) -> dict:
    torch, _, functional, data_loader_cls, tensor_dataset_cls = _import_torch()
    config = load_training_config(config_path)
    device = _select_device(torch, str(config["device"]))

    chosen_checkpoint = checkpoint_path if checkpoint_path else str(_checkpoint_path(config))
    model, _ = load_checkpoint(chosen_checkpoint, device)

    test_samples = load_split_samples(
        "test",
        synthetic_count=int(config["test_samples"]),
        seed=int(config["seed"]),
    )

    test_ds = tensor_dataset_cls(*_tensorize_samples(torch, test_samples))
    test_loader = data_loader_cls(test_ds, batch_size=int(config["batch_size"]), shuffle=False)
    stats = _run_epoch(
        torch=torch,
        functional=functional,
        model=model,
        loader=test_loader,
        optimizer=None,
        scaler=torch.cuda.amp.GradScaler(enabled=False),
        device=device,
        train_mode=False,
        cls_weight=float(config["classification_loss_weight"]),
        rank_weight=float(config["spearman_loss_weight"]),
        use_autocast=False,
    )
    result = stats["evaluation"]
    result["loss"] = round(float(stats["loss"]), 4)
    return result
