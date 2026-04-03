from pathlib import Path

import pytest

from models.scoring_head import ScoringHead


def test_resolve_checkpoint_path_supports_backend_relative_path() -> None:
    resolved = ScoringHead._resolve_checkpoint_path("models/checkpoints/phase6_gpu_nn_model.pt")

    assert resolved.exists()
    assert resolved.name == "phase6_gpu_nn_model.pt"


def test_calibrated_aggregate_stays_within_10_point_scale() -> None:
    aggregate = ScoringHead._calibrated_aggregate(text_avg=6.0, av_avg=4.0)

    assert aggregate == pytest.approx(5.2)
    assert 0.0 <= aggregate <= 10.0


def test_calibrated_aggregate_clamps_to_upper_bound() -> None:
    aggregate = ScoringHead._calibrated_aggregate(text_avg=20.0, av_avg=20.0)

    assert aggregate == 10.0
