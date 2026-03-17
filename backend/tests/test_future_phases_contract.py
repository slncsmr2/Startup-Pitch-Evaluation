from __future__ import annotations

from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT / "app"
MODELS_ROOT = BACKEND_ROOT / "models"
TRAINING_ROOT = BACKEND_ROOT / "training"
SCRIPTS_ROOT = BACKEND_ROOT / "scripts"
TESTS_ROOT = BACKEND_ROOT / "tests"


def _require_paths_for_phase(phase_name: str, required: list[Path]) -> None:
    missing = [str(path.relative_to(BACKEND_ROOT)) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"{phase_name} not implemented yet. Missing: {', '.join(missing)}")


def test_phase4_trainable_model_stack_contract() -> None:
    required = [
        MODELS_ROOT / "text_encoder.py",
        MODELS_ROOT / "visual_encoder.py",
        MODELS_ROOT / "audio_encoder.py",
        MODELS_ROOT / "fusion_head.py",
        MODELS_ROOT / "scoring_head.py",
        APP_ROOT / "services" / "extractors.py",
        APP_ROOT / "services" / "fusion.py",
        APP_ROOT / "services" / "scoring.py",
    ]
    _require_paths_for_phase("Phase 4", required)

    for path in required:
        assert path.is_file(), f"Expected file not found: {path}"

    model_files = [
        MODELS_ROOT / "text_encoder.py",
        MODELS_ROOT / "visual_encoder.py",
        MODELS_ROOT / "audio_encoder.py",
        MODELS_ROOT / "fusion_head.py",
        MODELS_ROOT / "scoring_head.py",
    ]
    for model_file in model_files:
        source = model_file.read_text(encoding="utf-8").lower()
        assert "class " in source, f"Expected at least one class in {model_file.name}"


def test_phase5_training_system_contract() -> None:
    required = [
        TRAINING_ROOT / "dataset_loader.py",
        TRAINING_ROOT / "trainer.py",
        TRAINING_ROOT / "metrics.py",
        TRAINING_ROOT / "losses.py",
        SCRIPTS_ROOT / "train.py",
        SCRIPTS_ROOT / "evaluate.py",
        MODELS_ROOT / "config" / "training_cpu.yaml",
        MODELS_ROOT / "config" / "training_gpu.yaml",
        MODELS_ROOT / "checkpoints",
    ]
    _require_paths_for_phase("Phase 5", required)

    for path in required:
        assert path.exists(), f"Expected training artifact missing: {path}"

    for script in [SCRIPTS_ROOT / "train.py", SCRIPTS_ROOT / "evaluate.py"]:
        source = script.read_text(encoding="utf-8").lower()
        assert "main" in source, f"Expected entrypoint-like function in {script.name}"


def test_phase6_cli_inference_contract() -> None:
    required = [
        SCRIPTS_ROOT / "infer_cli.py",
        APP_ROOT / "services" / "inference.py",
    ]
    _require_paths_for_phase("Phase 6", required)

    infer_cli_source = (SCRIPTS_ROOT / "infer_cli.py").read_text(encoding="utf-8").lower()
    assert "batch" in infer_cli_source, "Expected batch mode support in infer_cli.py"
    assert "json" in infer_cli_source, "Expected JSON output support in infer_cli.py"

    inference_source = (APP_ROOT / "services" / "inference.py").read_text(encoding="utf-8").lower()
    assert "transcrib" in inference_source
    assert "video" in inference_source


def test_phase6b_fastapi_cli_parity_contract() -> None:
    required = [
        APP_ROOT / "main.py",
        APP_ROOT / "services" / "inference.py",
    ]
    _require_paths_for_phase("Phase 6b", required)

    main_source = (APP_ROOT / "main.py").read_text(encoding="utf-8").lower()
    assert "/evaluate" in main_source
    assert "inference" in main_source, "Expected FastAPI to call shared inference service"


def test_phase7_quality_and_hardening_contract() -> None:
    benchmark_candidates = list(SCRIPTS_ROOT.glob("*benchmark*.py"))
    if not benchmark_candidates:
        pytest.skip("Phase 7 not implemented yet. Missing benchmark script in backend/scripts.")

    for benchmark_script in benchmark_candidates:
        assert benchmark_script.is_file()

    suite_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in TESTS_ROOT.glob("test_*.py"))
    assert "silent" in suite_text, "Expected silent audio edge-case tests"
    assert "short video" in suite_text or "short_video" in suite_text, "Expected short video edge-case tests"
    assert "corrupt" in suite_text, "Expected corrupt file edge-case tests"
    assert "tamil" in suite_text and "english" in suite_text, "Expected mixed English-Tamil tests"
