from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "Startup Pitch Evaluation API"
    app_version: str = "0.1.0"
    chunk_window_seconds: int = 5
    
    # Phase 0: Migration control flags
    use_heuristic_pipeline: bool = True
    use_local_transcriber: bool = True
    enable_visual_extraction: bool = True
    enable_audio_extraction: bool = True
    transcriber_min_audio_quality: float = 0.35
    transcriber_backend: str = "auto"
    openai_api_key: str = ""
    openai_transcriber_model: str = "whisper-1"
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    media_lookup_dir: str = "outputs/batch_input"

    # Neural Network Config (Phase 7)
    nn_checkpoint_path: str = "models/checkpoints/nn_model.pt"
    nn_text_encoder: str = "all-MiniLM-L6-v2"
    nn_visual_backbone: str = "mobilenet_v3_small"
    nn_audio_features: str = "mfcc"
    nn_device: str = "auto"

    model_config = SettingsConfigDict(
        env_file=(str(_PROJECT_ROOT / ".env"), str(_BACKEND_ROOT / ".env")),
        env_prefix="SPE_",
    )


settings = Settings()
