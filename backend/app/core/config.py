from pydantic_settings import BaseSettings, SettingsConfigDict


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
    openai_api_key: str = ""
    openai_transcriber_model: str = "whisper-1"
    media_lookup_dir: str = "outputs/batch_input"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SPE_")


settings = Settings()
