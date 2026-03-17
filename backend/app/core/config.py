from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Startup Pitch Evaluation API"
    app_version: str = "0.1.0"
    chunk_window_seconds: int = 5
    
    # Phase 0: Migration control flags
    use_heuristic_pipeline: bool = True
    use_local_transcriber: bool = False
    local_transcriber_backend: str = "faster-whisper"
    local_transcriber_model_path: str = ""
    transcriber_min_audio_quality: float = 0.35

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SPE_")


settings = Settings()
