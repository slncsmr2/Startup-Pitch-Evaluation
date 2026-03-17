from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Startup Pitch Evaluation API"
    app_version: str = "0.1.0"
    chunk_window_seconds: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SPE_")


settings = Settings()
