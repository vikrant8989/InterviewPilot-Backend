from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    # Database
    database_url: str

    # Redis (queues) - optional
    redis_url: str | None = None

    # R2 (Cloudflare)
    r2_endpoint_url: str | None = None
    r2_account_id: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str | None = None

    # Internal worker -> API callback
    internal_api_url: str | None = None
    internal_secret: str | None = None

    # CORS
    cors_origins: str = "*"

    # Dev convenience (do NOT enable in production)
    auto_create_tables: bool = False

    # Transcription
    whisper_provider: str = "openai"  # "openai" or "local"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_transcription_model: str = "whisper-1"
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "openai/gpt-oss-120b")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.7))

    # TTS
    tts_provider: str = "gtts"  # "gtts" for free-first
    tts_lang: str = "en"
    tts_audio_expires_seconds: int = 3600

    # Google OAuth
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000


settings = Settings()

