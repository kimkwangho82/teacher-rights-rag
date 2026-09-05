"""환경 변수 기반 설정. 모든 민감 정보는 .env 에서 읽는다 (하드코딩 금지)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # --- LLM / Embedding ---
    openai_api_key: str = ""
    # OpenAI 호환 엔드포인트로 전환할 때 사용 (OpenAI 호환 프록시/타 제공자)
    openai_base_url: str | None = None
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # --- 경로 ---
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    vectorstore_dir: Path = DATA_DIR / "vectorstore"
    manifest_path: Path = DATA_DIR / "raw" / "manifest.json"
    collection_name: str = "teacher-rights"

    # --- Chunking / Retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 이보다 짧은 청크(페이지 머리글/쪽번호 등)는 인덱스에서 제외
    min_chunk_chars: int = 50
    top_k: int = 4
    # 최고 관련도(cosine relevance, 0~1)가 이 값 미만이면 LLM 호출 없이 '정보 불충분' 반환
    similarity_threshold: float = 0.3


settings = Settings()
