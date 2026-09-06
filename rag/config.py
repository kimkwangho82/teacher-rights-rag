"""환경 변수 기반 설정. 모든 민감 정보는 .env 에서 읽는다 (하드코딩 금지)."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # --- LLM / Embedding ---
    openai_api_key: str = ""
    # OpenAI 호환 엔드포인트로 전환할 때 사용 (OpenAI 호환 프록시/타 제공자)
    openai_base_url: str | None = None
    chat_model: str = "gpt-4.1"  # baseline·개선 실험 리포트는 모두 이 모델로 실행
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    # 생성·심판 모두 같은 seed 를 사용해 재현성을 높인다 (OpenAI 는 best-effort)
    seed: int = 42

    # --- Eval (LLM-as-a-Judge) ---
    # 생성 모델과 분리된 심판 모델 (self-preference 회피)
    judge_model: str = "gpt-4.1"
    eval_dir: Path = BASE_DIR / "eval"
    reports_dir: Path = BASE_DIR / "eval" / "reports"
    judge_cache_dir: Path = BASE_DIR / "eval" / "cache"

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
    top_k: int = 5
    # 최고 관련도(cosine relevance, 0~1)가 이 값 미만이면 LLM 호출 없이 '정보 불충분' 반환
    similarity_threshold: float = 0.3

    # --- 개선 실험 스위치 (Part C) ---
    # dense: 벡터 검색만 / hybrid: BM25(문자 bigram) + dense 를 RRF 로 결합
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    hybrid_fetch_k: int = 20  # hybrid 시 각 검색기에서 가져올 후보 수
    # bigram: 문자 2-gram / kiwi: Kiwi 형태소 분석(명사·동사 어간·숫자 등 내용어만)
    bm25_tokenizer: Literal["bigram", "kiwi"] = "bigram"
    rrf_k: int = 60  # RRF 상수
    # baseline: 문맥 → 바로 답변 / evidence_first: 원문 인용 추출 후 인용에 근거해 답변
    prompt_mode: Literal["baseline", "evidence_first"] = "baseline"


settings = Settings()
