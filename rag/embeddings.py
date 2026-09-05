"""임베딩 모델. 인덱싱과 검색에 반드시 같은 모델을 사용한다 (config 단일 소스)."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from rag.config import settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )
