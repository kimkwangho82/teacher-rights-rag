"""Chroma 벡터 스토어 생성/로드.

- cosine 공간을 사용해 relevance score 를 0~1 로 해석할 수 있게 한다 (abstain 임계값에 사용).
- 임베딩은 rag.embeddings.get_embeddings() 단일 소스 → 인덱싱/검색 모델 불일치 방지.
"""
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import settings
from rag.embeddings import get_embeddings

COLLECTION_METADATA = {"hnsw:space": "cosine"}
BATCH_SIZE = 100


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(settings.vectorstore_dir),
        embedding_function=get_embeddings(),
        collection_name=settings.collection_name,
        collection_metadata=COLLECTION_METADATA,
    )


def reset_vectorstore() -> None:
    if settings.vectorstore_dir.exists():
        shutil.rmtree(settings.vectorstore_dir)


def build_vectorstore(chunks: list[Document], reset: bool = False) -> Chroma:
    if reset:
        reset_vectorstore()
    store = load_vectorstore()
    # 대량 문서 시 임베딩 API rate limit 을 고려해 배치로 추가한다.
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
    return store
