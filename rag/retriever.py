"""관련도 점수를 포함한 검색.

Chroma 를 cosine 공간으로 만들었으므로 relevance score 는 0(무관)~1(동일) 범위다.
점수는 abstain 판단(rag.chain)과 Eval 에서 재사용한다.
"""

from langchain_core.documents import Document

from rag.config import settings
from rag.vectorstore import load_vectorstore


def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    store = load_vectorstore()
    results = store.similarity_search_with_relevance_scores(
        question, k=k or settings.top_k
    )
    # 관련도 내림차순 보장
    return sorted(results, key=lambda r: r[1], reverse=True)
