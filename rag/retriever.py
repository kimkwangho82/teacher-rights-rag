"""관련도 점수를 포함한 검색.

Chroma 를 cosine 공간으로 만들었으므로 relevance score 는 0(무관)~1(동일) 범위다.
점수는 abstain 판단(rag.chain)과 Eval 에서 재사용한다.

retrieval_mode:
- dense : 벡터 검색 상위 k
- hybrid: dense + BM25 를 RRF 로 결합 (rag.hybrid). 반환 점수는 dense 점수를 유지하고,
          abstain 게이트에는 결합 순위와 무관하게 dense 최고 점수를 쓴다.
"""

from langchain_core.documents import Document

from rag.config import settings
from rag.vectorstore import load_vectorstore


def retrieve_with_gate(
    question: str, k: int | None = None
) -> tuple[list[tuple[Document, float]], float]:
    """(상위 k (doc, dense_score), 게이트용 dense 최고 점수)."""
    k = k or settings.top_k
    if settings.retrieval_mode == "hybrid":
        from rag.hybrid import hybrid_retrieve

        return hybrid_retrieve(question, k)

    store = load_vectorstore()
    results = store.similarity_search_with_relevance_scores(question, k=k)
    results = sorted(results, key=lambda r: r[1], reverse=True)  # 관련도 내림차순 보장
    return results, (results[0][1] if results else 0.0)


def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    return retrieve_with_gate(question, k)[0]
