"""Hybrid 검색 (개선 실험 1): dense(Chroma cosine) + BM25(문자 bigram) 를 RRF 로 결합.

반환 점수는 dense cosine relevance 를 유지한다 (abstain 게이트·Eval 의 점수 해석을 baseline 과 동일하게 유지).
BM25 에서만 올라온 청크는 dense 후보(fetch_k)에 없으므로 점수 0.0 으로 표기한다.
게이트에는 결합 순위와 무관하게 **dense 최고 점수**를 쓴다.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.bm25 import get_bm25_index, rrf_fuse
from rag.config import settings
from rag.vectorstore import load_vectorstore


def fuse(
    dense: list[tuple[Document, float]],
    lexical: list[tuple[Document, float]],
    k: int,
    rrf_k: int,
) -> list[tuple[Document, float]]:
    """두 후보 목록을 RRF 로 결합해 상위 k 개를 (doc, dense_score) 로 반환한다."""
    dense_by_id = {d.metadata["chunk_id"]: (d, s) for d, s in dense}
    lex_by_id = {d.metadata["chunk_id"]: d for d, _ in lexical}
    fused = rrf_fuse(
        [
            [d.metadata["chunk_id"] for d, _ in dense],
            [d.metadata["chunk_id"] for d, _ in lexical],
        ],
        k=rrf_k,
    )
    out: list[tuple[Document, float]] = []
    for cid, _ in fused[:k]:
        if cid in dense_by_id:
            out.append(dense_by_id[cid])
        else:
            out.append((lex_by_id[cid], 0.0))
    return out


def hybrid_retrieve(
    question: str, k: int | None = None
) -> tuple[list[tuple[Document, float]], float]:
    """(결합 상위 k, dense 최고 점수)."""
    k = k or settings.top_k
    fetch_k = max(settings.hybrid_fetch_k, k)
    dense = load_vectorstore().similarity_search_with_relevance_scores(
        question, k=fetch_k
    )
    dense = sorted(dense, key=lambda r: r[1], reverse=True)
    lexical = get_bm25_index().search(question, k=fetch_k)
    dense_top = dense[0][1] if dense else 0.0
    return fuse(dense, lexical, k=k, rrf_k=settings.rrf_k), dense_top
