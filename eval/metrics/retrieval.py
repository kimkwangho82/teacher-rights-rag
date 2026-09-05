"""검색 단계 규칙 기반 지표. gold 근거는 page 단위."""

from __future__ import annotations


def recall_at_k(
    gold_pages: list[int], retrieved_pages: list[int], k: int | None = None
) -> float | None:
    """gold 페이지 중 상위 k 검색 결과의 페이지에 포함된 비율. gold 가 없으면 None."""
    if not gold_pages:
        return None
    top = retrieved_pages if k is None else retrieved_pages[:k]
    found = set(top)
    return sum(1 for p in set(gold_pages) if p in found) / len(set(gold_pages))


def mrr(
    gold_pages: list[int], retrieved_pages: list[int], k: int | None = None
) -> float | None:
    """gold 페이지에 속하는 첫 검색 결과의 1/rank. 없으면 0, gold 가 없으면 None."""
    if not gold_pages:
        return None
    gold = set(gold_pages)
    top = retrieved_pages if k is None else retrieved_pages[:k]
    for rank, p in enumerate(top, start=1):
        if p in gold:
            return 1.0 / rank
    return 0.0
