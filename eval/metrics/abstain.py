"""Abstain 정확도: 답 없는 질의 거부율과 답 있는 질의 응답률을 분리해 집계."""

from __future__ import annotations

from eval.schema import AbstainMatrix


def abstain_matrix(items: list[tuple[bool, str]]) -> AbstainMatrix:
    """items: (answerable, status) 목록. status 는 'answered' | 'insufficient'."""
    m = AbstainMatrix()
    for answerable, status in items:
        answered = status == "answered"
        if answerable and answered:
            m.answerable_answered += 1
        elif answerable and not answered:
            m.answerable_abstained += 1
        elif not answerable and answered:
            m.unanswerable_answered += 1
        else:
            m.unanswerable_abstained += 1
    return m
