"""RAGAS(Es et al., 2023) 정의를 따르는 LLM Judge 지표. 라이브러리 대신 직접 구현.

- faithfulness      = 지지되는 주장 수 / 전체 주장 수        (식: F = |V| / |S|)
- context_relevance = 추출된 관련 문장 수 / 문맥 전체 문장 수 (식 2)
- answer_relevance  = mean cos(emb(q), emb(q_i)), q_i 는 답변에서 역생성한 질문 (식 1)
"""

from __future__ import annotations

import math
import re

from eval.judge import Judge
from eval.schema import Statement

# 문장 단위: PDF 줄바꿈은 문장 경계가 아니므로 먼저 공백으로 접고, 문장 종결부호 뒤 또는 글머리표 앞에서 나눈다.
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\s+(?=[•※▶■①-⑳]|\(\d+\)\s|\d+\.\s)")
_HEADER_RE = re.compile(r"^\[\d+\] \(.*?, p\.\d+\)$")
MIN_SENTENCE_CHARS = 10


def split_sentences(text: str) -> list[str]:
    """문맥을 문장 단위로 나눈다. 청크 헤더([n] (제목, p.N))와 10자 미만 조각은 제외."""
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not _HEADER_RE.match(ln.strip())
    ]
    flat = " ".join(lines)
    return [
        s.strip()
        for s in _SENT_SPLIT.split(flat)
        if len(s.strip()) >= MIN_SENTENCE_CHARS
    ]


def faithfulness(
    judge: Judge, question: str, answer: str, context: str
) -> tuple[float | None, list[Statement]]:
    out = judge.ask("faithfulness_statements", question=question, answer=answer)
    statements = [
        s.strip() for s in out.get("statements", []) if isinstance(s, str) and s.strip()
    ]
    if not statements:
        return None, []

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
    out = judge.ask("faithfulness_verify", context=context, statements=numbered)
    verdicts = out.get("verdicts", [])
    results: list[Statement] = []
    for i, s in enumerate(statements):
        v = verdicts[i] if i < len(verdicts) and isinstance(verdicts[i], dict) else {}
        supported = v.get("supported")
        results.append(
            Statement(
                text=s,
                supported=bool(supported) if supported is not None else None,
                reason=str(v.get("reason", "")),
            )
        )
    judged = [r for r in results if r.supported is not None]
    if not judged:
        return None, results
    return sum(1 for r in judged if r.supported) / len(judged), results


def context_relevance(
    judge: Judge, question: str, context: str
) -> tuple[float | None, int, int]:
    total = len(split_sentences(context))
    if total == 0:
        return None, 0, 0
    out = judge.ask("context_relevance", question=question, context=context)
    if out.get("insufficient"):
        return 0.0, total, 0
    extracted = [
        s for s in out.get("sentences", []) if isinstance(s, str) and s.strip()
    ]
    relevant = min(len(extracted), total)
    return relevant / total, total, relevant


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def answer_relevance(
    judge: Judge, embeddings, question: str, answer: str, n: int = 3
) -> tuple[float | None, list[str]]:
    out = judge.ask("answer_relevance", answer=answer, n=str(n))
    generated = [
        q for q in out.get("questions", []) if isinstance(q, str) and q.strip()
    ][:n]
    if not generated:
        return None, []
    vecs = embeddings.embed_documents([question, *generated])
    q_vec, gen_vecs = vecs[0], vecs[1:]
    return sum(_cosine(q_vec, g) for g in gen_vecs) / len(gen_vecs), generated
