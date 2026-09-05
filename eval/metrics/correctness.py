"""Correctness: gold 수용 기준(체크리스트) 충족률. required 항목 실패 시 0."""

from __future__ import annotations

from eval.judge import Judge
from eval.schema import Criterion, CriterionResult


def correctness(
    judge: Judge, question: str, answer: str, criteria: list[Criterion]
) -> tuple[float | None, list[CriterionResult]]:
    if not criteria:
        return None, []
    numbered = "\n".join(f"{i}. {c.text}" for i, c in enumerate(criteria, start=1))
    out = judge.ask("correctness", question=question, answer=answer, criteria=numbered)
    verdicts = out.get("results", [])
    results: list[CriterionResult] = []
    for i, c in enumerate(criteria):
        v = verdicts[i] if i < len(verdicts) and isinstance(verdicts[i], dict) else {}
        passed = v.get("passed")
        results.append(
            CriterionResult(
                text=c.text,
                required=c.required,
                passed=bool(passed) if passed is not None else None,
                reason=str(v.get("reason", "")),
            )
        )
    judged = [r for r in results if r.passed is not None]
    if not judged:
        return None, results
    if any(r.required and not r.passed for r in judged):
        return 0.0, results
    return sum(1 for r in judged if r.passed) / len(judged), results
