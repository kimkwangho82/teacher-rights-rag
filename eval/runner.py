"""Gold Set 순회: answer_question → 규칙 지표 → (선택) Judge 지표."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from eval.judge import Judge, prompt_hashes
from eval.metrics import correctness as corr
from eval.metrics import ragas
from eval.metrics.retrieval import mrr, recall_at_k
from eval.schema import (
    CriterionResult,
    GoldItem,
    ItemResult,
    JudgeScores,
    RetrievedChunk,
    RunConfig,
)
from rag.chain import INSUFFICIENT_MESSAGE, RagAnswer, answer_question, format_context
from rag.config import settings

logger = logging.getLogger("eval")

GOLD_PATH = Path(__file__).parent / "gold_set.jsonl"
PACKAGES = (
    "langchain",
    "langchain-openai",
    "langchain-chroma",
    "chromadb",
    "openai",
    "pypdf",
)


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    return [
        GoldItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gold_hash(path: Path = GOLD_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stratified_subset(items: list[GoldItem], n: int, seed: int) -> list[GoldItem]:
    """유형별로 최대한 균등하게 n 개 샘플 (CI 스모크용)."""
    rng = random.Random(seed)
    by_type: dict[str, list[GoldItem]] = defaultdict(list)
    for it in items:
        by_type[it.type].append(it)
    for lst in by_type.values():
        rng.shuffle(lst)
    picked: list[GoldItem] = []
    while len(picked) < n and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop())
    return sorted(picked, key=lambda i: i.id)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def index_chunk_count() -> int | None:
    try:
        from rag.vectorstore import load_vectorstore

        return load_vectorstore()._collection.count()
    except Exception:  # noqa: BLE001 - 인덱스가 없거나 열 수 없는 모든 경우 None
        return None


def build_run_config(
    name: str, judge_enabled: bool, repeat: int, subset: int | None, gold_size: int
) -> RunConfig:
    return RunConfig(
        name=name,
        started_at=datetime.now(tz=UTC).astimezone().isoformat(timespec="seconds"),
        git_sha=git_sha(),
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        judge_model=settings.judge_model if judge_enabled else None,
        seed=settings.seed,
        temperature=settings.temperature,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
        retrieval_mode=settings.retrieval_mode,
        bm25_tokenizer=settings.bm25_tokenizer,
        prompt_mode=settings.prompt_mode,
        prompt_hashes=prompt_hashes(),
        gold_set_hash=gold_hash(),
        gold_set_size=gold_size,
        index_chunks=index_chunk_count(),
        packages=package_versions(),
        repeat=repeat,
        subset=subset,
    )


def to_retrieved(result: RagAnswer) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            index=r.index,
            chunk_id=str(r.doc.metadata.get("chunk_id", "")),
            page=int(r.doc.metadata.get("page", 0)),
            score=round(r.score, 4),
            text=r.doc.page_content,
        )
        for r in result.retrieved
    ]


def judge_item(
    judge: Judge, gold: GoldItem, result: RagAnswer, embeddings=None
) -> JudgeScores:
    """Judge 지표 계산. abstain 답변은 faithfulness/answer_relevance/context_relevance 제외, correctness 만 채점."""
    scores = JudgeScores()
    context = format_context(result.retrieved)
    answered = result.status == "answered"

    def guarded(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 지표 하나의 실패가 런 전체를 막지 않게
            logger.warning("[%s] %s failed: %s", gold.id, name, e)
            scores.errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    if not answered:
        # 거부 응답의 Correctness 는 규칙으로 확정: 답 있는 질의 거부 = 0, 답 없는 질의 거부 = 1.
        # (Judge 가 부정 조건 규칙을 잘못 적용해 거부 응답을 통과시킨 사례가 있어 규칙으로 고정)
        scores.correctness = 1.0 if not gold.answerable else 0.0
        scores.criteria = [
            CriterionResult(
                text=c.text,
                required=c.required,
                passed=not gold.answerable,
                reason="규칙: 거부 응답",
            )
            for c in gold.acceptance_criteria
        ]
    else:
        r = guarded(
            "correctness",
            lambda: corr.correctness(
                judge, gold.question, result.answer, gold.acceptance_criteria
            ),
        )
        if r:
            scores.correctness, scores.criteria = r

    if answered and result.retrieved:
        r = guarded(
            "faithfulness",
            lambda: ragas.faithfulness(judge, gold.question, result.answer, context),
        )
        if r:
            scores.faithfulness, scores.statements = r
        r = guarded(
            "context_relevance",
            lambda: ragas.context_relevance(judge, gold.question, context),
        )
        if r:
            (
                scores.context_relevance,
                scores.context_sentences_total,
                scores.context_sentences_relevant,
            ) = r
        if embeddings is not None:
            r = guarded(
                "answer_relevance",
                lambda: ragas.answer_relevance(
                    judge, embeddings, gold.question, result.answer
                ),
            )
            if r:
                scores.answer_relevance, scores.generated_questions = r
    elif result.retrieved and gold.answerable:
        # 과잉 거부 케이스도 검색 문맥의 관련도는 측정 (검색 문제인지 생성 문제인지 분리)
        r = guarded(
            "context_relevance",
            lambda: ragas.context_relevance(judge, gold.question, context),
        )
        if r:
            (
                scores.context_relevance,
                scores.context_sentences_total,
                scores.context_sentences_relevant,
            ) = r
    return scores


def evaluate_item(
    gold: GoldItem,
    judges: list[Judge] | None,
    embeddings=None,
    answer_fn=answer_question,
) -> ItemResult:
    result = answer_fn(gold.question)
    retrieved = to_retrieved(result)
    pages = [c.page for c in retrieved]

    item = ItemResult(
        id=gold.id,
        type=gold.type,
        question=gold.question,
        answerable=gold.answerable,
        answer=result.answer if result.status == "answered" else INSUFFICIENT_MESSAGE,
        status=result.status,
        llm_called=result.llm_called,
        retrieved=retrieved,
        cited_indices=[c.index for c in result.citations],
        quotes=list(getattr(result, "quotes", [])),
        gate_score=getattr(result, "gate_score", None),
        evidence_pages=gold.evidence_pages,
        recall_at_k=recall_at_k(gold.evidence_pages, pages),
        recall_at_3=recall_at_k(gold.evidence_pages, pages, k=3),
        mrr=mrr(gold.evidence_pages, pages),
    )
    if judges:
        repeats = [judge_item(j, gold, result, embeddings) for j in judges]
        item.judge = repeats[0]
        item.judge_repeats = repeats
    return item


def run_all(
    gold_items: list[GoldItem],
    judges: list[Judge] | None,
    embeddings=None,
    answer_fn=answer_question,
) -> list[ItemResult]:
    results: list[ItemResult] = []
    for i, g in enumerate(gold_items, start=1):
        logger.info(
            "[%d/%d] %s (%s) %s", i, len(gold_items), g.id, g.type, g.question[:40]
        )
        item = evaluate_item(g, judges, embeddings, answer_fn)
        logger.info(
            "  status=%s recall=%s mrr=%s faith=%s corr=%s ctx_rel=%s",
            item.status,
            _fmt(item.recall_at_k),
            _fmt(item.mrr),
            _fmt(item.judge.faithfulness),
            _fmt(item.judge.correctness),
            _fmt(item.judge.context_relevance),
        )
        results.append(item)
    return results


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"
