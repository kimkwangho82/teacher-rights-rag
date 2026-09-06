"""BM25 어휘 검색 (hybrid 검색용).

토크나이저 (settings.bm25_tokenizer):
- bigram: 문자 2-gram. 형태소 분석기 없이 조사·띄어쓰기 변이에 강건하고 추가 의존성이 없다. (개선 실험 1)
  예) "시·도교권보호위원회" → ["시·", "·도", "도교", "교권", ...]
- kiwi  : Kiwi 형태소 분석 후 내용어(명사·수사·동사/형용사 어간·숫자·외래어 등)만 사용. 조사·어미·기호 제거. (개선 실험 3)
  예) "시·도교권보호위원회의 위원" → ["시·도", "교권", "보호", "위원회", "위원"]

인덱스 소스: data/processed/chunks.jsonl (scripts/ingest.py 가 생성). 벡터스토어와 같은 청크 집합이다.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.config import settings

logger = logging.getLogger(__name__)

_NON_TOKEN = re.compile(r"[\s　]+")
# Kiwi 품사 중 검색에 쓰는 내용어: 체언(N*), 용언 어간(VV/VA), 어근(XR), 접두사(XPN: '제'25조), 숫자(SN), 외래어/한자(SL/SH)
_KIWI_KEEP_PREFIX = ("N", "VV", "VA", "XR", "XPN", "SN", "SL", "SH")


def tokenize_bigram(text: str) -> list[str]:
    """문자 bigram. 공백은 제거하고, 길이 1 텍스트는 그대로 1개 토큰."""
    compact = _NON_TOKEN.sub("", text.lower())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


@lru_cache
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_kiwi(text: str) -> list[str]:
    """Kiwi 형태소 분석 → 내용어 형태(form)만, 소문자."""
    return [
        t.form.lower()
        for t in _kiwi().tokenize(text)
        if t.tag.startswith(_KIWI_KEEP_PREFIX)
    ]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "bigram": tokenize_bigram,
    "kiwi": tokenize_kiwi,
}


def tokenize(text: str, name: str | None = None) -> list[str]:
    return TOKENIZERS[name or settings.bm25_tokenizer](text)


class BM25Index:
    def __init__(
        self,
        docs: list[Document],
        tokenizer: Callable[[str], list[str]] = tokenize_bigram,
    ):
        self.docs = docs
        self.tokenizer = tokenizer
        self.by_chunk_id = {d.metadata["chunk_id"]: d for d in docs}
        self._bm25 = BM25Okapi([tokenizer(d.page_content) for d in docs])

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        scores = self._bm25.get_scores(self.tokenizer(query))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0]


def load_chunks(path: Path | None = None) -> list[Document]:
    path = path or settings.processed_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `uv run python -m scripts.ingest` 를 실행하세요."
        )
    docs: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.pop("text")
        docs.append(Document(page_content=text, metadata=row))
    return docs


@lru_cache
def _build_index(tokenizer_name: str) -> BM25Index:
    docs = load_chunks()
    index = BM25Index(docs, TOKENIZERS[tokenizer_name])
    logger.info("BM25 index built: %d chunks, tokenizer=%s", len(docs), tokenizer_name)
    return index


def get_bm25_index(tokenizer_name: str | None = None) -> BM25Index:
    return _build_index(tokenizer_name or settings.bm25_tokenizer)


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. ranked_lists: 검색기별 chunk_id 순위 목록. score = Σ 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
