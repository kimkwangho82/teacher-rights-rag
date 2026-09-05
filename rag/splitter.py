"""문서를 청크로 분할한다.

전략 (README 'Chunking 전략' 참고):
- RecursiveCharacterTextSplitter, 문단(\\n\\n) → 줄(\\n) → 문장(". ") → 공백 순으로 분리.
  교육부/교육청 매뉴얼은 항목·문단 단위 구조이므로 문단 경계를 최우선으로 존중한다.
- chunk_size / chunk_overlap 은 config 로 제어 (개선 실험 변수).
- min_chunk_chars 미만의 청크(쪽번호·머리글 조각)는 제거해 노이즈 검색을 막는다.
- 각 청크에 chunk_id = "{doc_id}:p{page}:c{idx}" 를 부여해 Citation 의 안정적 키로 사용한다.
"""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def get_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
        if chunk_overlap is None
        else chunk_overlap,
        separators=SEPARATORS,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = get_splitter(chunk_size, chunk_overlap)
    chunks = [
        c
        for c in splitter.split_documents(docs)
        if len(c.page_content) >= settings.min_chunk_chars
    ]

    counters: dict[tuple[str, int], int] = defaultdict(int)
    for c in chunks:
        key = (c.metadata.get("doc_id", "doc"), int(c.metadata.get("page", 1)))
        idx = counters[key]
        counters[key] += 1
        c.metadata["chunk_id"] = f"{key[0]}:p{key[1]}:c{idx}"
        c.metadata["chunk_index"] = idx
    return chunks
