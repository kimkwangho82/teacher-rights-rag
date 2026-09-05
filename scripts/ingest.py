"""Ingest 파이프라인: data/raw 로드 → 정규화 → 청킹 → 임베딩 → data/vectorstore 저장.

실행:
  uv run python -m scripts.ingest            # 기존 컬렉션에 추가(같은 chunk_id 는 덮어씀)
  uv run python -m scripts.ingest --reset    # 벡터스토어 삭제 후 재생성
  uv run python -m scripts.ingest --dry-run  # 임베딩 없이 로드/청킹 통계만 출력

부산물: data/processed/chunks.jsonl (청크 전문 + 메타데이터, Gold Set 작성 시 근거 참조용)
"""
import argparse
import json
from collections import Counter

from rag.config import settings
from rag.loaders import load_documents
from rag.splitter import split_documents
from rag.vectorstore import build_vectorstore


def dump_chunks(chunks) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.processed_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}, ensure_ascii=False) + "\n")
    print(f"chunks dumped -> {out}")


def print_stats(docs, chunks) -> None:
    n_docs = len({d.metadata["doc_id"] for d in docs})
    pages_per_doc = Counter(d.metadata["doc_id"] for d in docs)
    lengths = [len(c.page_content) for c in chunks]
    print("=== ingest stats ===")
    print(f"documents : {n_docs}")
    print(f"pages     : {len(docs)}")
    print(f"chunks    : {len(chunks)}")
    if lengths:
        print(f"chunk len : avg {sum(lengths)/len(lengths):.0f} / min {min(lengths)} / max {max(lengths)}")
    print(f"settings  : chunk_size={settings.chunk_size} overlap={settings.chunk_overlap} "
          f"embedding={settings.embedding_model}")
    print("--- pages per document ---")
    for doc_id, n in sorted(pages_per_doc.items()):
        print(f"  {doc_id}: {n}p")
    if n_docs < 20 and len(docs) < 50:
        print("WARNING: Corpus 규모가 작습니다 (권장: 문서 20개 이상 또는 50페이지 이상)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="벡터스토어를 삭제하고 재생성")
    parser.add_argument("--dry-run", action="store_true", help="임베딩/저장 없이 통계만 출력")
    args = parser.parse_args()

    docs = load_documents()
    if not docs:
        raise SystemExit(f"no documents found in {settings.raw_dir}")
    chunks = split_documents(docs)
    print_stats(docs, chunks)
    dump_chunks(chunks)

    if args.dry_run:
        return
    build_vectorstore(chunks, reset=args.reset)
    print(f"vectorstore built -> {settings.vectorstore_dir} (collection={settings.collection_name})")


if __name__ == "__main__":
    main()
