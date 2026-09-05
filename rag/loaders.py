"""data/raw 아래의 문서를 로드한다 (PDF, TXT, MD).

각 Document 의 metadata 를 아래 형태로 정규화한다.
  doc_id   : 파일명 기반 slug (Citation 의 안정적 키)
  title    : manifest.json 의 제목, 없으면 파일명
  publisher/year/url : manifest.json 에 있으면 포함
  source   : data/raw 기준 상대 경로
  page     : 1-based 페이지 번호 (TXT/MD 는 1)
"""
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from rag.config import settings

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    """manifest.json 을 {file: meta} 로 반환. 없으면 빈 dict."""
    path = path or settings.manifest_path
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {d["file"]: d for d in data.get("documents", []) if "file" in d}


def make_doc_id(rel_path: str) -> str:
    stem = Path(rel_path).with_suffix("").as_posix()
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "-", stem).strip("-").lower()
    return slug or "doc"


_WS_RE = re.compile(r"[ \t　]+")
_NL_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """공문 PDF 특유의 연속 공백/개행을 정리한다. 내용은 바꾸지 않는다."""
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"page": i})
            for i, page in enumerate(reader.pages, start=1)
        ]
    if suffix in {".txt", ".md"}:
        return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"page": 1})]
    return []


def load_documents(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = raw_dir or settings.raw_dir
    manifest = load_manifest()
    docs: list[Document] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(raw_dir).as_posix()
        meta = manifest.get(rel, {})
        doc_id = make_doc_id(rel)

        for d in _load_file(path):
            text = clean_text(d.page_content)
            if not text:
                continue  # 빈 페이지(스캔 이미지 등) 스킵
            d.page_content = text
            d.metadata = {
                "doc_id": doc_id,
                "title": meta.get("title", path.stem),
                "publisher": meta.get("publisher", ""),
                "year": meta.get("year", ""),
                "url": meta.get("url", ""),
                "source": rel,
                "page": d.metadata["page"],
            }
            docs.append(d)
    return docs
