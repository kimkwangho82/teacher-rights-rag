from langchain_core.documents import Document

from rag.loaders import clean_text, make_doc_id
from rag.splitter import split_documents


def test_make_doc_id_slugifies_korean_filenames():
    assert (
        make_doc_id("2025+교육활동+보호+매뉴얼(배포용).pdf")
        == "2025-교육활동-보호-매뉴얼-배포용"
    )


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc  \n d") == "a b\n\nc\nd"


def test_split_assigns_sequential_chunk_ids_per_page():
    doc = Document(
        page_content=("문단 " * 150) + "\n\n" + ("둘째 " * 150),
        metadata={"doc_id": "x", "page": 3},
    )
    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks] == [
        f"x:p3:c{i}" for i in range(len(chunks))
    ]
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_split_preserves_source_metadata_and_overlap():
    doc = Document(
        page_content="가나다라 " * 200,
        metadata={"doc_id": "x", "page": 1, "title": "T"},
    )
    chunks = split_documents([doc], chunk_size=200, chunk_overlap=40)
    assert all(c.metadata["title"] == "T" for c in chunks)
    # overlap: 다음 청크의 시작이 이전 청크의 끝부분과 겹친다
    tail = chunks[0].page_content[-20:]
    assert tail.split()[0] in chunks[1].page_content[:60]


def test_split_drops_tiny_chunks():
    doc = Document(page_content="짧음", metadata={"doc_id": "x", "page": 1})
    assert split_documents([doc]) == []
