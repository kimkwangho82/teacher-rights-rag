import pytest
from langchain_core.documents import Document


def make_doc(
    text: str, page: int = 1, idx: int = 0, doc_id: str = "manual"
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "title": "테스트 매뉴얼",
            "source": "manual.pdf",
            "page": page,
            "chunk_id": f"{doc_id}:p{page}:c{idx}",
        },
    )


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeChatModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return FakeResponse(self.reply)


@pytest.fixture
def retrieved_docs():
    return [
        (
            make_doc(
                "학교장은 침해 사안을 인지하면 즉시 피해 교원을 보호한다.",
                page=45,
                idx=0,
            ),
            0.71,
        ),
        (
            make_doc(
                "교권보호위원회는 사안을 심의하여 조치를 결정한다.", page=46, idx=0
            ),
            0.64,
        ),
        (make_doc("무관한 내용의 청크.", page=10, idx=1), 0.40),
    ]
