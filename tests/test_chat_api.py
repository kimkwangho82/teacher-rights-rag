from fastapi.testclient import TestClient

from api.main import app
from api.routers import chat as chat_router
from rag.chain import RagAnswer, Retrieved
from tests.conftest import make_doc

client = TestClient(app)


def test_chat_returns_citations(monkeypatch):
    doc = make_doc("학교장은 즉시 보호한다. " * 20, page=45)
    r = Retrieved(index=1, doc=doc, score=0.7123456)
    monkeypatch.setattr(
        chat_router,
        "answer_question",
        lambda q, top_k=None: RagAnswer(
            answer="즉시 보호한다 [1]",
            status="answered",
            citations=[r],
            retrieved=[r],
            llm_called=True,
        ),
    )
    res = client.post("/chat", json={"question": "절차는?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    assert body["answer"] == "즉시 보호한다 [1]"
    assert body["citations"][0] == {
        "index": 1,
        "chunk_id": "manual:p45:c0",
        "doc_id": "manual",
        "title": "테스트 매뉴얼",
        "page": 45,
        "snippet": doc.page_content[:200],
        "score": 0.7123,
    }
    assert isinstance(body["latency_ms"], int) and body["model"]


def test_chat_insufficient(monkeypatch):
    monkeypatch.setattr(
        chat_router,
        "answer_question",
        lambda q, top_k=None: RagAnswer(answer="답변 불가", status="insufficient"),
    )
    body = client.post("/chat", json={"question": "날씨?"}).json()
    assert body["status"] == "insufficient" and body["citations"] == []


def test_chat_passes_top_k(monkeypatch):
    seen = {}

    def fake(q, top_k=None):
        seen["top_k"] = top_k
        return RagAnswer(answer="x", status="insufficient")

    monkeypatch.setattr(chat_router, "answer_question", fake)
    client.post("/chat", json={"question": "q", "top_k": 7})
    assert seen["top_k"] == 7


def test_chat_rejects_empty_question():
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "q", "top_k": 0}).status_code == 422


def test_chat_maps_upstream_error_to_502(monkeypatch):
    def boom(q, top_k=None):
        raise RuntimeError("api down")

    monkeypatch.setattr(chat_router, "answer_question", boom)
    res = client.post("/chat", json={"question": "q"})
    assert res.status_code == 502 and "RuntimeError" in res.json()["detail"]
