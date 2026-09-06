import json

import pytest

from rag import hybrid
from rag.bm25 import BM25Index
from rag.evidence_prompt import parse_evidence_output
from rag.hybrid import fuse
from rag.jsonutil import JSONExtractError, extract_json
from tests.conftest import make_doc


def test_fuse_keeps_dense_scores_and_marks_lexical_only_as_zero():
    a, b, c, d = (make_doc(f"doc {n}", page=n, idx=0) for n in (1, 2, 3, 4))
    dense = [(a, 0.7), (b, 0.6), (c, 0.5)]
    lexical = [(b, 9.0), (d, 8.0), (a, 7.0)]
    out = fuse(dense, lexical, k=3, rrf_k=60)
    ids = [doc.metadata["chunk_id"] for doc, _ in out]
    assert ids[:2] == ["manual:p2:c0", "manual:p1:c0"]  # 양쪽 상위
    scores = {doc.metadata["chunk_id"]: s for doc, s in out}
    assert scores["manual:p2:c0"] == 0.6 and scores["manual:p1:c0"] == 0.7
    assert scores.get("manual:p4:c0", None) in (None, 0.0)


def test_hybrid_retrieve_returns_dense_top_score_for_gate(monkeypatch):
    p23 = make_doc("시·도교권보호위원회의 위원은 10명 이상 20명 이하", page=23, idx=0)
    p26 = make_doc("지역교권보호위원회의 위원은 10명 이상 50명 이하", page=26, idx=0)
    p28 = make_doc("지역교권보호위원회 회의 소집 및 정족수", page=28, idx=0)

    class FakeStore:
        def similarity_search_with_relevance_scores(self, q, k):
            return [
                (p26, 0.55),
                (p28, 0.50),
                (p23, 0.45),
            ]  # dense 는 지역위를 상위에 둠

    monkeypatch.setattr(hybrid, "load_vectorstore", lambda: FakeStore())
    monkeypatch.setattr(hybrid, "get_bm25_index", lambda: BM25Index([p23, p26, p28]))
    monkeypatch.setattr(hybrid.settings, "hybrid_fetch_k", 3)
    results, dense_top = hybrid.hybrid_retrieve("시·도교권보호위원회 위원 정수", k=2)
    assert dense_top == 0.55
    pages = [doc.metadata["page"] for doc, _ in results]
    assert 23 in pages  # BM25 가 시·도 위원회를 끌어올림
    assert all(s in (0.55, 0.50, 0.45, 0.0) for _, s in results)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('설명 {"a": [1]} 끝') == {"a": [1]}
    with pytest.raises(JSONExtractError):
        extract_json("no json")
    with pytest.raises(JSONExtractError):
        extract_json("[1, 2]")


def test_parse_evidence_output_answered():
    out = parse_evidence_output(
        json.dumps(
            {
                "quotes": [{"source": 2, "text": "10명 이상 20명 이하"}],
                "answer": "10~20명입니다 [2]",
                "insufficient": False,
            }
        )
    )
    assert out.parsed and not out.insufficient
    assert (
        out.quotes == [{"source": 2, "text": "10명 이상 20명 이하"}]
        and out.answer == "10~20명입니다 [2]"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"quotes": [], "answer": "", "insufficient": True},
        {
            "quotes": [],
            "answer": "뭔가 답",
            "insufficient": False,
        },  # 인용 없으면 insufficient
        {
            "quotes": [{"source": 1, "text": "x"}],
            "answer": "",
            "insufficient": False,
        },  # 답 없으면 insufficient
    ],
)
def test_parse_evidence_output_insufficient(payload):
    assert parse_evidence_output(json.dumps(payload)).insufficient


def test_parse_evidence_output_fallback_on_bad_json():
    out = parse_evidence_output("그냥 텍스트 답변 [1]")
    assert (
        not out.parsed and out.answer == "그냥 텍스트 답변 [1]" and not out.insufficient
    )
