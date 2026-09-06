import json

import pytest

from rag import chain
from rag.chain import INSUFFICIENT_MESSAGE, answer_question
from tests.conftest import FakeChatModel


@pytest.fixture
def patch_evidence(monkeypatch, retrieved_docs):
    def _patch(reply: str, docs=retrieved_docs):
        fake = FakeChatModel(reply)
        monkeypatch.setattr(
            chain,
            "retrieve_with_gate",
            lambda q, k=None: (docs, docs[0][1] if docs else 0.0),
        )
        monkeypatch.setattr(chain, "get_chat_model", lambda: fake)
        monkeypatch.setattr(chain.settings, "similarity_threshold", 0.3)
        monkeypatch.setattr(chain.settings, "prompt_mode", "evidence_first")
        return fake

    return _patch


def test_evidence_first_answered_with_quotes_and_citations(patch_evidence):
    fake = patch_evidence(
        json.dumps(
            {
                "quotes": [
                    {
                        "source": 2,
                        "text": "교권보호위원회는 사안을 심의하여 조치를 결정한다.",
                    }
                ],
                "answer": "위원회가 심의하여 결정합니다 [2].",
                "insufficient": False,
            }
        )
    )
    r = answer_question("누가 결정하나?")
    assert r.status == "answered" and r.llm_called
    assert r.quotes[0]["source"] == 2 and [c.index for c in r.citations] == [2]
    assert (
        "[참고 자료]" in fake.last_messages[0]["content"]
        and "quotes" in fake.last_messages[0]["content"]
    )


def test_evidence_first_uses_quote_sources_when_answer_lacks_markers(patch_evidence):
    patch_evidence(
        json.dumps(
            {
                "quotes": [{"source": 1, "text": "x"}, {"source": 3, "text": "y"}],
                "answer": "표시 없는 답",
                "insufficient": False,
            }
        )
    )
    r = answer_question("q")
    assert [c.index for c in r.citations] == [1, 3]


@pytest.mark.parametrize(
    "payload",
    [
        {"quotes": [], "answer": "", "insufficient": True},
        {"quotes": [], "answer": "근거 없이 쓴 답", "insufficient": False},
    ],
)
def test_evidence_first_abstains_without_quotes(patch_evidence, payload):
    fake = patch_evidence(json.dumps(payload))
    r = answer_question("문서에 없는 질문")
    assert (
        r.status == "insufficient"
        and r.answer == INSUFFICIENT_MESSAGE
        and fake.calls == 1
    )


def test_evidence_first_falls_back_to_text_on_bad_json(patch_evidence):
    patch_evidence("JSON 이 아닌 답변입니다 [1]")
    r = answer_question("q")
    assert (
        r.status == "answered"
        and r.answer == "JSON 이 아닌 답변입니다 [1]"
        and [c.index for c in r.citations] == [1]
    )


def test_score_gate_applies_before_prompt(patch_evidence, retrieved_docs):
    low = [(d, 0.1) for d, _ in retrieved_docs]
    fake = patch_evidence(
        '{"quotes": [], "answer": "", "insufficient": true}', docs=low
    )
    r = answer_question("무관")
    assert r.status == "insufficient" and fake.calls == 0 and r.gate_score == 0.1
