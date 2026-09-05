import pytest

from rag import chain
from rag.chain import INSUFFICIENT_MESSAGE, INSUFFICIENT_TOKEN, answer_question
from tests.conftest import FakeChatModel


@pytest.fixture
def patch_chain(monkeypatch, retrieved_docs):
    def _patch(reply: str, docs=retrieved_docs, threshold: float = 0.3):
        fake = FakeChatModel(reply)
        monkeypatch.setattr(chain, "retrieve", lambda q, k=None: docs)
        monkeypatch.setattr(chain, "get_chat_model", lambda: fake)
        monkeypatch.setattr(chain.settings, "similarity_threshold", threshold)
        return fake

    return _patch


def test_low_score_abstains_without_calling_llm(patch_chain, retrieved_docs):
    fake = patch_chain("should not be used", threshold=0.9)
    result = answer_question("오늘 날씨는?")
    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_MESSAGE
    assert result.llm_called is False
    assert fake.calls == 0
    assert result.citations == []


def test_empty_retrieval_abstains(patch_chain):
    fake = patch_chain("x", docs=[])
    result = answer_question("아무거나")
    assert result.status == "insufficient" and fake.calls == 0


def test_model_token_abstains(patch_chain):
    fake = patch_chain(f"  {INSUFFICIENT_TOKEN}\n")
    result = answer_question("문서에 없는 질문")
    assert result.status == "insufficient"
    assert result.llm_called is True and fake.calls == 1
    assert result.citations == []


def test_answer_extracts_cited_indices_in_order(patch_chain):
    fake = patch_chain("위원회가 심의한다 [2]. 학교장은 즉시 보호한다 [1][2].")
    result = answer_question("절차는?")
    assert result.status == "answered"
    assert [c.index for c in result.citations] == [2, 1]
    assert result.citations[0].doc.metadata["page"] == 46
    # 프롬프트에 번호 매긴 문맥이 들어갔는지
    system = fake.last_messages[0]["content"]
    assert "[1] (테스트 매뉴얼, p.45)" in system and "[3] (테스트 매뉴얼, p.10)" in system


def test_answer_without_markers_returns_all_retrieved(patch_chain):
    patch_chain("인용 표시 없는 답변")
    result = answer_question("절차는?")
    assert result.status == "answered"
    assert [c.index for c in result.citations] == [1, 2, 3]


def test_unknown_index_is_ignored(patch_chain):
    patch_chain("근거 [9] 와 [1]")
    result = answer_question("절차는?")
    assert [c.index for c in result.citations] == [1]
