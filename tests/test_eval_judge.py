import json

import pytest

from eval.judge import Judge, JudgeError, parse_json
from eval.metrics import ragas
from eval.metrics.correctness import correctness
from eval.schema import Criterion
from tests.conftest import FakeResponse


class ScriptedLLM:
    """호출 순서대로 미리 정한 응답을 돌려준다."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeResponse(self.replies.pop(0))


def make_judge(tmp_path, replies, use_cache=True, repeat_index=0):
    return Judge(
        model="fake",
        seed=1,
        cache_dir=tmp_path,
        use_cache=use_cache,
        repeat_index=repeat_index,
        llm=ScriptedLLM(replies),
    )


def test_parse_json_handles_code_fence_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('결과입니다: {"a": [1, 2]} 끝') == {"a": [1, 2]}
    with pytest.raises(JudgeError):
        parse_json("no json here")


def test_judge_caches_by_input_and_repeat_index(tmp_path):
    j = make_judge(tmp_path, ['{"statements": ["a"]}'])
    out1 = j.ask("faithfulness_statements", question="q", answer="a")
    out2 = j.ask(
        "faithfulness_statements", question="q", answer="a"
    )  # cache hit, LLM not called
    assert out1 == out2 == {"statements": ["a"]}
    assert j.calls == 1 and j.cache_hits == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    # 다른 repeat_index 는 별도 캐시 키 → 재호출
    j2 = make_judge(tmp_path, ['{"statements": ["b"]}'], repeat_index=1)
    assert j2.ask("faithfulness_statements", question="q", answer="a") == {
        "statements": ["b"]
    }
    assert j2.calls == 1


def test_judge_retries_once_on_bad_json(tmp_path):
    j = make_judge(tmp_path, ["garbage", '{"ok": true}'])
    assert j.ask("answer_relevance", answer="x", n="3") == {"ok": True}
    assert j.calls == 2


def test_judge_raises_after_two_failures(tmp_path):
    j = make_judge(tmp_path, ["garbage", "still garbage"], use_cache=False)
    with pytest.raises(JudgeError):
        j.ask("answer_relevance", answer="x", n="3")


def test_faithfulness_ratio_and_statement_log(tmp_path):
    j = make_judge(
        tmp_path,
        [
            json.dumps({"statements": ["s1", "s2", "s3"]}),
            json.dumps(
                {
                    "verdicts": [
                        {"statement": "s1", "reason": "r", "supported": True},
                        {"statement": "s2", "reason": "r", "supported": False},
                        {"statement": "s3", "reason": "r", "supported": True},
                    ]
                }
            ),
        ],
    )
    score, statements = ragas.faithfulness(j, "q", "answer", "context")
    assert score == pytest.approx(2 / 3)
    assert [s.supported for s in statements] == [True, False, True]


def test_faithfulness_none_when_no_statements(tmp_path):
    j = make_judge(tmp_path, ['{"statements": []}'])
    assert ragas.faithfulness(j, "q", "안녕하세요", "ctx") == (None, [])


def test_context_relevance_ratio(tmp_path):
    ctx = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다. 네 번째 문장입니다."
    j = make_judge(
        tmp_path,
        [
            '{"sentences": ["첫 번째 문장입니다.", "세 번째 문장입니다."], "insufficient": false}'
        ],
    )
    score, total, relevant = ragas.context_relevance(j, "q", ctx)
    assert (total, relevant) == (4, 2) and score == 0.5


def test_context_relevance_insufficient_is_zero(tmp_path):
    j = make_judge(tmp_path, ['{"sentences": [], "insufficient": true}'])
    assert (
        ragas.context_relevance(
            j, "q", "첫 번째 문장은 충분히 길다. 두 번째 문장도 충분히 길다."
        )[0]
        == 0.0
    )


def test_answer_relevance_uses_embedding_cosine(tmp_path):
    class FakeEmb:
        def embed_documents(self, texts):
            # 원 질문 [1,0], 생성 질문 [1,0], [0,1] → cos 1.0, 0.0 → 평균 0.5
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    j = make_judge(tmp_path, ['{"questions": ["g1", "g2"]}'])
    score, gen = ragas.answer_relevance(j, FakeEmb(), "q", "a", n=2)
    assert score == pytest.approx(0.5) and gen == ["g1", "g2"]


def test_correctness_required_failure_zeroes_score(tmp_path):
    criteria = [Criterion(text="A", required=True), Criterion(text="B", required=False)]
    j = make_judge(
        tmp_path,
        [
            json.dumps(
                {
                    "results": [
                        {"criterion": "A", "reason": "", "passed": False},
                        {"criterion": "B", "reason": "", "passed": True},
                    ]
                }
            )
        ],
    )
    score, results = correctness(j, "q", "a", criteria)
    assert score == 0.0 and [r.passed for r in results] == [False, True]


def test_correctness_partial(tmp_path):
    criteria = [
        Criterion(text="A", required=True),
        Criterion(text="B"),
        Criterion(text="C"),
    ]
    j = make_judge(
        tmp_path,
        [
            json.dumps(
                {
                    "results": [
                        {"criterion": "A", "passed": True},
                        {"criterion": "B", "passed": False},
                        {"criterion": "C", "passed": True},
                    ]
                }
            )
        ],
    )
    assert correctness(j, "q", "a", criteria)[0] == pytest.approx(2 / 3)
