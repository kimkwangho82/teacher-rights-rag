import pytest

from eval.metrics.abstain import abstain_matrix
from eval.metrics.retrieval import mrr, recall_at_k


@pytest.mark.parametrize(
    "gold,retrieved,expected",
    [
        ([41, 42], [41, 10, 42, 5], 1.0),
        ([41, 42], [10, 42, 5, 6], 0.5),
        ([41], [1, 2, 3], 0.0),
        ([41, 41], [41], 1.0),  # 중복 gold 는 한 번만
    ],
)
def test_recall_at_k(gold, retrieved, expected):
    assert recall_at_k(gold, retrieved) == expected


def test_recall_none_without_gold():
    assert recall_at_k([], [1, 2]) is None
    assert mrr([], [1, 2]) is None


def test_recall_respects_k():
    assert recall_at_k([5], [1, 2, 5], k=2) == 0.0
    assert recall_at_k([5], [1, 2, 5], k=3) == 1.0


@pytest.mark.parametrize(
    "gold,retrieved,expected",
    [
        ([41], [41, 2], 1.0),
        ([41], [2, 41], 0.5),
        ([41, 42], [9, 9, 42], 1 / 3),
        ([41], [1, 2], 0.0),
    ],
)
def test_mrr(gold, retrieved, expected):
    assert mrr(gold, retrieved) == pytest.approx(expected)


def test_abstain_matrix_and_rates():
    m = abstain_matrix(
        [
            (True, "answered"),
            (True, "answered"),
            (True, "insufficient"),
            (False, "insufficient"),
            (False, "answered"),
        ]
    )
    assert (m.answerable_answered, m.answerable_abstained) == (2, 1)
    assert (m.unanswerable_answered, m.unanswerable_abstained) == (1, 1)
    assert m.answer_rate == pytest.approx(2 / 3)
    assert m.abstain_rate == pytest.approx(0.5)


def test_abstain_rates_none_when_empty():
    m = abstain_matrix([(True, "answered")])
    assert m.abstain_rate is None and m.answer_rate == 1.0
