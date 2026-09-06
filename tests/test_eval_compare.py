import json

from eval.compare import compare, load, load_items


def _report(
    tmp_path, name, recall, correctness, answer_rate, items=None, retrieval_mode="dense"
):
    d = tmp_path / name
    d.mkdir()
    rep = {
        "config": {
            "name": name,
            "chat_model": "m",
            "top_k": 5,
            "retrieval_mode": retrieval_mode,
            "prompt_mode": "baseline",
        },
        "overall": {
            "recall_at_k": {"mean": recall, "n": 3},
            "recall_at_3": {"mean": recall, "n": 3},
            "mrr": {"mean": None, "n": 0},
            "context_relevance": {"mean": 0.1, "n": 3},
            "faithfulness": {"mean": 0.9, "n": 3},
            "correctness": {"mean": correctness, "n": 3},
            "answer_relevance": {"mean": None, "n": 0},
        },
        "by_type": {
            "factual": {
                "correctness": {"mean": correctness, "n": 3},
                "recall_at_k": {"mean": recall, "n": 3},
                "faithfulness": {"mean": 0.9, "n": 3},
            }
        },
        "abstain": {
            "answerable_answered": 2,
            "answerable_abstained": 1,
            "unanswerable_answered": 0,
            "unanswerable_abstained": 1,
        },
        "abstain_answer_rate": answer_rate,
        "abstain_abstain_rate": 1.0,
    }
    (d / "report.json").write_text(json.dumps(rep))
    if items:
        (d / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    return d


def _item(id_, status, recall, corr):
    return {
        "id": id_,
        "type": "factual",
        "question": "질문 " + id_,
        "status": status,
        "recall_at_k": recall,
        "judge": {"correctness": corr},
    }


def test_compare_averages_multiple_reports_and_lists_item_changes(tmp_path):
    b1 = _report(
        tmp_path,
        "base-a",
        0.4,
        0.6,
        0.8,
        items=[
            _item("q1", "insufficient", 0.0, 0.0),
            _item("q2", "answered", 1.0, 1.0),
        ],
    )
    b2 = _report(
        tmp_path,
        "base-b",
        0.6,
        0.6,
        0.8,
        items=[
            _item("q1", "insufficient", 0.0, 0.0),
            _item("q2", "answered", 1.0, 1.0),
        ],
    )
    a1 = _report(
        tmp_path,
        "exp-a",
        0.8,
        0.7,
        0.9,
        items=[_item("q1", "answered", 1.0, 1.0), _item("q2", "answered", 1.0, 1.0)],
        retrieval_mode="hybrid",
    )
    md = compare(
        [load(b1), load(b2)],
        [load(a1)],
        [load_items(b1), load_items(b2)],
        [load_items(a1)],
    )
    assert "| recall_at_k | 0.500 (0.400~0.600) | 0.800 | +0.300 |" in md
    assert "retrieval_mode ⚠️" in md
    assert (
        "| q1 | factual | 질문 q1 | ins/ins → ans | 0.000 → 1.000 | 0.000 → 1.000 |"
        in md
    )
    assert "q2" not in md.split("## 문항별 변화")[1]  # 변화 없는 문항은 제외
    assert "+10.0pp" in md  # 응답률 delta


def test_compare_single_reports_without_items(tmp_path):
    b = _report(tmp_path, "b", 0.5, 0.5, 0.5)
    a = _report(tmp_path, "a", 0.5, 0.5, 0.5)
    md = compare([load(b)], [load(a)], [load_items(b)], [load_items(a)])
    assert "| correctness | 0.500 | 0.500 | +0.000 |" in md
    assert "문항별 변화" not in md
