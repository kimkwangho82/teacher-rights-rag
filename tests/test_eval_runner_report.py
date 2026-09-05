import json

from eval.report import (
    build_report,
    cohen_kappa,
    human_alignment,
    to_markdown,
    write_report,
)
from eval.runner import evaluate_item, load_gold, stratified_subset
from eval.schema import GoldItem, ItemResult, JudgeScores, RunConfig
from rag.chain import RagAnswer, Retrieved
from tests.conftest import make_doc


def gold(id_="q1", answerable=True, pages=(45,), type_="factual"):
    return GoldItem(
        id=id_,
        type=type_,
        question="질문?",
        answerable=answerable,
        evidence_pages=list(pages),
        acceptance_criteria=[{"text": "c", "required": True}],
    )


def fake_answer(status="answered", pages=(45, 46)):
    items = [
        Retrieved(index=i + 1, doc=make_doc(f"내용 {p}", page=p), score=0.7 - i * 0.1)
        for i, p in enumerate(pages)
    ]

    def fn(question, top_k=None):
        return RagAnswer(
            answer="답 [1]",
            status=status,
            citations=items[:1] if status == "answered" else [],
            retrieved=items,
            llm_called=status == "answered",
        )

    return fn


def cfg(**kw):
    base = {
        "name": "t",
        "started_at": "now",
        "git_sha": "abc",
        "chat_model": "m",
        "embedding_model": "e",
        "judge_model": None,
        "seed": 1,
        "temperature": 0.0,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "min_chunk_chars": 50,
        "top_k": 4,
        "similarity_threshold": 0.3,
        "prompt_hashes": {},
        "gold_set_hash": "g",
        "gold_set_size": 1,
        "index_chunks": None,
        "packages": {},
    }
    base.update(kw)
    return RunConfig(**base)


def test_gold_set_loads_and_is_well_formed():
    items = load_gold()
    assert len(items) >= 20
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    types = {i.type for i in items}
    assert {
        "summary",
        "reasoning",
        "unanswerable_out",
        "unanswerable_in",
    } <= types  # 생성형 + 답 없는 질의 포함
    for it in items:
        assert it.acceptance_criteria, it.id
        if it.answerable:
            assert it.evidence_pages, it.id
        else:
            assert not it.evidence_pages, it.id


def test_stratified_subset_covers_types_deterministically():
    items = load_gold()
    a = stratified_subset(items, 7, seed=1)
    b = stratified_subset(items, 7, seed=1)
    assert [i.id for i in a] == [i.id for i in b]
    assert len({i.type for i in a}) == 7


def test_evaluate_item_rule_metrics_without_judge():
    item = evaluate_item(gold(pages=(46,)), judges=None, answer_fn=fake_answer())
    assert item.status == "answered" and item.recall_at_k == 1.0 and item.mrr == 0.5
    assert item.cited_indices == [1] and [c.page for c in item.retrieved] == [45, 46]


def test_report_aggregates_and_flags():
    items = [
        evaluate_item(gold("a", pages=(45,)), None, answer_fn=fake_answer()),
        evaluate_item(
            gold("b", pages=(99,)), None, answer_fn=fake_answer()
        ),  # recall 0
        evaluate_item(
            gold("c", answerable=False, pages=(), type_="unanswerable_out"),
            None,
            answer_fn=fake_answer(),
        ),  # 답 없는데 답변
        evaluate_item(
            gold("d", pages=(45,)), None, answer_fn=fake_answer(status="insufficient")
        ),  # 과잉 거부
    ]
    items[0].judge = JudgeScores(faithfulness=0.5, correctness=1.0)
    r = build_report(cfg(), items)
    assert r.overall["recall_at_k"].mean == 2 / 3 and r.overall["recall_at_k"].n == 3
    assert r.abstain_answer_rate == 2 / 3 and r.abstain_abstain_rate == 0.0
    md = to_markdown(r)
    assert "과잉 거부" in md and "답 없는 질의에 답변" in md and "faith 0.50" in md


def test_write_report_files(tmp_path):
    items = [evaluate_item(gold(), None, answer_fn=fake_answer())]
    write_report(build_report(cfg(), items), tmp_path)
    assert (tmp_path / "report.md").exists()
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["config"]["name"] == "t" and "items" not in data
    assert len((tmp_path / "items.jsonl").read_text().splitlines()) == 1


def test_cohen_kappa():
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa([True, False, True, False], [True, True, False, False]) == 0.0
    assert cohen_kappa([], []) is None


def test_human_alignment(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"id": "a", "faithful": true, "correct": true}\n{"id": "b", "faithful": false, "correct": null}\n'
    )
    items = [
        ItemResult(
            id=i,
            type="factual",
            question="q",
            answerable=True,
            answer="a",
            status="answered",
            llm_called=True,
            retrieved=[],
            cited_indices=[],
            evidence_pages=[],
        )
        for i in ("a", "b")
    ]
    items[0].judge = JudgeScores(faithfulness=1.0, correctness=1.0)
    items[1].judge = JudgeScores(
        faithfulness=0.9, correctness=0.0
    )  # human says unfaithful, judge says faithful
    h = human_alignment(items, labels)
    assert h["faithfulness_n"] == 2 and h["faithfulness_agreement"] == 0.5
    assert h["correctness_n"] == 1 and h["correctness_agreement"] == 1.0
