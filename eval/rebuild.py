"""기존 리포트 디렉터리의 items.jsonl 을 현재 집계 규칙으로 다시 집계한다 (API 호출 없음).

  uv run python -m eval.rebuild eval/reports/baseline-a_* [eval/reports/exp1-*]

집계 규칙이 바뀌었을 때(예: 거부 응답 Correctness 규칙화) 과거 실행과 새 실행을 같은 기준으로 비교하기 위한 도구.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics.retrieval import mrr, recall_at_k
from eval.report import build_report, write_report
from eval.run import LABELS_PATH
from eval.runner import gold_hash, load_gold
from eval.schema import CriterionResult, GoldItem, ItemResult, RunConfig


def normalize_item(item: ItemResult, gold_by_id: dict[str, GoldItem]) -> ItemResult:
    """현재 Gold Set 기준으로 규칙 지표를 재계산하고, 거부 응답의 Correctness 를 규칙으로 확정."""
    gold = gold_by_id.get(item.id)
    if gold is not None:
        pages = [c.page for c in item.retrieved]
        item.evidence_pages = gold.evidence_pages
        item.answerable = gold.answerable
        item.recall_at_k = recall_at_k(gold.evidence_pages, pages)
        item.recall_at_3 = recall_at_k(gold.evidence_pages, pages, k=3)
        item.mrr = mrr(gold.evidence_pages, pages)
    if item.status != "insufficient":
        return item
    answerable = item.answerable
    for scores in [item.judge, *item.judge_repeats]:
        scores.correctness = 1.0 if not answerable else 0.0
        scores.criteria = [
            CriterionResult(
                text=c.text,
                required=c.required,
                passed=not answerable,
                reason="규칙: 거부 응답",
            )
            for c in scores.criteria
        ]
    return item


def rebuild(dir_: Path) -> Path:
    config = RunConfig(
        **json.loads((dir_ / "report.json").read_text(encoding="utf-8"))["config"]
    )
    items = [
        ItemResult(**json.loads(l))
        for l in (dir_ / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    gold_by_id = {g.id: g for g in load_gold()}
    items = [normalize_item(it, gold_by_id) for it in items]
    config.gold_set_hash = gold_hash()
    write_report(build_report(config, items, LABELS_PATH), dir_)
    return dir_


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dirs", nargs="+", type=Path)
    for d in p.parse_args().dirs:
        print("rebuilt", rebuild(d))


if __name__ == "__main__":
    main()
