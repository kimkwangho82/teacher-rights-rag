"""Eval Harness 진입점 — 단일 명령으로 Gold Set 전체 평가 + Report 생성.

uv run python -m eval.run --name baseline            # 전체 (규칙 + Judge)
uv run python -m eval.run --name smoke --subset 5    # 유형별 균등 샘플
uv run python -m eval.run --name x --no-judge        # 규칙 기반 지표만 (API 키 불필요, 무료)
uv run python -m eval.run --name x --repeat 3        # Judge 일관성 측정
uv run python -m eval.run --name x --no-cache        # Judge 캐시 무시
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from eval.judge import Judge
from eval.report import build_report, write_report
from eval.runner import build_run_config, load_gold, run_all, stratified_subset
from rag.config import settings
from rag.logging_config import setup_logging

logger = logging.getLogger("eval")
LABELS_PATH = Path(__file__).parent / "human_labels.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", default="run", help="리포트 디렉터리 이름 접두어")
    p.add_argument(
        "--subset", type=int, default=None, help="유형별 균등 샘플 n 개만 평가"
    )
    p.add_argument("--no-judge", action="store_true", help="LLM Judge 지표 생략")
    p.add_argument(
        "--repeat", type=int, default=1, help="Judge 반복 횟수 (일관성 측정)"
    )
    p.add_argument("--no-cache", action="store_true", help="Judge 캐시 사용 안 함")
    p.add_argument(
        "--no-answer-relevance",
        action="store_true",
        help="Answer Relevance(보조 지표) 생략",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="리포트 출력 디렉터리 (기본: eval/reports/<name>_<시각>)",
    )
    args = p.parse_args()
    setup_logging()

    gold = load_gold()
    if args.subset:
        gold = stratified_subset(gold, args.subset, settings.seed)
    judge_enabled = not args.no_judge
    config = build_run_config(
        args.name, judge_enabled, args.repeat, args.subset, len(gold)
    )
    logger.info(
        "run=%s items=%d judge=%s repeat=%d model=%s judge_model=%s",
        args.name,
        len(gold),
        judge_enabled,
        args.repeat,
        config.chat_model,
        config.judge_model,
    )

    judges = None
    embeddings = None
    if judge_enabled:
        judges = [
            Judge(use_cache=not args.no_cache, repeat_index=i)
            for i in range(args.repeat)
        ]
        if not args.no_answer_relevance:
            from rag.embeddings import get_embeddings

            embeddings = get_embeddings()

    items = run_all(gold, judges, embeddings)
    report = build_report(config, items, LABELS_PATH)

    out_dir = (
        args.out
        or settings.reports_dir
        / f"{args.name}_{datetime.now(tz=UTC).astimezone().strftime('%Y%m%d-%H%M%S')}"
    )
    write_report(report, out_dir)

    if judges:
        logger.info(
            "judge calls=%d cache_hits=%d",
            sum(j.calls for j in judges),
            sum(j.cache_hits for j in judges),
        )
    logger.info("report -> %s", out_dir / "report.md")
    print()
    print((out_dir / "report.md").read_text(encoding="utf-8").split("## 유형별")[0])


if __name__ == "__main__":
    main()
