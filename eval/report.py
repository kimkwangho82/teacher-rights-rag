"""집계 및 리포트(JSON + Markdown) 생성. Judge 일관성(반복)과 Human alignment 계산 포함."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from eval.metrics.abstain import abstain_matrix
from eval.schema import ItemResult, MetricSummary, Report, RunConfig

METRICS = {
    "recall_at_k": lambda it: it.recall_at_k,
    "mrr": lambda it: it.mrr,
    "context_relevance": lambda it: it.judge.context_relevance,
    "faithfulness": lambda it: it.judge.faithfulness,
    "correctness": lambda it: it.judge.correctness,
    "answer_relevance": lambda it: it.judge.answer_relevance,
}
RETRIEVAL = ("recall_at_k", "mrr", "context_relevance")
GENERATION = ("faithfulness", "correctness", "answer_relevance")

# 이진화 임계값 (human alignment 비교용)
FAITHFUL_THRESHOLD = 0.8
CORRECT_THRESHOLD = 0.5


def summarize(items: list[ItemResult], getter) -> MetricSummary:
    vals = [v for v in (getter(it) for it in items) if v is not None]
    return MetricSummary(mean=sum(vals) / len(vals) if vals else None, n=len(vals))


def consistency(items: list[ItemResult]) -> dict[str, float] | None:
    """반복 실행 간 Judge 일관성: 지표별 항목 표준편차 평균 + 이진 판정 일치율."""
    multi = [it for it in items if len(it.judge_repeats) > 1]
    if not multi:
        return None
    out: dict[str, float] = {
        "repeats": float(len(multi[0].judge_repeats)),
        "items": float(len(multi)),
    }
    for name in ("faithfulness", "correctness", "context_relevance"):
        stds, agree = [], []
        for it in multi:
            vals = [
                getattr(r, name)
                for r in it.judge_repeats
                if getattr(r, name) is not None
            ]
            if len(vals) > 1:
                stds.append(statistics.pstdev(vals))
                thr = (
                    FAITHFUL_THRESHOLD if name == "faithfulness" else CORRECT_THRESHOLD
                )
                bins = [v >= thr for v in vals]
                agree.append(1.0 if all(bins) or not any(bins) else 0.0)
        if stds:
            out[f"{name}_mean_std"] = sum(stds) / len(stds)
            out[f"{name}_binary_agreement"] = sum(agree) / len(agree)
    return out


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def human_alignment(
    items: list[ItemResult], labels_path: Path
) -> dict[str, float | int] | None:
    if not labels_path.exists():
        return None
    labels = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            labels[d["id"]] = d
    by_id = {it.id: it for it in items}
    out: dict[str, float | int] = {}
    for name, thr, key in (
        ("faithfulness", FAITHFUL_THRESHOLD, "faithful"),
        ("correctness", CORRECT_THRESHOLD, "correct"),
    ):
        human, judge = [], []
        for id_, lab in labels.items():
            it = by_id.get(id_)
            if it is None or lab.get(key) is None:
                continue
            score = getattr(it.judge, name)
            if score is None:
                continue
            human.append(bool(lab[key]))
            judge.append(score >= thr)
        if human:
            out[f"{name}_n"] = len(human)
            out[f"{name}_agreement"] = sum(
                1 for h, j in zip(human, judge) if h == j
            ) / len(human)
            k = cohen_kappa(human, judge)
            if k is not None:
                out[f"{name}_kappa"] = k
    return out or None


def build_report(
    config: RunConfig, items: list[ItemResult], labels_path: Path | None = None
) -> Report:
    overall = {name: summarize(items, g) for name, g in METRICS.items()}
    by_type: dict[str, dict[str, MetricSummary]] = {}
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for it in items:
        groups[it.type].append(it)
    for t, its in sorted(groups.items()):
        by_type[t] = {name: summarize(its, g) for name, g in METRICS.items()}
        by_type[t]["n_items"] = MetricSummary(mean=float(len(its)), n=len(its))
        by_type[t]["abstain_rate"] = MetricSummary(
            mean=sum(1 for i in its if i.status == "insufficient") / len(its),
            n=len(its),
        )

    m = abstain_matrix([(it.answerable, it.status) for it in items])
    return Report(
        config=config,
        overall=overall,
        by_type=by_type,
        abstain=m,
        abstain_answer_rate=m.answer_rate,
        abstain_abstain_rate=m.abstain_rate,
        consistency=consistency(items),
        human_alignment=human_alignment(items, labels_path) if labels_path else None,
        items=items,
    )


def _f(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def to_markdown(r: Report) -> str:
    c = r.config
    lines: list[str] = []
    lines += [f"# Eval Report: {c.name}", ""]
    lines += ["## Run config", "", "| 항목 | 값 |", "|---|---|"]
    for k in (
        "started_at",
        "git_sha",
        "chat_model",
        "embedding_model",
        "judge_model",
        "seed",
        "temperature",
        "chunk_size",
        "chunk_overlap",
        "min_chunk_chars",
        "top_k",
        "similarity_threshold",
        "gold_set_hash",
        "gold_set_size",
        "index_chunks",
        "repeat",
        "subset",
    ):
        lines.append(f"| {k} | {getattr(c, k)} |")
    lines.append(
        f"| prompt_hashes | {', '.join(f'{k}={v}' for k, v in c.prompt_hashes.items())} |"
    )
    lines.append(
        f"| packages | {', '.join(f'{k}={v}' for k, v in c.packages.items())} |"
    )
    lines.append("")

    lines += ["## 검색 단계 (Retrieval)", "", "| 지표 | 평균 | n |", "|---|---|---|"]
    for k in RETRIEVAL:
        s = r.overall[k]
        lines.append(f"| {k} | {_f(s.mean)} | {s.n} |")
    lines += [
        "",
        "## 생성 단계 (Generation)",
        "",
        "| 지표 | 평균 | n |",
        "|---|---|---|",
    ]
    for k in GENERATION:
        s = r.overall[k]
        lines.append(f"| {k} | {_f(s.mean)} | {s.n} |")

    m = r.abstain
    lines += [
        "",
        "## Abstain",
        "",
        f"- 답 있는 질의 응답률: **{_f(r.abstain_answer_rate, pct=True)}** ({m.answerable_answered}/{m.answerable_answered + m.answerable_abstained})",
        f"- 답 없는 질의 거부율: **{_f(r.abstain_abstain_rate, pct=True)}** ({m.unanswerable_abstained}/{m.unanswerable_answered + m.unanswerable_abstained})",
        "",
        "| | 시스템: 답변 | 시스템: 거부 |",
        "|---|---|---|",
        f"| Gold: 답 있음 | {m.answerable_answered} | {m.answerable_abstained} (과잉 거부) |",
        f"| Gold: 답 없음 | {m.unanswerable_answered} (hallucination 위험) | {m.unanswerable_abstained} |",
    ]

    lines += [
        "",
        "## 유형별",
        "",
        "| 유형 | n | recall@k | mrr | ctx_rel | faith | correct | ans_rel | 거부율 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, d in r.by_type.items():
        lines.append(
            f"| {t} | {int(d['n_items'].mean or 0)} | {_f(d['recall_at_k'].mean)} | {_f(d['mrr'].mean)} | {_f(d['context_relevance'].mean)} | "
            f"{_f(d['faithfulness'].mean)} | {_f(d['correctness'].mean)} | {_f(d['answer_relevance'].mean)} | {_f(d['abstain_rate'].mean, pct=True)} |"
        )

    if r.consistency:
        lines += [
            "",
            f"## Judge 일관성 (반복 {int(r.consistency['repeats'])}회, {int(r.consistency['items'])}개 항목)",
            "",
            "| 지표 | 항목별 표준편차 평균 | 이진 판정 일치율 |",
            "|---|---|---|",
        ]
        for k in ("faithfulness", "correctness", "context_relevance"):
            if f"{k}_mean_std" in r.consistency:
                lines.append(
                    f"| {k} | {_f(r.consistency[f'{k}_mean_std'])} | {_f(r.consistency[f'{k}_binary_agreement'], pct=True)} |"
                )

    if r.human_alignment:
        h = r.human_alignment
        lines += [
            "",
            "## Human alignment (수동 라벨 vs Judge)",
            "",
            "| 지표 | n | 일치율 | Cohen's κ |",
            "|---|---|---|---|",
        ]
        for k in ("faithfulness", "correctness"):
            if f"{k}_n" in h:
                lines.append(
                    f"| {k} (임계값 {FAITHFUL_THRESHOLD if k == 'faithfulness' else CORRECT_THRESHOLD}) | {int(h[f'{k}_n'])} | {_f(h.get(f'{k}_agreement'), pct=True)} | {_f(h.get(f'{k}_kappa'))} |"
                )

    lines += ["", "## 주의가 필요한 항목", ""]
    flagged = []
    for it in r.items:
        reasons = []
        if it.answerable and it.status == "insufficient":
            reasons.append("과잉 거부")
        if not it.answerable and it.status == "answered":
            reasons.append("답 없는 질의에 답변")
        if it.recall_at_k is not None and it.recall_at_k < 0.5:
            reasons.append(f"recall {it.recall_at_k:.2f}")
        if it.judge.faithfulness is not None and it.judge.faithfulness < 0.8:
            bad = [s.text for s in it.judge.statements if s.supported is False][:2]
            reasons.append(
                f"faith {it.judge.faithfulness:.2f}"
                + (f" (미지지: {' / '.join(bad)})" if bad else "")
            )
        if it.judge.correctness is not None and it.judge.correctness < 0.5:
            failed = [c.text for c in it.judge.criteria if c.passed is False][:2]
            reasons.append(
                f"correct {it.judge.correctness:.2f}"
                + (f" (실패: {' / '.join(failed)})" if failed else "")
            )
        if it.judge.errors:
            reasons.append("judge 오류")
        if reasons:
            flagged.append((it, reasons))
    if not flagged:
        lines.append("없음")
    for it, reasons in flagged:
        lines.append(
            f"- **{it.id}** ({it.type}) {it.question}  \n  → {'; '.join(reasons)}"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: Report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        report.model_dump_json(indent=2, exclude={"items"}), encoding="utf-8"
    )
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for it in report.items:
            f.write(it.model_dump_json() + "\n")
    (out_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return out_dir
