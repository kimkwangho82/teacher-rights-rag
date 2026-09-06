"""리포트 비교 (Before/After). 한쪽에 여러 리포트를 주면 평균과 min~max 범위를 표시한다.

uv run python -m eval.compare --before eval/reports/baseline-* --after eval/reports/exp1-* [--out compare.md]
uv run python -m eval.compare eval/reports/baseline_x eval/reports/exp_y     # 1:1 도 가능
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "recall_at_k",
    "recall_at_3",
    "mrr",
    "context_relevance",
    "faithfulness",
    "correctness",
    "answer_relevance",
)
RATES = (
    ("abstain_answer_rate", "답 있는 질의 응답률"),
    ("abstain_abstain_rate", "답 없는 질의 거부율"),
)
CONFIG_KEYS = (
    "chat_model",
    "embedding_model",
    "judge_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "retrieval_mode",
    "bm25_tokenizer",
    "prompt_mode",
    "git_sha",
    "gold_set_hash",
)


def load(dir_: Path) -> dict:
    return json.loads((dir_ / "report.json").read_text(encoding="utf-8"))


def load_items(dir_: Path) -> dict[str, dict]:
    p = dir_ / "items.jsonl"
    if not p.exists():
        return {}
    return {
        json.loads(l)["id"]: json.loads(l)
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _f(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _delta(a, b, pct=False):
    if a is None or b is None:
        return "-"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}pp" if pct else f"{sign}{d:.3f}"


def _cell(vals, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = _mean(vals)
    if len(vals) == 1:
        return _f(m, pct)
    return f"{_f(m, pct)} ({_f(min(vals), pct)}~{_f(max(vals), pct)})"


def overall_metric(rep: dict, m: str):
    return rep["overall"].get(m, {}).get("mean")


def by_type_metric(rep: dict, t: str, m: str):
    return rep["by_type"].get(t, {}).get(m, {}).get("mean")


def compare(
    befores: list[dict],
    afters: list[dict],
    before_items: list[dict],
    after_items: list[dict],
) -> str:
    na = ", ".join(r["config"]["name"] for r in befores)
    nb = ", ".join(r["config"]["name"] for r in afters)
    lines = [
        f"# Compare: [{na}] → [{nb}]",
        "",
        f"Before {len(befores)}회 / After {len(afters)}회 평균. 괄호는 실행 간 min~max.",
        "",
    ]

    lines += ["## 설정 차이", "", "| 항목 | Before | After |", "|---|---|---|"]
    for k in CONFIG_KEYS:
        va = {str(r["config"].get(k)) for r in befores}
        vb = {str(r["config"].get(k)) for r in afters}
        mark = " ⚠️" if va != vb else ""
        lines.append(f"| {k}{mark} | {'/'.join(sorted(va))} | {'/'.join(sorted(vb))} |")

    lines += [
        "",
        "## 지표",
        "",
        "| 지표 | Before | After | Δ(평균) |",
        "|---|---|---|---|",
    ]
    for m in METRICS:
        a = [overall_metric(r, m) for r in befores]
        b = [overall_metric(r, m) for r in afters]
        lines.append(
            f"| {m} | {_cell(a)} | {_cell(b)} | {_delta(_mean(a), _mean(b))} |"
        )
    for key, label in RATES:
        a = [r.get(key) for r in befores]
        b = [r.get(key) for r in afters]
        lines.append(
            f"| {label} | {_cell(a, True)} | {_cell(b, True)} | {_delta(_mean(a), _mean(b), True)} |"
        )

    lines += [
        "",
        "## Abstain 혼동행렬 (합계)",
        "",
        "| | Before 답변/거부 | After 답변/거부 |",
        "|---|---|---|",
    ]

    def _sum(reps, k):
        return sum(r["abstain"][k] for r in reps)

    lines.append(
        f"| Gold: 답 있음 | {_sum(befores, 'answerable_answered')} / {_sum(befores, 'answerable_abstained')} | {_sum(afters, 'answerable_answered')} / {_sum(afters, 'answerable_abstained')} |"
    )
    lines.append(
        f"| Gold: 답 없음 | {_sum(befores, 'unanswerable_answered')} / {_sum(befores, 'unanswerable_abstained')} | {_sum(afters, 'unanswerable_answered')} / {_sum(afters, 'unanswerable_abstained')} |"
    )

    lines += [
        "",
        "## 유형별 (correctness / recall@k / faithfulness)",
        "",
        "| 유형 | correctness | recall@k | faithfulness |",
        "|---|---|---|---|",
    ]
    types = sorted({t for r in befores + afters for t in r["by_type"]})
    for t in types:
        cells = []
        for m in ("correctness", "recall_at_k", "faithfulness"):
            a = _mean([by_type_metric(r, t, m) for r in befores])
            b = _mean([by_type_metric(r, t, m) for r in afters])
            cells.append(f"{_f(a)} → {_f(b)} ({_delta(a, b)})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    if any(before_items) and any(after_items):
        lines += ["", "## 문항별 변화", ""]
        ids = sorted({i for d in before_items + after_items for i in d})
        changes = []
        for id_ in ids:
            bs = [d[id_] for d in before_items if id_ in d]
            as_ = [d[id_] for d in after_items if id_ in d]
            if not bs or not as_:
                continue
            b_status = "/".join(x["status"][:3] for x in bs)
            a_status = "/".join(x["status"][:3] for x in as_)
            b_corr = _mean([x["judge"].get("correctness") for x in bs])
            a_corr = _mean([x["judge"].get("correctness") for x in as_])
            b_rec = _mean([x.get("recall_at_k") for x in bs])
            a_rec = _mean([x.get("recall_at_k") for x in as_])
            status_changed = {x["status"] for x in bs} != {x["status"] for x in as_}
            corr_changed = (
                b_corr is not None
                and a_corr is not None
                and abs(a_corr - b_corr) >= 0.34
            )
            rec_changed = (
                b_rec is not None and a_rec is not None and abs(a_rec - b_rec) >= 0.34
            )
            if status_changed or corr_changed or rec_changed:
                q = bs[0]["question"]
                q = q if len(q) <= 40 else q[:40] + "…"
                changes.append(
                    f"| {id_} | {bs[0]['type']} | {q} | {b_status} → {a_status} | {_f(b_rec)} → {_f(a_rec)} | {_f(b_corr)} → {_f(a_corr)} |"
                )
        if changes:
            lines += [
                "| id | 유형 | 질문 | status | recall@k | correctness |",
                "|---|---|---|---|---|---|",
                *changes,
            ]
        else:
            lines.append("status·recall·correctness 가 유의미하게 바뀐 문항 없음")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("positional", nargs="*", type=Path, help="before after (1:1 비교)")
    p.add_argument("--before", nargs="+", type=Path, default=[])
    p.add_argument("--after", nargs="+", type=Path, default=[])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    befores, afters = list(args.before), list(args.after)
    if args.positional:
        if len(args.positional) != 2:
            p.error("positional 인자는 before after 두 개여야 합니다")
        befores, afters = [args.positional[0]], [args.positional[1]]
    if not befores or not afters:
        p.error("--before 와 --after 를 지정하세요")
    md = compare(
        [load(d) for d in befores],
        [load(d) for d in afters],
        [load_items(d) for d in befores],
        [load_items(d) for d in afters],
    )
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
