"""두 리포트(Before/After) 비교 표 생성.

uv run python -m eval.compare eval/reports/baseline_xxx eval/reports/exp_yyy [--out compare.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "recall_at_k",
    "mrr",
    "context_relevance",
    "faithfulness",
    "correctness",
    "answer_relevance",
)
CONFIG_KEYS = (
    "chat_model",
    "embedding_model",
    "judge_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "git_sha",
    "gold_set_hash",
)


def load(dir_: Path) -> dict:
    return json.loads((dir_ / "report.json").read_text(encoding="utf-8"))


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


def compare(a: dict, b: dict) -> str:
    na, nb = a["config"]["name"], b["config"]["name"]
    lines = [f"# Compare: {na} → {nb}", ""]
    lines += ["## 설정 차이", "", "| 항목 | Before | After |", "|---|---|---|"]
    for k in CONFIG_KEYS:
        va, vb = a["config"].get(k), b["config"].get(k)
        mark = " ⚠️" if va != vb else ""
        lines.append(f"| {k}{mark} | {va} | {vb} |")
    lines += ["", "## 지표", "", f"| 지표 | {na} | {nb} | Δ |", "|---|---|---|---|"]
    for m in METRICS:
        va, vb = a["overall"][m]["mean"], b["overall"][m]["mean"]
        lines.append(f"| {m} | {_f(va)} | {_f(vb)} | {_delta(va, vb)} |")
    for key, label in (
        ("abstain_answer_rate", "답 있는 질의 응답률"),
        ("abstain_abstain_rate", "답 없는 질의 거부율"),
    ):
        va, vb = a.get(key), b.get(key)
        lines.append(
            f"| {label} | {_f(va, True)} | {_f(vb, True)} | {_delta(va, vb, True)} |"
        )

    lines += [
        "",
        "## 유형별 Δ (correctness / recall@k / faithfulness)",
        "",
        "| 유형 | correctness | recall@k | faithfulness |",
        "|---|---|---|---|",
    ]
    for t in sorted(set(a["by_type"]) | set(b["by_type"])):
        cells = []
        for m in ("correctness", "recall_at_k", "faithfulness"):
            va = a["by_type"].get(t, {}).get(m, {}).get("mean")
            vb = b["by_type"].get(t, {}).get(m, {}).get("mean")
            cells.append(f"{_f(va)} → {_f(vb)} ({_delta(va, vb)})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    md = compare(load(args.before), load(args.after))
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
