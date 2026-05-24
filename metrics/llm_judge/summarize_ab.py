#!/usr/bin/env python3
"""A/B-only summary: drop tie + both_bad on BOTH sides, then compute
accuracy + Pearson + Cohen's kappa on the remaining decisive pairs.

Usage:
    python /lab/yipeng/infinigen/eval/judge/summarize_ab.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path("/lab/yipeng/infinigen/eval/arena/judge_data/results")
SUMMARY_DIR = Path("/lab/yipeng/infinigen/eval/arena/judge_data/summary")


def load_results(judge_dir: Path) -> list[dict]:
    out = []
    for f in judge_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.append(r)
    return out


def cohen_kappa(rows: list[dict]) -> float:
    """Binary kappa on (human, judge) ∈ {a, b}."""
    n = len(rows)
    if n == 0:
        return float("nan")
    po = sum(1 for r in rows if r["human_winner"] == r["judge_winner"]) / n
    h_a = sum(1 for r in rows if r["human_winner"] == "a") / n
    j_a = sum(1 for r in rows if r["judge_winner"] == "a") / n
    pe = h_a * j_a + (1 - h_a) * (1 - j_a)
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def pearson_phi(rows: list[dict]) -> float:
    """Pearson on a=0, b=1 == phi coefficient on a 2x2 table."""
    n = len(rows)
    if n == 0:
        return float("nan")
    h = [0 if r["human_winner"] == "a" else 1 for r in rows]
    j = [0 if r["judge_winner"] == "a" else 1 for r in rows]
    mh = sum(h) / n
    mj = sum(j) / n
    num = sum((h[i] - mh) * (j[i] - mj) for i in range(n))
    dh = math.sqrt(sum((x - mh) ** 2 for x in h))
    dj = math.sqrt(sum((x - mj) ** 2 for x in j))
    if dh == 0 or dj == 0:
        return float("nan")
    return num / (dh * dj)


def summarize(rows_all: list[dict]) -> dict:
    # 1. A/B-only on BOTH sides
    rows = [r for r in rows_all
            if r["human_winner"] in ("a", "b") and r["judge_winner"] in ("a", "b")]
    n_total       = len(rows_all)
    n_human_decisive = sum(1 for r in rows_all if r["human_winner"] in ("a", "b"))
    n_ab          = len(rows)
    correct       = sum(1 for r in rows if r["human_winner"] == r["judge_winner"])
    acc           = correct / n_ab if n_ab else 0.0
    coverage      = n_ab / n_human_decisive if n_human_decisive else 0.0
    # 2x2 confusion: [human_a -> judge_a/b, human_b -> judge_a/b]
    conf = {"a": {"a": 0, "b": 0}, "b": {"a": 0, "b": 0}}
    for r in rows:
        conf[r["human_winner"]][r["judge_winner"]] += 1
    by_mod = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        m = r["modality"]
        by_mod[m]["n"] += 1
        if r["human_winner"] == r["judge_winner"]:
            by_mod[m]["correct"] += 1
    by_modality = {m: {"n": d["n"], "correct": d["correct"],
                       "acc": (d["correct"] / d["n"] if d["n"] else 0.0)}
                   for m, d in by_mod.items()}
    return {
        "n_total":          n_total,
        "n_human_decisive": n_human_decisive,
        "n_ab":             n_ab,
        "coverage":         coverage,           # judge-decisive / human-decisive
        "correct":          correct,
        "accuracy":         acc,
        "kappa":            cohen_kappa(rows),
        "phi":              pearson_phi(rows),
        "confusion":        conf,
        "by_modality":      by_modality,
    }


def fmt_pct(x: float) -> str:
    if x != x:  # nan
        return "  nan "
    return f"{x*100:5.1f}%"


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for judge_dir in sorted(RESULTS_DIR.glob("*")):
        if not judge_dir.is_dir():
            continue
        for mode_dir in sorted(judge_dir.glob("*")):
            if not mode_dir.is_dir():
                continue
            judge, mode = judge_dir.name, mode_dir.name
            data = load_results(mode_dir)
            if not data:
                continue
            s = summarize(data)
            (SUMMARY_DIR / f"{judge}__{mode}__ab.json").write_text(
                json.dumps(s, indent=2))
            rows.append((judge, mode, s))

    # Print + save table
    lines = []
    lines.append(f"{'judge':<33} {'mode':<6} "
                 f"{'n_ab':>5} {'acc':>7} {'kappa':>7} {'phi':>7} "
                 f"{'cov':>6}  {'image':>7} {'text':>7}  conf [hA,hB]x[jA,jB]")
    lines.append("-" * 130)
    for judge, mode, s in rows:
        c = s["confusion"]
        conf_str = f"[{c['a']['a']:>3}/{c['a']['b']:>3}]/[{c['b']['a']:>3}/{c['b']['b']:>3}]"
        img = s["by_modality"].get("image", {})
        txt = s["by_modality"].get("text",  {})
        lines.append(f"{judge:<33} {mode:<6} "
                     f"{s['n_ab']:>5} "
                     f"{fmt_pct(s['accuracy']):>7} "
                     f"{s['kappa']:>+7.3f} "
                     f"{s['phi']:>+7.3f} "
                     f"{fmt_pct(s['coverage']):>6}  "
                     f"{fmt_pct(img.get('acc',float('nan'))):>7} "
                     f"{fmt_pct(txt.get('acc',float('nan'))):>7}  "
                     f"{conf_str}")
    out = "\n".join(lines)
    (SUMMARY_DIR / "table_ab.txt").write_text(out + "\n")
    print(out)
    print(f"\nWrote {SUMMARY_DIR}/table_ab.txt + per-judge *__ab.json")


if __name__ == "__main__":
    main()
