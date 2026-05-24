#!/usr/bin/env python3
"""Aggregate judge outputs vs human votes.

For each (judge, mode) directory under arena/judge_data/results/, computes:
  - n              : number of OK rows
  - agreement      : exact-match rate (judge_winner == human_winner)
  - winner_dist    : distribution of judge verdicts
  - confusion[h][j]: counts of (human, judge) pairs
  - by_modality    : agreement split by modality
  - position_bias  : fraction of "a"/"b" verdicts before unswap (sanity)

Writes per-judge summaries to arena/judge_data/summary/<judge>__<mode>.json
and a top-level arena/judge_data/summary/overall.json + table.txt.

Usage:
    python /lab/yipeng/infinigen/eval/judge/summarize.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path("/lab/yipeng/infinigen/eval/arena/judge_data/results")
SUMMARY_DIR = Path("/lab/yipeng/infinigen/eval/arena/judge_data/summary")
LABELS = ["a", "b", "tie", "both_bad"]


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


def summarize_one(rows: list[dict]) -> dict:
    n = len(rows)
    agree = sum(1 for r in rows if r["judge_winner"] == r["human_winner"])
    winner_dist = Counter(r["judge_winner"] for r in rows)
    human_dist  = Counter(r["human_winner"] for r in rows)
    conf = {h: {j: 0 for j in LABELS} for h in LABELS}
    for r in rows:
        conf[r["human_winner"]][r["judge_winner"]] += 1
    # Modality split
    by_mod = defaultdict(lambda: {"n": 0, "agree": 0})
    for r in rows:
        m = r["modality"]
        by_mod[m]["n"] += 1
        if r["judge_winner"] == r["human_winner"]:
            by_mod[m]["agree"] += 1
    by_modality = {m: {"n": d["n"], "agree": d["agree"],
                       "rate": (d["agree"] / d["n"] if d["n"] else 0.0)}
                   for m, d in by_mod.items()}
    # Position bias: among the swap-space verdict ("judge_winner_swapped"),
    # what fraction is "a" vs "b"? If unbiased and the swap is 50/50, this
    # should match the underlying winner distribution.
    swap_verdicts = Counter(r.get("judge_winner_swapped") for r in rows)
    # Restricted-decisive agreement: only on rows where human picked a or b.
    decisive = [r for r in rows if r["human_winner"] in ("a", "b")]
    dec_agree = sum(1 for r in decisive
                    if r["judge_winner"] == r["human_winner"])
    return {
        "n":               n,
        "agree":           agree,
        "agreement_rate":  agree / n if n else 0.0,
        "decisive_n":      len(decisive),
        "decisive_agree":  dec_agree,
        "decisive_rate":   dec_agree / len(decisive) if decisive else 0.0,
        "winner_dist":     dict(winner_dist),
        "human_dist":      dict(human_dist),
        "confusion":       conf,
        "by_modality":     by_modality,
        "position_bias_swap_dist": dict(swap_verdicts),
    }


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    overall = {}
    table_rows = []
    for judge_dir in sorted(RESULTS_DIR.glob("*")):
        if not judge_dir.is_dir():
            continue
        for mode_dir in sorted(judge_dir.glob("*")):
            if not mode_dir.is_dir():
                continue
            judge, mode = judge_dir.name, mode_dir.name
            rows = load_results(mode_dir)
            if not rows:
                continue
            s = summarize_one(rows)
            key = f"{judge}__{mode}"
            (SUMMARY_DIR / f"{key}.json").write_text(
                json.dumps(s, indent=2))
            overall[key] = {
                "n": s["n"],
                "agreement_rate": s["agreement_rate"],
                "decisive_rate":  s["decisive_rate"],
            }
            table_rows.append((judge, mode, s))

    (SUMMARY_DIR / "overall.json").write_text(json.dumps(overall, indent=2))

    # Pretty table
    lines = []
    lines.append(f"{'judge':<33} {'mode':<6} {'n':>5} "
                 f"{'agree':>7} {'decisive':>8}  modality(rate)  judge_dist")
    lines.append("-" * 130)
    for judge, mode, s in table_rows:
        mod_str = ", ".join(f"{m}:{fmt_pct(d['rate']).strip()}"
                            for m, d in sorted(s["by_modality"].items()))
        wd = s["winner_dist"]
        wd_str = "  ".join(f"{lbl}:{wd.get(lbl,0)}" for lbl in LABELS)
        lines.append(f"{judge:<33} {mode:<6} {s['n']:>5} "
                     f"{fmt_pct(s['agreement_rate']):>7} "
                     f"{fmt_pct(s['decisive_rate']):>8}  "
                     f"{mod_str:<30}  {wd_str}")
    table = "\n".join(lines)
    (SUMMARY_DIR / "table.txt").write_text(table + "\n")
    print(table)
    print(f"\nWrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
