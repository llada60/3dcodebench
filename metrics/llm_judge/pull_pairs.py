#!/usr/bin/env python3
"""Build a normalized pairs.jsonl from arena votes + local renders/code.

For each human vote (pulled live from Supabase), resolve:
  - prompt slug + text + reference images (image modality)
  - model_a / model_b slug
  - per-side code path + 4 render paths (via cells.csv variants)

Writes to /lab/yipeng/infinigen/eval/arena/judge_data/pairs.jsonl.

Run:
    python /lab/yipeng/infinigen/eval/judge/pull_pairs.py
    python /lab/yipeng/infinigen/eval/judge/pull_pairs.py --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

EVAL_ROOT  = Path("/lab/yipeng/infinigen/eval")
ARENA_DIR  = EVAL_ROOT / "arena"
DATA_ROOT  = EVAL_ROOT / "data"
RESULTS_ROOT = EVAL_ROOT / "results"
CELLS_CSV  = ARENA_DIR / "pipeline" / "cells.csv"
ENV_FILE   = ARENA_DIR / ".env"
OUT_DIR    = ARENA_DIR / "judge_data"
OUT_FILE   = OUT_DIR / "pairs.jsonl"
MODELS_SNAPSHOT_GLOB = "models_elo_snapshot_*.json"
VIEW_FRAMES = (5, 15, 25, 35)


def load_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def models_id_to_slug() -> dict[int, str]:
    backups = sorted((ARENA_DIR / "db" / "backups").glob(MODELS_SNAPSHOT_GLOB))
    if not backups:
        sys.exit(f"No models snapshot found under {ARENA_DIR/'db'/'backups'}")
    snap = json.loads(backups[-1].read_text())
    # Two known shapes: bare list (old) or {"exported_at": ..., "models": [...]} (new)
    rows = snap if isinstance(snap, list) else snap.get("models", [])
    return {row["id"]: row["slug"] for row in rows}


def fetch_supabase_table(client, table: str, select: str = "*") -> list[dict]:
    """Page through every row (Supabase caps single fetch at 1000)."""
    PAGE = 1000
    rows, start = [], 0
    while True:
        chunk = (client.table(table)
                 .select(select)
                 .range(start, start + PAGE - 1)
                 .execute().data)
        rows.extend(chunk)
        if len(chunk) < PAGE:
            break
        start += PAGE
    return rows


def build_cell_index() -> dict[tuple[str, str], dict]:
    """(model_slug, prompt_slug) -> {code_path, renders[4], status, variant}."""
    out = {}
    for r in csv.DictReader(CELLS_CSV.open()):
        if r["status"] != "ok":
            continue
        glb_rel = r["glb_relpath"]
        # glb_rel: 'results/<sub>/<variant>/<seed>/glb/<seed>.glb'
        m = re.match(r"results/(?P<sub>[^/]+(?:/[^/]+)?)/(?P<variant>[^/]+)/"
                     r"(?P<seed>[^/]+)/glb/(?P<name>[^/]+)\.glb$", glb_rel)
        if not m:
            continue
        sub, variant, seed, name = m["sub"], m["variant"], m["seed"], m["name"]
        cell_dir = EVAL_ROOT / "results" / sub / variant / seed
        code_path = cell_dir / f"{name}.py"
        renders = [cell_dir / "renders" / f"Image_{n:03d}.png" for n in VIEW_FRAMES]
        if not code_path.exists() or not all(p.exists() for p in renders):
            continue
        out[(r["model"], r["prompt"])] = {
            "code_path":    str(code_path),
            "render_paths": [str(p) for p in renders],
            "variant":      r["variant"],
        }
    return out


def reference_image_paths(prompt_slug: str) -> list[str]:
    """For image modality only. prompt_slug = 'Foo_seed0__image'."""
    factory = prompt_slug.rsplit("__", 1)[0]
    img_dir = DATA_ROOT / factory / "images"
    paths = [img_dir / f"Image_{n:03d}.png" for n in VIEW_FRAMES]
    return [str(p) for p in paths if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of votes processed (for sampling)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling (only when --limit set)")
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    os.environ["SUPABASE_URL"] = env["SUPABASE_URL"]
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = env["SUPABASE_SERVICE_ROLE_KEY"]

    from supabase import create_client
    sb = create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])

    print("Loading prompts from Supabase ...")
    prompts = fetch_supabase_table(sb, "prompts", "id, slug, text, modality")
    pid_to_prompt = {p["id"]: p for p in prompts}
    print(f"  prompts: {len(prompts)}")

    print("Loading votes from Supabase ...")
    votes = fetch_supabase_table(sb, "votes",
        "id, pair_id, prompt_id, model_a_id, model_b_id, modality, winner")
    print(f"  votes:   {len(votes)}")

    mid_to_slug = models_id_to_slug()
    cell_idx = build_cell_index()
    print(f"  cells (ok with code+renders): {len(cell_idx)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out, skipped = [], {"missing_prompt": 0, "missing_model": 0,
                        "missing_cell_a": 0, "missing_cell_b": 0,
                        "missing_refs": 0}
    for v in votes:
        pinfo = pid_to_prompt.get(v["prompt_id"])
        if pinfo is None:
            skipped["missing_prompt"] += 1
            continue
        slug_a = mid_to_slug.get(v["model_a_id"])
        slug_b = mid_to_slug.get(v["model_b_id"])
        if not slug_a or not slug_b:
            skipped["missing_model"] += 1
            continue
        cell_a = cell_idx.get((slug_a, pinfo["slug"]))
        cell_b = cell_idx.get((slug_b, pinfo["slug"]))
        if cell_a is None:
            skipped["missing_cell_a"] += 1
            continue
        if cell_b is None:
            skipped["missing_cell_b"] += 1
            continue
        rec = {
            "vote_id":      v["id"],
            "pair_id":      v["pair_id"],
            "prompt_id":    v["prompt_id"],
            "prompt_slug":  pinfo["slug"],
            "modality":     v["modality"],
            "prompt_text":  pinfo["text"],
            "reference_images": [],
            "model_a_slug": slug_a,
            "model_b_slug": slug_b,
            "human_winner": v["winner"],
            "a": cell_a,
            "b": cell_b,
        }
        if v["modality"] == "image":
            refs = reference_image_paths(pinfo["slug"])
            if not refs:
                skipped["missing_refs"] += 1
                continue
            rec["reference_images"] = refs
        out.append(rec)

    if args.limit and len(out) > args.limit:
        import random
        rng = random.Random(args.seed)
        out = rng.sample(out, args.limit)
        out.sort(key=lambda r: r["vote_id"])

    with OUT_FILE.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(out)} pairs -> {OUT_FILE}")
    print(f"Skipped: {skipped}")
    by_winner = {}
    for r in out:
        by_winner[r["human_winner"]] = by_winner.get(r["human_winner"], 0) + 1
    by_modality = {}
    for r in out:
        by_modality[r["modality"]] = by_modality.get(r["modality"], 0) + 1
    print(f"Winner distribution: {by_winner}")
    print(f"Modality distribution: {by_modality}")


if __name__ == "__main__":
    main()
