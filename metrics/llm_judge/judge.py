#!/usr/bin/env python3
"""Run a single (judge_model, mode) sweep over arena pairs.

For each pair in pairs.jsonl, builds a prompt that mimics the arena UI
(four outcomes: a / b / tie / both_bad) and asks a Gemini-family judge to
return a strict JSON verdict + short reasoning. A/B order is randomized
per call to remove position bias; the swap is recorded so we can map back.

Output: one JSON file per pair under
    arena/judge_data/results/<judge_slug>/<mode>/<vote_id>.json

Usage:
    python metrics/llm_judge/judge.py \
        --judge gemini-3-flash-preview --mode image
    python metrics/llm_judge/judge.py \
        --judge gemini-3.1-pro-preview --mode code --limit 50
"""
from __future__ import annotations

import argparse
import os
import json
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_ROOT  = Path(os.environ.get("EVAL_ROOT", "."))
JUDGE_DIR  = EVAL_ROOT / "judge"
PROMPTS_DIR = JUDGE_DIR / "prompts"
PAIRS_FILE  = EVAL_ROOT / "arena" / "judge_data" / "pairs.jsonl"
RESULTS_DIR = EVAL_ROOT / "arena" / "judge_data" / "results"

# --- Model setup -----------------------------------------------------------
# All four judges live in the same Google AI Studio API. The same key from
# eval/config/gemini_*.yaml works for Gemma too.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Judges supported. Slug = arena canonical, model = google-genai model id.
JUDGES = {
    "gemini-3.1-pro-preview":        "gemini-3.1-pro-preview",
    "gemini-3-flash-preview":        "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    "gemma-4-31b-it":                "gemma-4-31b-it",
}

# Models that don't support thinking_config. Gemma is non-thinking.
NO_THINKING = {"gemma-4-31b-it"}


def load_prompts(variant: str = "") -> dict[str, str]:
    suffix = f"_{variant}" if variant else ""
    return {
        "image": (PROMPTS_DIR / f"image_judge{suffix}.txt").read_text(),
        "code":  (PROMPTS_DIR / f"code_judge{suffix}.txt").read_text(),
    }


def load_pairs(limit: int | None = None,
               vote_ids_file: str | None = None) -> list[dict]:
    pairs = [json.loads(line) for line in PAIRS_FILE.read_text().splitlines()
             if line.strip()]
    if vote_ids_file:
        keep = {int(x) for x in Path(vote_ids_file).read_text().split()
                if x.strip()}
        pairs = [p for p in pairs if p["vote_id"] in keep]
    if limit:
        pairs = pairs[:limit]
    return pairs


def read_image_bytes(path: str) -> tuple[bytes, str]:
    return Path(path).read_bytes(), "image/png"


def read_code(path: str, max_chars: int = 60000) -> str:
    """Cap source to max_chars to keep prompts within token budget."""
    txt = Path(path).read_text(errors="replace")
    if len(txt) > max_chars:
        head = txt[: max_chars - 200]
        return head + f"\n\n# [... truncated, original was {len(txt)} chars ...]\n"
    return txt


# --- Prompt building -------------------------------------------------------

def build_image_user_content(types, system_prompt: str, pair: dict, swap: bool):
    """Compose Gemini Parts list for image-mode judging."""
    a, b = ("b", "a") if swap else ("a", "b")  # what to label as "A" / "B"
    label_a, label_b = pair[a], pair[b]
    parts = []
    parts.append(types.Part.from_text(text="--- ORIGINAL PROMPT ---\n" + pair["prompt_text"].strip()))
    if pair["modality"] == "image" and pair.get("reference_images"):
        parts.append(types.Part.from_text(
            text=f"\n--- REFERENCE IMAGES ({len(pair['reference_images'])}) ---"))
        for p in pair["reference_images"]:
            data, mime = read_image_bytes(p)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(types.Part.from_text(
        text=f"\n--- SYSTEM A: {len(label_a['render_paths'])} renders ---"))
    for p in label_a["render_paths"]:
        data, mime = read_image_bytes(p)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(types.Part.from_text(
        text=f"\n--- SYSTEM B: {len(label_b['render_paths'])} renders ---"))
    for p in label_b["render_paths"]:
        data, mime = read_image_bytes(p)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(types.Part.from_text(
        text="\nReturn the JSON verdict now."))
    return parts


def build_code_user_content(types, system_prompt: str, pair: dict, swap: bool):
    a, b = ("b", "a") if swap else ("a", "b")
    label_a, label_b = pair[a], pair[b]
    body = (
        "--- ORIGINAL PROMPT ---\n" + pair["prompt_text"].strip() + "\n\n"
        f"--- SYSTEM A: Blender Python source ---\n```python\n"
        + read_code(label_a["code_path"]) + "\n```\n\n"
        f"--- SYSTEM B: Blender Python source ---\n```python\n"
        + read_code(label_b["code_path"]) + "\n```\n\n"
        "Return the JSON verdict now."
    )
    return [types.Part.from_text(text=body)]


# --- Verdict parsing -------------------------------------------------------

VERDICTS = {"a", "b", "tie", "both_bad"}


def parse_verdict(text: str) -> tuple[str | None, str]:
    """Return (winner_in_swapped_space, reasoning). winner=None on parse fail."""
    if not text:
        return None, ""
    # Try to find a JSON object anywhere in the response.
    # First strip ```json ... ``` fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.MULTILINE)
    # Find the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, ""
    blob = cleaned[start:end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # try a more permissive cleanup
        blob2 = re.sub(r",(\s*[}\]])", r"\1", blob)
        try:
            obj = json.loads(blob2)
        except Exception:
            return None, ""
    winner = str(obj.get("winner", "")).strip().lower()
    reasoning = str(obj.get("reasoning", "")).strip()
    if winner not in VERDICTS:
        return None, reasoning
    return winner, reasoning


def unswap(winner: str, swap: bool) -> str:
    """Convert verdict from swapped-label space back to original a/b."""
    if not swap or winner in ("tie", "both_bad"):
        return winner
    return "b" if winner == "a" else "a"


# --- Per-pair worker -------------------------------------------------------

def judge_one(client, types, judge_slug: str, mode: str,
              system_prompt: str, pair: dict, out_dir: Path,
              max_tokens: int, retries: int = 2) -> dict:
    out_path = out_dir / f"{pair['vote_id']}.json"
    if out_path.exists():
        return {"vote_id": pair["vote_id"], "status": "skip"}

    rng = random.Random(pair["vote_id"])
    swap = rng.random() < 0.5  # deterministic per vote_id
    if mode == "image":
        contents = build_image_user_content(types, system_prompt, pair, swap)
    else:
        contents = build_code_user_content(types, system_prompt, pair, swap)

    cfg_kwargs = dict(
        system_instruction=system_prompt,
        temperature=0.0,
        max_output_tokens=max_tokens,
    )
    if judge_slug not in NO_THINKING:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level="low", include_thoughts=False)

    last_err = None
    raw_text = ""
    usage = {}
    quota_attempts = 0
    transient_attempts = 0
    QUOTA_CAP = 10
    QUOTA_BACKOFF = [30, 60, 120, 180, 240, 300, 300, 300, 300, 300]
    while True:
        try:
            resp = client.models.generate_content(
                model=JUDGES[judge_slug],
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            raw_text = resp.text or ""
            u = getattr(resp, "usage_metadata", None)
            if u:
                usage = {
                    "input_tokens":  getattr(u, "prompt_token_count", None),
                    "output_tokens": getattr(u, "candidates_token_count", None),
                    "total_tokens":  getattr(u, "total_token_count", None),
                }
            break
        except Exception as e:
            err_repr = repr(e)
            last_err = err_repr
            is_quota = ("429" in err_repr or "RESOURCE_EXHAUSTED" in err_repr
                        or "quota" in err_repr.lower())
            if is_quota:
                if quota_attempts >= QUOTA_CAP:
                    # Don't persist a quota error file -> let next run retry.
                    return {"vote_id": pair["vote_id"], "status": "quota_skip"}
                time.sleep(QUOTA_BACKOFF[quota_attempts])
                quota_attempts += 1
                continue
            if transient_attempts >= retries:
                # Persistent non-quota error -> persist so we don't retry forever.
                out_path.write_text(json.dumps({
                    "vote_id": pair["vote_id"], "judge": judge_slug,
                    "mode": mode, "error": err_repr, "status": "error",
                }) + "\n")
                return {"vote_id": pair["vote_id"], "status": "error"}
            time.sleep(min(60, 2 ** transient_attempts * 5))
            transient_attempts += 1

    winner_swapped, reasoning = parse_verdict(raw_text)
    if winner_swapped is None:
        result = {
            "vote_id":      pair["vote_id"],
            "judge":        judge_slug,
            "mode":         mode,
            "swap":         swap,
            "status":       "parse_error",
            "raw_response": raw_text,
            "usage":        usage,
        }
    else:
        result = {
            "vote_id":      pair["vote_id"],
            "pair_id":      pair["pair_id"],
            "prompt_slug":  pair["prompt_slug"],
            "modality":     pair["modality"],
            "model_a":      pair["model_a_slug"],
            "model_b":      pair["model_b_slug"],
            "human_winner": pair["human_winner"],
            "judge":        judge_slug,
            "mode":         mode,
            "swap":         swap,
            "judge_winner": unswap(winner_swapped, swap),
            "judge_winner_swapped": winner_swapped,
            "reasoning":    reasoning,
            "raw_response": raw_text,
            "usage":        usage,
            "status":       "ok",
        }
    out_path.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    return {"vote_id": pair["vote_id"], "status": result["status"]}


# --- Driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--mode",  required=True, choices=["image", "code"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of pairs (for sanity tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="per-call output tokens")
    ap.add_argument("--variant", default="",
                    help="prompt variant suffix (e.g. v1 → loads "
                         "image_judge_v1.txt). Empty = baseline.")
    ap.add_argument("--vote-ids-file", default=None,
                    help="path with one vote_id per line; only judge those.")
    args = ap.parse_args()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("google-genai not installed in this Python env")
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompts = load_prompts(variant=args.variant)
    system_prompt = prompts[args.mode]
    pairs = load_pairs(limit=args.limit, vote_ids_file=args.vote_ids_file)
    judge_subdir = f"{args.judge}__{args.variant}" if args.variant else args.judge
    out_dir = RESULTS_DIR / judge_subdir / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Judge:  {args.judge}")
    print(f"Mode:   {args.mode}")
    print(f"Pairs:  {len(pairs)}")
    print(f"Out:    {out_dir}")
    print(f"Workers:{args.workers}")
    print()

    t0 = time.time()
    counts = {"ok": 0, "parse_error": 0, "error": 0, "skip": 0, "quota_skip": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_id = {
            ex.submit(judge_one, client, types, args.judge, args.mode,
                      system_prompt, p, out_dir, args.max_tokens): p["vote_id"]
            for p in pairs
        }
        for fut in as_completed(fut_to_id):
            try:
                r = fut.result()
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            except Exception as e:
                counts["error"] = counts.get("error", 0) + 1
                print(f"  worker crashed on vote {fut_to_id[fut]}: {e}")
                traceback.print_exc()
            done += 1
            if done % 25 == 0 or done == len(pairs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(pairs) - done) / max(rate, 1e-6)
                print(f"  [{done}/{len(pairs)}] {counts}  "
                      f"{rate:.1f}/s  eta {eta/60:.1f}m")

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. Counts: {counts}")


if __name__ == "__main__":
    main()
