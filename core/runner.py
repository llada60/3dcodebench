#!/usr/bin/env python3
"""Run text-to-Blender-code or image-to-Blender-code inference on the
3DProBench eval set.

Per-instance input is read from `eval/data/<instance>/`:
  - text_to_3d :  prompt_<description|instruction>.txt
  - image_to_3d:  images/*.png (multi-view, all images sent together)

The system prompt defaults to `eval/prompt/<task>_system_prompt.txt` and
output goes under `eval/results/<model>[_<task>]/<instance>/<instance>.py`.

Settings are loaded from a YAML config file (see `eval/config/`); CLI
flags override individual fields. Three providers are supported, dispatched
by the `provider` field (default: gemini):
    - gemini    -> google-genai SDK
    - anthropic -> anthropic SDK (Claude)
    - openai    -> openai SDK (GPT-5.x)

Examples:
    python run_inference.py --config config/gemini_3_flash.yaml
    python run_inference.py --config config/claude_sonnet_4_6.yaml --task image_to_3d
    python run_inference.py --config config/gpt_5_5.yaml --instances ArmChair_seed0
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import yaml

from .model_api_price import cost_usd  # noqa: E402
from .inputs import load_user_content, serialize_user_content, strip_code_fence  # noqa: E402
from .providers import build_provider_ctx, call_provider  # noqa: E402
from .feedback import build_render_feedback_content, brief_error  # noqa: E402
from .visual_critique import (  # noqa: E402
    build_critique_user_content,
    parse_critique_response,
    critique_system_prompt,
    has_baseline_renders,
)

CORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CORE_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark" / "categories"
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"
# Kept for back-compat with code below that still references EVAL_ROOT for utils paths.
EVAL_ROOT = CORE_DIR

DEFAULTS = {
    "provider": "gemini",       # gemini | anthropic | openai
    "api_key": None,
    "model": "gemini-3-flash",
    "temperature": 0.7,
    "thinking": "medium",       # minimal | low | medium | high | max | xhigh
                                # Unified across providers. Adapters clamp to
                                # the model's nearest supported tier (e.g. on
                                # Sonnet 4.6, xhigh -> max; on OpenAI, max -> xhigh).
    "max_output_tokens": 16384,
    "task": "text_to_3d",       # text_to_3d | image_to_3d
    "prompt_type": "description",  # description | instruction; only used for text_to_3d
    "max_images": None,         # image_to_3d only: limit to first N sorted images
                                # (None = use all; canonical order 005 < 015 < 025 < 035)
    "image_subdir": "images",   # image_to_3d only: which subdir under data/<inst>/
                                # holds the reference images. "images" = Infinigen
                                # turntable renders (4-view ground truth); set to
                                # "nano_banana_pro" to use the generated reference
                                # produced by scripts/generate_reference_image_nbp.py.
    "include_description": False,  # image_to_3d only: prepend the original text
                                # description to the user message (combined "text +
                                # image" mode). Lets the model use both modalities.
    "api_reference": None,      # Optional path to an auxiliary reference doc that
                                # gets appended to the system prompt (e.g.
                                # eval/prompt/blender_5_api_reference.txt). The doc
                                # is added under a clearly delimited section so the
                                # model knows it's reference material, not a new
                                # instruction.
    "seed": None,               # null = no explicit seed; int = deterministic sampling
                                # (currently only honoured by Gemini)
    "data_dir": str(DEFAULT_DATA_DIR),
    "system_prompt": None,      # null = <prompt_dir>/<task>_system_prompt.txt
    "output_dir": None,         # null = <results>/<model>[_<task>]/
    "instances": None,
    "max_workers": 4,
    "overwrite": False,
    "rpm": None,                # provider RPM cap; null disables RPM throttle
    "tpm": None,                # provider TPM cap; null disables TPM throttle
    "max_sweeps": 3,            # extra full-pass retries after the main run
                                # (each pass only re-attempts instances with no .py)
    "parse_retries": 3,         # in-call resamples if response fails ast.parse
    "glb_export": False,        # if true, spawn Blender to export <inst>/glb/<inst>.glb
                                # immediately after each successful inference
    "blender_path": os.environ.get("BLENDER", "blender"),
    "glb_timeout": 180,         # seconds; per-instance hard cap for the Blender export
    "render_views": False,      # if true, spawn Blender to render <inst>/renders/Image_*.png
                                # immediately after each successful inference. The render_log.json
                                # this writes is exactly what metrics/executability.py reads, so
                                # turning this on means the post-batch render pass can be skipped
                                # entirely and executability falls out for free.
    "render_samples": 64,       # Blender Cycles sample count per view
    "render_resolution": 512,   # square render resolution
    "render_engine": "CYCLES",  # CYCLES | BLENDER_EEVEE
    "render_timeout": 240,      # seconds; per-instance hard cap for the Blender render
    "max_render_retries": 0,    # multi_turn_debug: if >0 and render_views=True, on render failure
                                # (status != OK or n_meshes==0) feed the Blender stderr + previous
                                # code back to the model and ask for a fix. Total LLM calls per
                                # instance bounded by (max_render_retries+1) × parse_retries+1.
                                # Failed attempts archived under <inst>/renders_history/attemptN/.
                                # Default 0 keeps single-turn baseline behavior.
    "max_critique_iterations": 0,  # visual_feedback (self-critique): if >0, replaces the standard
                                # process_one with process_one_visual_feedback. Each instance must have a
                                # baseline <inst>.py + renders/Image_*.png pre-populated in
                                # the output dir (the prep step copies these from a prior run).
                                # Each iteration: model is shown its own code + renders + the
                                # original task; it either says NEEDS_FIX: NO (stop) or outputs
                                # corrected full Python (we re-render and continue). Mutually
                                # exclusive with max_render_retries.
}


def parse_cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Path to a YAML config file under eval/config/.")
    # All overrides default to None so we can tell whether the user passed them.
    p.add_argument("--provider", choices=["gemini", "anthropic", "openai"], default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--thinking", default=None)
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--task", choices=["text_to_3d", "image_to_3d"], default=None)
    p.add_argument("--prompt-type", choices=["description", "instruction"], default=None)
    p.add_argument("--max-images", type=int, default=None,
                   help="image_to_3d only: cap number of input views (1..4 in canonical order).")
    p.add_argument("--image-subdir", default=None,
                   help="image_to_3d only: subdir under data/<inst>/ holding refs "
                        "(default 'images' = Infinigen turntable; e.g. 'nano_banana_pro').")
    p.add_argument("--include-description", dest="include_description",
                   action="store_true", default=None,
                   help="image_to_3d only: prepend the original text description "
                        "to the user message (combined text+image mode).")
    p.add_argument("--api-reference", type=Path, default=None,
                   help="Path to an auxiliary reference doc to append to the system "
                        "prompt (e.g. prompt/blender_5_api_reference.txt).")
    p.add_argument("--seed", type=int, default=None,
                   help="Deterministic sampling seed (Gemini only).")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--system-prompt", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--instances", nargs="*", default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--glb-export", dest="glb_export", action="store_true", default=None,
                   help="Spawn Blender per-instance to export <inst>/glb/<inst>.glb after inference.")
    p.add_argument("--render-views", dest="render_views", action="store_true", default=None,
                   help="Spawn Blender per-instance to render <inst>/renders/Image_*.png "
                        "after inference. Writes render_log.json that executability.py reads, "
                        "so the post-batch render pass becomes unnecessary.")
    p.add_argument("--max-render-retries", type=int, default=None,
                   help="multi_turn_debug: on render failure, feed Blender stderr + previous code "
                        "back to the model up to N times. Requires --render-views. Default 0.")
    p.add_argument("--max-critique-iterations", type=int, default=None,
                   help="visual_feedback (self-critique): on each instance, show the model its previous "
                        "code + renders and let it iterate up to N times. Requires --render-views "
                        "and a pre-populated baseline (prep step copies <inst>.py + renders/ from "
                        "an earlier baseline run). Default 0 (visual_feedback disabled).")
    return p.parse_args()


def resolve_settings(cli):
    """Merge defaults < config file < CLI overrides into a single namespace."""
    settings = dict(DEFAULTS)

    cfg = yaml.safe_load(cli.config.read_text()) or {}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"Unknown config keys in {cli.config}: {sorted(unknown)}")
    # Expand ${ENV_VAR} placeholders in string values (e.g. api_key: ${GEMINI_API_KEY}).
    _envref = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
    for k, v in list(cfg.items()):
        if isinstance(v, str):
            m = _envref.match(v.strip())
            if m:
                cfg[k] = os.environ.get(m.group(1)) or None
    settings.update(cfg)

    for key in DEFAULTS:
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            settings[key] = cli_val

    # API key: explicit value > provider-specific env var.
    provider = settings["provider"]
    env_var = {
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }.get(provider)
    if not env_var:
        raise SystemExit(f"Unknown provider: {provider!r}")
    if not settings["api_key"]:
        settings["api_key"] = os.environ.get(env_var)
    if not settings["api_key"]:
        raise SystemExit(
            f"No API key found for provider={provider!r}. "
            f"Set api_key in the config file or export {env_var}."
        )

    settings["data_dir"] = Path(settings["data_dir"])
    if settings["system_prompt"] is None:
        settings["system_prompt"] = DEFAULT_PROMPT_DIR / f"{settings['task']}_system_prompt.txt"
    settings["system_prompt"] = Path(settings["system_prompt"])
    if settings["output_dir"]:
        settings["output_dir"] = Path(settings["output_dir"])
    if settings["api_reference"]:
        settings["api_reference"] = Path(settings["api_reference"])

    return SimpleNamespace(**settings)


def default_output_dir(settings):
    suffix = "" if settings.task == "text_to_3d" else f"_{settings.task}"
    return DEFAULT_OUTPUT_ROOT / f"{settings.model}{suffix}"


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "thoughts_tokens",
                 "total_tokens", "cache_read_tokens", "cache_creation_tokens")


def _save_thinking_trace(out_dir, attempt_idx, thoughts_text):
    """Persist the model's chain-of-thought for every attempt — success
    and failure alike — so we can audit reasoning patterns later."""
    path = out_dir / f"thinking_attempt{attempt_idx}.txt"
    path.write_text(thoughts_text)


def _export_glb_for_instance(out_dir, name, blender_path, timeout):
    """Spawn Blender to export <out_dir>/glb/<name>.glb. Skips if already
    done. Records {status, error} into the inference log; never raises."""
    script = out_dir / f"{name}.py"
    glb_dir = out_dir / "glb"
    out_glb = glb_dir / f"{name}.glb"
    log_path = glb_dir / "export_log.json"
    if log_path.exists():
        return
    glb_dir.mkdir(parents=True, exist_ok=True)
    exporter = CORE_DIR / "export_glb.py"
    cmd = [blender_path, "--background", "--python", str(exporter), "--",
           "--blender-export",
           "--script",   str(script),
           "--out-glb",  str(out_glb),
           "--log-path", str(log_path)]
    t0 = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log_path.write_text(json.dumps({
            "script": str(script), "status": "ERR_TIMEOUT",
            "error": f"exceeded {timeout}s",
            "latency_s": round(time.time() - t0, 2),
        }, indent=2))


def _render_views_for_instance(out_dir, name, blender_path,
                               samples, resolution, engine, timeout):
    """Spawn Blender to render <out_dir>/renders/Image_{005,015,025,035}.png +
    render_log.json. Skips if render_log.json already exists. The schema written
    here matches what utils/render.py and metrics/executability.py expect, so the
    post-batch render pass can be skipped when this hook is enabled."""
    script = out_dir / f"{name}.py"
    renders_dir = out_dir / "renders"
    log_path = renders_dir / "render_log.json"
    if log_path.exists():
        return
    renders_dir.mkdir(parents=True, exist_ok=True)
    renderer = CORE_DIR / "render.py"
    cmd = [blender_path, "--background", "--python", str(renderer), "--",
           "--blender-render",
           "--script",     str(script),
           "--output-dir", str(renders_dir),
           "--samples",    str(samples),
           "--resolution", str(resolution),
           "--engine",     engine]
    t0 = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log_path.write_text(json.dumps({
            "script":   str(script),
            "status":   "ERR_TIMEOUT",
            "error":    f"Blender subprocess exceeded {timeout}s",
            "n_meshes": 0,
            "n_views_rendered": 0,
            "latency_s": round(time.time() - t0, 2),
        }, indent=2))


def _accumulate_usage(record, usage, dt):
    """Sum latency / tokens across in-call retries so the log reflects
    the full cost of producing this output, not just the last attempt."""
    record["latency_s"] = round((record.get("latency_s") or 0) + dt, 2)
    for k in _TOKEN_FIELDS:
        v = usage.get(k)
        if v is None:
            continue
        record[k] = (record.get(k) or 0) + v


def _archive_failed_render(out_dir, attempt_idx):
    """Move <out_dir>/renders → <out_dir>/renders_history/attempt{N} so the
    next render attempt can write fresh into renders/ (which existing metric
    scripts read). Idempotent — silently no-ops if renders/ is missing."""
    src = out_dir / "renders"
    if not src.exists():
        return
    dst_root = out_dir / "renders_history"
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / f"attempt{attempt_idx}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _revert_after_failed_fix(out_dir, broken_iter_idx, restore_from_attempt):
    """visual_feedback do-no-harm: when an iter's fix produces a broken
    render, undo the damage so renders/ + <inst>.py reflect the previous
    good state. The broken render is archived to
    `renders_history/attempt{N}_failed/` for forensics; the previous good
    render is COPIED back from `renders_history/attempt{R}/` (kept in
    history too, so the archive remains intact)."""
    rh = out_dir / "renders_history"
    src_broken = out_dir / "renders"
    if src_broken.exists():
        dst_failed = rh / f"attempt{broken_iter_idx}_failed"
        if dst_failed.exists():
            shutil.rmtree(dst_failed)
        rh.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_broken), str(dst_failed))
    src_good = rh / f"attempt{restore_from_attempt}"
    if src_good.exists():
        # COPY (not move): keep the archive intact for inspection.
        shutil.copytree(src_good, out_dir / "renders")


def _read_render_log(out_dir):
    """Return parsed render_log.json or a stub if missing/malformed."""
    rl_path = out_dir / "renders" / "render_log.json"
    if not rl_path.exists():
        return {
            "status": "ERR_NOLOG",
            "n_meshes": 0,
            "n_views_rendered": 0,
            "error": "render_log.json was not produced (likely Blender segfault before write)",
        }
    try:
        return json.loads(rl_path.read_text())
    except Exception as e:
        return {
            "status": "ERR_NOLOG",
            "n_meshes": 0,
            "n_views_rendered": 0,
            "error": f"render_log.json unreadable: {type(e).__name__}: {e}",
        }


def _call_and_parse(ctx, settings, system_prompt, user_content, record, out_dir):
    """One render-attempt's worth of LLM calls (with in-call parse-retries).

    Mutates `record` to accumulate token / latency / parse_attempts and to
    persist thinking traces. Returns `(code, parse_err, fatal_exc)`:
      - `code` is the parseable script if successful, else None
      - `parse_err` is the last parse-error message if `code` is None
      - `fatal_exc` is set to a string when the provider itself raised
        (network / quota); the caller should bail immediately.
    """
    code = None
    parse_err = None
    for _ in range(settings.parse_retries + 1):
        record["parse_attempts"] += 1
        try:
            t0 = time.time()
            text, usage = call_provider(ctx, settings, system_prompt, user_content)
            dt = time.time() - t0
        except Exception as e:
            return None, None, f"{type(e).__name__}: {e}"

        thoughts_text = usage.pop("_thoughts_text", "") if usage else ""
        _accumulate_usage(record, usage, dt)
        if thoughts_text:
            _save_thinking_trace(out_dir, record["parse_attempts"], thoughts_text)

        if not text.strip():
            parse_err = "empty response"
            # Empty response = model exhausted thinking budget. Retrying
            # re-burns the same budget for the same outcome.
            break

        candidate = strip_code_fence(text)
        try:
            ast.parse(candidate)
            code = candidate
            parse_err = None
            break
        except SyntaxError as e:
            parse_err = f"SyntaxError: {e}"

    return code, parse_err, None


def process_one(ctx, settings, system_prompt, output_dir, instance_dir):
    name = instance_dir.name
    out_dir  = output_dir / name
    code_path   = out_dir / f"{name}.py"
    log_path    = out_dir / "log.json"
    prompt_path = out_dir / "prompt.txt"

    record = {
        "instance":         name,
        "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider":         settings.provider,
        "model":            settings.model,
        "temperature":      settings.temperature,
        "thinking":         settings.thinking,
        "task":             settings.task,
        "prompt_type":      settings.prompt_type if settings.task == "text_to_3d" else None,
        "max_images":       settings.max_images if settings.task == "image_to_3d" else None,
        "seed":             settings.seed,
        "status":           None,
        "latency_s":        None,
        "code_chars":       None,
        "input_tokens":     None, "output_tokens":   None,
        "thoughts_tokens":  None, "total_tokens":    None,
        "cache_read_tokens": 0,   "cache_creation_tokens": 0,
        "cost_usd":         None,
        "parse_attempts":   0,
        # multi_turn_debug fields. render_attempts = how many render calls were
        # made; render_history = per-attempt status/error; final_render_status
        # = last attempt's render status (or None if --render-views off).
        "render_attempts":  0,
        "render_history":   [],
        "final_render_status": None,
        "error":            None,
    }

    if code_path.exists() and not settings.overwrite:
        record["status"] = "SKIP"
        return name, "SKIP", None, record

    try:
        user_content = load_user_content(instance_dir, settings.task, settings.prompt_type,
                                         max_images=settings.max_images,
                                         image_subdir=settings.image_subdir,
                                         include_description=settings.include_description)
    except FileNotFoundError as e:
        record["status"] = "ERR"
        record["error"] = str(e)
        return name, "ERR", record["error"], record

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(serialize_user_content(user_content) + "\n")

    multi_turn = settings.render_views and settings.max_render_retries > 0
    max_render_attempts = (settings.max_render_retries + 1) if multi_turn else 1

    last_code = None
    current_user_content = user_content

    for render_attempt in range(max_render_attempts):
        code, parse_err, fatal = _call_and_parse(
            ctx, settings, system_prompt, current_user_content, record, out_dir)

        if fatal is not None:
            # Provider itself raised (network / quota). Same handling as
            # before: write what we have, bail with ERR.
            record["status"] = "ERR"
            record["error"] = fatal
            log_path.write_text(json.dumps(record, indent=2) + "\n")
            return name, "ERR", record["error"], record

        if code is None:
            # Parse failed (or empty response).
            if render_attempt == 0:
                # Initial generation produced nothing usable: same terminal
                # behaviour as the original single-turn pipeline.
                if parse_err == "empty response":
                    (out_dir / ".terminal_fail").write_text(
                        "thinking budget exhausted: empty response\n"
                    )
                    record["status"] = "ERR_PARSE"
                    record["error"] = "empty response"
                else:
                    (out_dir / ".terminal_fail").write_text(
                        f"parse retries exhausted ({record['parse_attempts']} "
                        f"attempts): {parse_err}\n"
                    )
                    record["status"] = "ERR_PARSE"
                    record["error"] = (
                        f"unparseable response after {record['parse_attempts']}"
                        f" attempt(s): {parse_err}"
                    )
                _finalise_cost(record, settings)
                log_path.write_text(json.dumps(record, indent=2) + "\n")
                return name, "ERR_PARSE", record["error"], record
            else:
                # Retry attempt failed to parse: keep the prior good code as
                # final and exit the loop. Don't write .terminal_fail (we DO
                # have a valid script on disk from earlier attempts).
                record["render_history"].append({
                    "attempt": render_attempt,
                    "status": "ERR_RETRY_PARSE",
                    "n_meshes": 0,
                    "error_brief": f"retry response unparseable: {parse_err}",
                })
                break

        # Parseable code in hand.
        last_code = code
        code_path.write_text(code)
        if multi_turn:
            (out_dir / f"{name}.attempt{render_attempt}.py").write_text(code)
        record["status"] = "OK"
        record["code_chars"] = len(code)

        if not settings.render_views:
            # No render → no signal to retry on. Single-turn behaviour.
            break

        # Render this attempt. _render_views_for_instance only runs when
        # renders/render_log.json doesn't exist; we ensure that by archiving
        # any prior failed render directory below before the next iteration.
        _render_views_for_instance(out_dir, name,
                                   settings.blender_path,
                                   settings.render_samples,
                                   settings.render_resolution,
                                   settings.render_engine,
                                   settings.render_timeout)
        record["render_attempts"] += 1

        rl = _read_render_log(out_dir)
        status = rl.get("status")
        n_meshes = rl.get("n_meshes", 0) or 0
        record["render_history"].append({
            "attempt": render_attempt,
            "status": status,
            "n_meshes": n_meshes,
            "error_brief": brief_error(rl),
        })
        record["final_render_status"] = status

        succeeded = (status == "OK" and n_meshes > 0)
        if succeeded or render_attempt == max_render_attempts - 1:
            if not succeeded and multi_turn:
                # All multi-turn retries exhausted. Mark .terminal_fail so
                # future sweeps don't waste budget re-running this. Only set
                # in the multi-turn path; single-turn baseline never wrote
                # terminal_fail for render failures.
                (out_dir / ".terminal_fail").write_text(
                    f"render retries exhausted: {record['render_attempts']} "
                    f"attempts, final status={status}\n"
                )
            break

        # Prepare next-attempt feedback content. Archive the failed render
        # so renders/ is empty for the next call (and we keep history).
        current_user_content = build_render_feedback_content(
            original_user_content=user_content,
            prev_code=last_code,
            render_log=rl,
            attempt_num=render_attempt + 2,
            max_attempts=max_render_attempts,
        )
        if multi_turn:
            (out_dir / f"prompt.attempt{render_attempt + 1}.txt").write_text(
                serialize_user_content(current_user_content) + "\n"
            )
        _archive_failed_render(out_dir, render_attempt)

    _finalise_cost(record, settings)
    log_path.write_text(json.dumps(record, indent=2) + "\n")

    if getattr(settings, "glb_export", False):
        _export_glb_for_instance(out_dir, name,
                                 settings.blender_path, settings.glb_timeout)

    cost_s = f"${record['cost_usd']:.5f}" if record.get("cost_usd") is not None else "?"
    extra = f", parse_tries={record['parse_attempts']}" if record["parse_attempts"] > 1 else ""
    if record["render_attempts"] > 0:
        extra += (f", render_tries={record['render_attempts']}"
                  f"→{record['final_render_status']}")
    info = (f"{record['latency_s']}s, {len(last_code) if last_code else 0} chars, "
            f"in={record['input_tokens']}, out={record['output_tokens']}, "
            f"think={record['thoughts_tokens']}, cost={cost_s}{extra}")
    return name, "OK", info, record


def _finalise_cost(record, settings):
    record["cost_usd"] = cost_usd(
        settings.model,
        input_tokens=record.get("input_tokens") or 0,
        output_tokens=record.get("output_tokens") or 0,
        thoughts_tokens=record.get("thoughts_tokens") or 0,
        cache_read_tokens=record.get("cache_read_tokens") or 0,
        cache_creation_tokens=record.get("cache_creation_tokens") or 0,
    )


# ────────────────────────────────────────────────────────────────────────────
# visual_feedback — visual self-critique loop
# ────────────────────────────────────────────────────────────────────────────

def process_one_visual_feedback(ctx, settings, _system_prompt_unused, output_dir, instance_dir):
    """visual_feedback process: critique own renders + iterate.

    Pre-requisites (the visual_feedback prep step is responsible for these):
      - <output_dir>/<inst>/<inst>.py  exists  (baseline code)
      - <output_dir>/<inst>/renders/Image_005|015|025|035.png exists (≥ 1)

    Skips with status:
      SKIP_VISUAL_FEEDBACK_DONE        — already ranvisual_feedback (`.visual_feedback_done` marker present)
      SKIP_NO_BASELINE    — baseline code or renders missing
    """
    name = instance_dir.name
    out_dir = output_dir / name
    code_path = out_dir / f"{name}.py"
    log_path = out_dir / "log.json"
    prompt_path = out_dir / "prompt.txt"

    record = {
        "instance":         name,
        "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider":         settings.provider,
        "model":            settings.model,
        "temperature":      settings.temperature,
        "thinking":         settings.thinking,
        "task":             settings.task,
        "prompt_type":      settings.prompt_type if settings.task == "text_to_3d" else None,
        "max_images":       settings.max_images if settings.task == "image_to_3d" else None,
        "seed":             settings.seed,
        "status":           None,
        "latency_s":        None,
        "code_chars":       None,
        "input_tokens":     None, "output_tokens":   None,
        "thoughts_tokens":  None, "total_tokens":    None,
        "cache_read_tokens": 0,   "cache_creation_tokens": 0,
        "cost_usd":         None,
        "parse_attempts":   0,
        "render_attempts":  0,
        "render_history":   [],
        "final_render_status": None,
        "critique_history": [],   # visual_feedback-specific
        "error":            None,
    }

    # Resumption: if .visual_feedback_done exists, skip (the user can pass --overwrite to redo).
    if (out_dir / ".visual_feedback_done").exists() and not settings.overwrite:
        record["status"] = "SKIP_VISUAL_FEEDBACK_DONE"
        return name, "SKIP_VISUAL_FEEDBACK_DONE", None, record

    if not code_path.exists():
        record["status"] = "SKIP_NO_BASELINE"
        record["error"] = "baseline <inst>.py missing —visual_feedback needs a baseline to critique"
        return name, "SKIP_NO_BASELINE", record["error"], record
    if not has_baseline_renders(out_dir):
        record["status"] = "SKIP_NO_BASELINE"
        record["error"] = "baseline renders missing —visual_feedback needs render PNGs to critique"
        return name, "SKIP_NO_BASELINE", record["error"], record

    try:
        user_content = load_user_content(instance_dir, settings.task, settings.prompt_type,
                                         max_images=settings.max_images,
                                         image_subdir=settings.image_subdir,
                                         include_description=settings.include_description)
    except FileNotFoundError as e:
        record["status"] = "ERR"
        record["error"] = str(e)
        return name, "ERR", record["error"], record

    out_dir.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        prompt_path.write_text(serialize_user_content(user_content) + "\n")

    # Archive baseline code as attempt0 (idempotent).
    baseline_code = code_path.read_text()
    last_code = baseline_code
    attempt0_path = out_dir / f"{name}.attempt0.py"
    if not attempt0_path.exists():
        attempt0_path.write_text(baseline_code)

    task = settings.task
    sys_prompt = critique_system_prompt(task)
    max_iter = settings.max_critique_iterations

    # Track which archive holds the last KNOWN-GOOD render. After iter k's
    # fix renders OK, the previous good render lives at
    # renders_history/attempt{k-1}/ (we just archived it). After a revert,
    # this index doesn't change because we restored from there.
    last_good_archive_idx = None  # set after first successful fix's archive
    final_decision = None
    for iter_idx in range(1, max_iter + 1):
        # The critique sees whatever is currently in renders/ — baseline at
        # iter 1, last good fix's render at iter ≥ 2.
        critique_user = build_critique_user_content(
            user_content, last_code, out_dir / "renders",
            iter_num=iter_idx, max_iter=max_iter,
            task=task,
        )
        (out_dir / f"prompt.vf_iter{iter_idx}.txt").write_text(
            serialize_user_content(critique_user) + "\n"
        )

        record["parse_attempts"] += 1
        try:
            t0 = time.time()
            text, usage = call_provider(ctx, settings, sys_prompt, critique_user)
            dt = time.time() - t0
        except Exception as e:
            record["status"] = "ERR"
            record["error"] = f"{type(e).__name__}: {e}"
            log_path.write_text(json.dumps(record, indent=2) + "\n")
            return name, "ERR", record["error"], record

        thoughts_text = usage.pop("_thoughts_text", "") if usage else ""
        _accumulate_usage(record, usage, dt)
        if thoughts_text:
            _save_thinking_trace(out_dir, record["parse_attempts"], thoughts_text)

        (out_dir / f"critique_response_iter{iter_idx}.txt").write_text(text or "")

        decision, assessment, new_code = parse_critique_response(text or "")
        entry = {
            "iter":       iter_idx,
            "decision":   decision,
            "assessment": (assessment or "")[:600],
            "raw_chars":  len(text or ""),
        }
        record["critique_history"].append(entry)
        final_decision = decision

        if decision == "DONE":
            break

        if decision == "MALFORMED":
            entry["error"] = "model response unparseable; keeping previous code"
            break

        # decision == "FIX" — validate code parses before adopting it.
        try:
            ast.parse(new_code or "")
        except SyntaxError as e:
            entry["error"] = f"FIX code unparseable: {e}; keeping previous code"
            break

        # Archive the render this critique was based on, then prepare to render
        # the new code. Iter 1: archive baseline as attempt0; iter N (>1):
        # archive the prior iter's render as attempt{N-1}.
        archive_idx = iter_idx - 1
        _archive_failed_render(out_dir, archive_idx)

        # Adopt the fix tentatively. <inst>.py points at the new code, but
        # if the render fails we'll revert below.
        prev_good_code = last_code
        prev_good_archive_idx = last_good_archive_idx if last_good_archive_idx is not None else archive_idx
        last_code = new_code
        code_path.write_text(last_code)
        (out_dir / f"{name}.attempt{iter_idx}.py").write_text(last_code)

        _render_views_for_instance(out_dir, name,
                                   settings.blender_path,
                                   settings.render_samples,
                                   settings.render_resolution,
                                   settings.render_engine,
                                   settings.render_timeout)
        record["render_attempts"] += 1
        rl = _read_render_log(out_dir)
        record["render_history"].append({
            "iter":         iter_idx,
            "status":       rl.get("status"),
            "n_meshes":     rl.get("n_meshes", 0) or 0,
            "error_brief":  brief_error(rl),
        })
        render_ok = rl.get("status") == "OK" and (rl.get("n_meshes") or 0) > 0

        if render_ok:
            # Fix accepted — its render is now the new "current" state.
            # Next iter's archive_idx = this iter's idx.
            last_good_archive_idx = archive_idx
            record["final_render_status"] = rl.get("status")
        else:
            # Revert: restore renders/ from the previous good archive, and
            # reset <inst>.py to the previous good code. The broken renders
            # (and any partial PNGs) move to renders_history/attempt{N}_failed/
            # for forensic inspection.
            _revert_after_failed_fix(out_dir, iter_idx, prev_good_archive_idx)
            code_path.write_text(prev_good_code)
            last_code = prev_good_code
            entry["reverted"] = True
            entry.setdefault("note",
                f"post-fix render failed ({rl.get('status')}); reverted to "
                f"previous good code (renders_history/attempt{prev_good_archive_idx}/)")
            # final_render_status stays as the LAST GOOD render's status (OK)
            # — by definition we restored to a good state.
            record["final_render_status"] = "OK_REVERTED"
            break

    # Mark done so reruns of the script SKIP this instance.
    (out_dir / ".visual_feedback_done").write_text(
        f"max_critique_iterations={max_iter}, final_decision={final_decision}\n"
    )

    record["status"] = "OK"
    record["code_chars"] = len(last_code)
    _finalise_cost(record, settings)
    log_path.write_text(json.dumps(record, indent=2) + "\n")

    cost_s = f"${record['cost_usd']:.5f}" if record.get("cost_usd") is not None else "?"
    decisions = "/".join(c["decision"] for c in record["critique_history"]) or "(no calls)"
    extra = f", iters={len(record['critique_history'])}({decisions})"
    if record["render_attempts"] > 0:
        extra += f", re-renders={record['render_attempts']}→{record['final_render_status']}"
    info = (f"{record['latency_s']}s, {len(last_code)} chars, "
            f"in={record['input_tokens']}, out={record['output_tokens']}, "
            f"think={record['thoughts_tokens']}, cost={cost_s}{extra}")
    return name, "OK", info, record


def main():
    cli = parse_cli()
    settings = resolve_settings(cli)

    ctx = build_provider_ctx(settings)
    system_prompt = settings.system_prompt.read_text()
    if settings.api_reference:
        if not settings.api_reference.is_file():
            raise SystemExit(f"--api-reference not found: {settings.api_reference}")
        ref_text = settings.api_reference.read_text()
        system_prompt = (
            system_prompt.rstrip()
            + "\n\n# === Auxiliary Blender 5.0 API reference (read before emitting code) ===\n\n"
            + ref_text.rstrip()
            + "\n"
        )

    instance_dirs = sorted(p for p in settings.data_dir.iterdir()
                           if p.is_dir() and not p.name.startswith(("_", ".")))
    if settings.instances:
        keep = set(settings.instances)
        instance_dirs = [p for p in instance_dirs if p.name in keep]
        if not instance_dirs:
            raise SystemExit(f"No matches for instances {settings.instances}")

    output_dir = settings.output_dir or default_output_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = DEFAULT_LOG_ROOT / settings.model
    log_dir.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc)
    log_path = log_dir / f"{run_started.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_lock = Lock()

    header = {
        "event":          "run_start",
        "timestamp":      run_started.isoformat(timespec="seconds"),
        "provider":       settings.provider,
        "model":          settings.model,
        "temperature":    settings.temperature,
        "thinking":       settings.thinking,
        "task":           settings.task,
        "prompt_type":    settings.prompt_type if settings.task == "text_to_3d" else None,
        "max_images":     settings.max_images if settings.task == "image_to_3d" else None,
        "n_instances":    len(instance_dirs),
        "system_prompt":  str(settings.system_prompt),
        "output_dir":     str(output_dir),
        "config_file":    str(cli.config),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(header) + "\n")

    def log_record(record):
        with log_lock, log_path.open("a") as f:
            f.write(json.dumps({"event": "instance", **record}) + "\n")

    print(f"Config:        {cli.config}")
    print(f"Provider:      {settings.provider}")
    print(f"Model:         {settings.model}")
    print(f"Temperature:   {settings.temperature}")
    print(f"Thinking:      {settings.thinking}")
    print(f"Task:          {settings.task}")
    if settings.task == "text_to_3d":
        print(f"Prompt type:   {settings.prompt_type}")
    if settings.task == "image_to_3d":
        print(f"Max images:    {settings.max_images}")
    print(f"System prompt: {settings.system_prompt}")
    if settings.api_reference:
        print(f"API reference: {settings.api_reference} (+{len(ref_text)} chars appended)")
    print(f"Instances:     {len(instance_dirs)}")
    print(f"Output dir:    {output_dir}")
    print(f"Log file:      {log_path}")
    print(f"Workers:       {settings.max_workers}")
    print(f"Overwrite:     {settings.overwrite}")
    if settings.max_render_retries > 0 and settings.max_critique_iterations > 0:
        raise SystemExit("--max-render-retries (multi_turn_debug) and "
                         "--max-critique-iterations (visual_feedback) are "
                         "mutually exclusive; pick one.")
    if settings.render_views and settings.max_render_retries > 0:
        print(f"Multi-turn:    multi_turn_debug error-feedback retry, max_render_retries={settings.max_render_retries}")
    if settings.max_critique_iterations > 0:
        if not settings.render_views:
            raise SystemExit("--max-critique-iterations (visual_feedback) requires --render-views.")
        print(f"Multi-turn:    visual_feedback (self-critique), max_critique_iterations={settings.max_critique_iterations}")
    print()

    counts = {"OK": 0, "SKIP": 0, "ERR": 0, "ERR_PARSE": 0}
    totals = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    cost_total = 0.0

    def consume(name, status, info, record):
        nonlocal cost_total
        counts[status] = counts.get(status, 0) + 1
        log_record(record)
        for k in totals:
            v = record.get(k)
            if v is not None:
                totals[k] += v
        if record.get("cost_usd") is not None:
            cost_total += record["cost_usd"]
        detail = f"  {info}" if info else ""
        print(f"  [{status:<9}] {name}{detail}")

    handler = process_one_visual_feedback if settings.max_critique_iterations > 0 else process_one

    def run_pass(targets):
        if settings.max_workers > 1:
            with ThreadPoolExecutor(max_workers=settings.max_workers) as ex:
                futures = {
                    ex.submit(handler, ctx, settings, system_prompt,
                              output_dir, d): d
                    for d in targets
                }
                for fut in as_completed(futures):
                    consume(*fut.result())
        else:
            for d in targets:
                consume(*handler(ctx, settings, system_prompt, output_dir, d))

    run_pass(instance_dirs)

    # Auto-sweep: any instance whose .py is still missing gets re-attempted.
    # SKIP'd instances (already have a .py) are never revisited.
    # Instances marked .terminal_fail (e.g. thinking budget exhausted) are
    # also skipped — retrying produces the same empty response.
    for sweep in range(1, settings.max_sweeps + 1):
        missing = [d for d in instance_dirs
                   if not (output_dir / d.name / f"{d.name}.py").exists()
                   and not (output_dir / d.name / ".terminal_fail").exists()]
        if not missing:
            break
        print(f"\n--- Auto-sweep {sweep}/{settings.max_sweeps}: "
              f"retrying {len(missing)} missing output(s) ---\n")
        run_pass(missing)

    duration_s = round((datetime.now(timezone.utc) - run_started).total_seconds(), 1)
    footer = {
        "event":     "run_end",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "ok":        counts["OK"],
        "skip":      counts["SKIP"],
        "err":       counts["ERR"],
        "err_parse": counts.get("ERR_PARSE", 0),
        "sum_input_tokens":    totals["input_tokens"],
        "sum_output_tokens":   totals["output_tokens"],
        "sum_thoughts_tokens": totals["thoughts_tokens"],
        "sum_total_tokens":    totals["total_tokens"],
        "sum_cost_usd":        round(cost_total, 4),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(footer) + "\n")

    print()
    print(f"Done. OK={counts['OK']}  SKIP={counts['SKIP']}  "
          f"ERR={counts['ERR']}  ERR_PARSE={counts.get('ERR_PARSE', 0)}  "
          f"in={totals['input_tokens']}  out={totals['output_tokens']}  "
          f"think={totals['thoughts_tokens']}  total={totals['total_tokens']}  "
          f"cost=${cost_total:.4f}  dur={duration_s}s")
    print(f"Log:  {log_path}")
    if counts["ERR"] or counts.get("ERR_PARSE", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
