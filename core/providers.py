"""SDK adapters for Gemini / Anthropic / OpenAI.

Public surface:
    build_provider_ctx(settings) -> dict
        Instantiate the SDK client (and any helper modules) for the chosen
        provider.

    call_provider(ctx, settings, system_prompt, user_content) -> (text, usage)
        Dispatch to the right adapter, with shared retry-on-transient logic.
        `user_content` is the provider-neutral shape produced by
        utils.inputs.load_user_content (str | list[part]).

`usage` keys are normalized across providers:
    input_tokens, output_tokens, thoughts_tokens, total_tokens,
    cache_read_tokens, cache_creation_tokens

Thinking control: the unified `settings.thinking` field accepts the
official Gemini-3 enum {minimal, low, medium, high}. Each provider
adapter translates this string to its native form:
- Gemini 3.x: passed straight to ThinkingConfig(thinking_level=...).
- Anthropic 4.x: mapped to output_config.effort (minimal -> disable).
- OpenAI GPT-5.x: passed to reasoning_effort (minimal supported natively).
"""

import base64
import collections
import threading
import time

# Unified thinking enum (ordered low -> high). "minimal" = effectively no
# thinking (disabled / smallest tier). "max" / "xhigh" are top tiers that
# only some providers expose; adapters clamp to the closest supported tier
# when a model doesn't accept the requested level.
THINKING_LEVELS = ("minimal", "low", "medium", "high", "max", "xhigh")

# Gemini 3.x thinking_level supports {minimal, low, medium, high} only.
# max / xhigh have no native equivalent -> clamp to high.
GEMINI_LEVEL = {
    "minimal": "minimal",
    "low":     "low",
    "medium":  "medium",
    "high":    "high",
    "max":     "high",
    "xhigh":   "high",
}

# Anthropic 4.x thinking surface (verified empirically against the API):
#
#   Haiku 4.5  -> only `thinking={enabled, budget_tokens}`; rejects
#                 `output_config.effort` entirely. Manual budget only.
#   Sonnet 4.6 -> supports both `enabled+budget_tokens` (deprecated) AND
#                 `adaptive + output_config.effort`. Effort levels:
#                 low / medium / high / max  (xhigh -> 400).
#   Opus 4.7   -> ONLY `adaptive`; `enabled+budget_tokens` -> 400.
#                 Effort levels: low / medium / high / max / xhigh.
#
# Mapping policy: Haiku uses the budget path; everything else uses adaptive
# + effort. Sonnet 4.6 doesn't accept xhigh — call_anthropic clamps that
# down to max for Sonnet only. minimal -> None disables thinking.
ANTHROPIC_EFFORT = {
    "minimal": None,                     # disable thinking entirely
    "low":     "low",
    "medium":  "medium",
    "high":    "high",
    "max":     "max",
    "xhigh":   "xhigh",                  # Opus 4.7 only; clamped to max for Sonnet
}
ANTHROPIC_THINKING_BUDGET = {
    "low":     4000,
    "medium":  16000,
    "high":    32000,
    "max":     48000,
    "xhigh":   56000,                    # capped per max_output_tokens at call time
}

# OpenAI gpt-5.x reasoning_effort: {none, low, medium, high, xhigh}.
# - "minimal" is gpt-5 only; gpt-5.4/5.5 reject it -> remap to "low".
# - "max" is not a valid OpenAI tier; the closest top tier is "xhigh".
OPENAI_EFFORT = {
    "minimal": "low",
    "low":     "low",
    "medium":  "medium",
    "high":    "high",
    "max":     "xhigh",
    "xhigh":   "xhigh",
}

# Fatal billing/quota errors — DO NOT retry, raise immediately. OpenAI's
# `insufficient_quota` is a 429 but it means "the account is out of credit",
# not "wait and retry". Without this branch, every worker burns ~5h
# (QUOTA_RETRY_CAP × QUOTA_BACKOFF_S[-1]) before giving up.
FATAL_QUOTA_MARKERS = (
    "insufficient_quota",          # OpenAI: out of credit
    "billing_hard_limit_reached",  # OpenAI: hit hard cap
    "credit_balance_too_low",      # Anthropic: out of credit
)

# Quota / availability errors that ARE worth retrying (provider throttling
# or transient unavailability). Don't count against the retry budget.
QUOTA_MARKERS = (
    "429", "resource_exhausted",
    "503", "unavailable", "service_unavailable",
    "rate_limit", "rate-limit", "overloaded", "quota",
    "usage limit",   # codex CLI: subscription window cap ("You've hit your
                     # usage limit ... try again at <time>") — resets on its
                     # own, so waiting it out beats failing the instance
)
QUOTA_BACKOFF_S = (10, 30, 60, 120, 300)  # caps at 300s after 5th attempt
QUOTA_RETRY_CAP = 60                       # absolute upper bound (~hours)

# Other transient errors (infra blips) burn one slot from RETRY_BACKOFF_S.
OTHER_TRANSIENT_MARKERS = (
    "500", "internal", "deadline_exceeded", "timeout",
)
RETRY_BACKOFF_S = (5, 15, 45, 120)


def _is_fatal_quota(exc):
    msg = str(exc).lower()
    return any(m in msg for m in FATAL_QUOTA_MARKERS)


def _is_quota(exc):
    if _is_fatal_quota(exc):
        return False                       # fatal handled separately, not as retryable
    msg = str(exc).lower()
    return any(m in msg for m in QUOTA_MARKERS)


def _is_other_transient(exc):
    msg = str(exc).lower()
    return any(m in msg for m in OTHER_TRANSIENT_MARKERS)


def _is_transient(exc):
    return _is_quota(exc) or _is_other_transient(exc)


class RateLimiter:
    """Thread-safe sliding-window RPM + TPM leaky bucket.

    `acquire(est_tokens)` blocks until a fresh request would not push the
    last-60s request count past ``rpm`` or the last-60s token sum past
    ``tpm``. It returns a mutable handle; after the call returns, pass
    the actual ``total_tokens`` to ``reconcile(handle, actual)`` so the
    bucket reflects truth (failed calls should reconcile to ``0``).
    """

    def __init__(self, rpm=None, tpm=None, default_tokens=25_000):
        self.rpm = rpm or None
        self.tpm = tpm or None
        self.default_tokens = default_tokens
        self.lock = threading.Lock()
        self.req_times = collections.deque()      # request timestamps
        self.tok_events = collections.deque()     # [timestamp, tokens]

    def acquire(self, est_tokens=None):
        if not self.rpm and not self.tpm:
            return None
        est = est_tokens if est_tokens is not None else self.default_tokens
        while True:
            with self.lock:
                now = time.monotonic()
                cutoff = now - 60
                while self.req_times and self.req_times[0] < cutoff:
                    self.req_times.popleft()
                while self.tok_events and self.tok_events[0][0] < cutoff:
                    self.tok_events.popleft()
                rpm_wait = 0
                if self.rpm and len(self.req_times) >= self.rpm:
                    rpm_wait = self.req_times[0] + 60 - now
                tok_used = sum(t for _, t in self.tok_events)
                tpm_wait = 0
                if self.tpm and tok_used + est > self.tpm:
                    tpm_wait = self.tok_events[0][0] + 60 - now
                wait = max(rpm_wait, tpm_wait)
                if wait <= 0:
                    self.req_times.append(now)
                    handle = [now, est]
                    self.tok_events.append(handle)
                    return handle
            time.sleep(min(wait + 0.05, 5.0))

    def reconcile(self, handle, actual_tokens):
        if handle is None or actual_tokens is None:
            return
        with self.lock:
            handle[1] = actual_tokens


# ---- Per-provider content translation ------------------------------------
# `user_content` is either a str (text-only) or a list of dicts
#   {"type": "text",  "text": str}
#   {"type": "image", "mime": str, "data": bytes, "name": str}

def _gemini_contents(types, user_content):
    if isinstance(user_content, str):
        return user_content
    parts = []
    for p in user_content:
        if p["type"] == "text":
            parts.append(types.Part.from_text(text=p["text"]))
        elif p["type"] == "image":
            parts.append(types.Part.from_bytes(data=p["data"], mime_type=p["mime"]))
        else:
            raise ValueError(f"unknown part type: {p['type']!r}")
    return parts


def _anthropic_message_content(user_content):
    if isinstance(user_content, str):
        return user_content
    blocks = []
    for p in user_content:
        if p["type"] == "text":
            blocks.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p["mime"],
                    "data": base64.b64encode(p["data"]).decode("ascii"),
                },
            })
        else:
            raise ValueError(f"unknown part type: {p['type']!r}")
    return blocks


def _openai_message_content(user_content):
    if isinstance(user_content, str):
        return user_content
    blocks = []
    for p in user_content:
        if p["type"] == "text":
            blocks.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            url = f"data:{p['mime']};base64,{base64.b64encode(p['data']).decode('ascii')}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError(f"unknown part type: {p['type']!r}")
    return blocks


def _openai_responses_content(user_content):
    """Same provider-neutral input but for OpenAI's /v1/responses endpoint.
    Differs from chat.completions: text part is `input_text`, image is
    `input_image` (image_url is a string URL, not a nested dict)."""
    if isinstance(user_content, str):
        return user_content   # string short-circuit accepted as `input`
    blocks = []
    for p in user_content:
        if p["type"] == "text":
            blocks.append({"type": "input_text", "text": p["text"]})
        elif p["type"] == "image":
            url = f"data:{p['mime']};base64,{base64.b64encode(p['data']).decode('ascii')}"
            blocks.append({"type": "input_image", "image_url": url})
        else:
            raise ValueError(f"unknown part type: {p['type']!r}")
    return blocks


# ---- Provider adapters ---------------------------------------------------

def call_gemini(ctx, settings, system_prompt, user_content):
    client, types = ctx["client"], ctx["types"]
    gemini_level = GEMINI_LEVEL.get(settings.thinking)
    if gemini_level is None:
        raise SystemExit(
            f"thinking must be one of {THINKING_LEVELS}; "
            f"got {settings.thinking!r}"
        )
    config_kwargs = dict(
        system_instruction=system_prompt,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )
    if getattr(settings, "seed", None) is not None:
        config_kwargs["seed"] = settings.seed
    # Gemini 2.5 Pro auto-thinks but rejects an explicit thinking_level.
    if not settings.model.startswith("gemini-2.5"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=gemini_level,
            include_thoughts=True,        # so we can extract thinking text on failure
        )
    resp = client.models.generate_content(
        model=settings.model,
        contents=_gemini_contents(types, user_content),
        config=types.GenerateContentConfig(**config_kwargs),
    )
    u = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens":    getattr(u, "prompt_token_count", None),
        "output_tokens":   getattr(u, "candidates_token_count", None),
        "thoughts_tokens": getattr(u, "thoughts_token_count", None),
        "total_tokens":    getattr(u, "total_token_count", None),
    } if u else {}
    # Extract thinking content for caller (saved selectively to keep disk small)
    thoughts_text = ""
    try:
        for part in resp.candidates[0].content.parts:
            if getattr(part, "thought", False) and part.text:
                thoughts_text += part.text
    except Exception:
        pass
    if thoughts_text:
        usage["_thoughts_text"] = thoughts_text
    return resp.text or "", usage


def call_anthropic(ctx, settings, system_prompt, user_content):
    client = ctx["client"]
    effort = ANTHROPIC_EFFORT.get(settings.thinking, "medium")

    kwargs = {
        "model":      settings.model,
        "max_tokens": settings.max_output_tokens,
        # Cache the system prompt across instances (free if prefix < min,
        # ~10x cheaper if it caches). Anthropic min is 1024-2048 tokens.
        "system": [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": _anthropic_message_content(user_content),
        }],
    }
    if effort is None:
        kwargs["thinking"] = {"type": "disabled"}
    elif "haiku" in settings.model:
        # Haiku 4.5: only enabled + budget_tokens (no effort param).
        budget = min(ANTHROPIC_THINKING_BUDGET[effort],
                     settings.max_output_tokens - 1024)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        # Sonnet 4.6 / Opus 4.5+: adaptive + output_config.effort.
        # Sonnet 4.6 rejects xhigh (Opus-only); clamp to max for Sonnet.
        if effort == "xhigh" and "opus" not in settings.model:
            effort = "max"
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    # Opus 4.7 removed sampling params. Skip temperature there.
    # Also: when thinking is enabled/adaptive, Anthropic only accepts
    # temperature=1 (any other value 400s). Just omit the param in that
    # case and let the API default to 1.
    if (
        "opus-4-7" not in settings.model
        and effort is None
        and settings.temperature is not None
    ):
        kwargs["temperature"] = settings.temperature

    # Anthropic SDK refuses non-streaming for requests that *might* exceed
    # 10 minutes (driven by max_tokens). With max_output_tokens=65536 +
    # thinking enabled this kicks in, so stream and accumulate.
    with client.messages.stream(**kwargs) as stream:
        for _ in stream.text_stream:
            pass
        resp = stream.get_final_message()
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    u = getattr(resp, "usage", None)
    if u is None:
        return text, {}
    base_in   = getattr(u, "input_tokens", 0) or 0
    cache_r   = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_w   = getattr(u, "cache_creation_input_tokens", 0) or 0
    out_tok   = getattr(u, "output_tokens", 0) or 0
    return text, {
        # input_tokens = uncached input only (so cost calc can apply
        # the discounted cached_input rate to cache_read_tokens).
        "input_tokens":          base_in,
        "output_tokens":         out_tok,
        "thoughts_tokens":       None,   # Anthropic does not split thinking tokens out
        "total_tokens":          base_in + cache_r + cache_w + out_tok,
        "cache_read_tokens":     cache_r,
        "cache_creation_tokens": cache_w,
    }


def _is_responses_only(model):
    """Pro models (gpt-5-pro, gpt-5.4-pro, gpt-5.5-pro, o1-pro) and the
    GPT-5 family are only available via /v1/responses, not chat.completions."""
    return "-pro" in model


# pro models reject low / minimal / max — only medium / high / xhigh work.
# Map any unsupported request up to the nearest supported tier.
OPENAI_PRO_EFFORT = {
    "minimal": "medium",
    "low":     "medium",
    "medium":  "medium",
    "high":    "high",
    "max":     "xhigh",
    "xhigh":   "xhigh",
}


def call_openai_responses(ctx, settings, system_prompt, user_content):
    """OpenAI /v1/responses adapter — for pro/o-series models that aren't
    exposed on /v1/chat/completions. Mirrors call_openai's return shape."""
    client = ctx["client"]
    effort = OPENAI_PRO_EFFORT.get(settings.thinking, "medium")

    user_payload = _openai_responses_content(user_content)
    if isinstance(user_payload, list):
        user_input = [{"role": "user", "content": user_payload}]
    else:
        user_input = user_payload

    kwargs = {
        "model":             settings.model,
        "instructions":      system_prompt,
        "input":             user_input,
        "reasoning":         {"effort": effort},
        "max_output_tokens": settings.max_output_tokens,
    }
    # pro models don't accept temperature; omit it (defaults to 1).

    # Pro models routinely exceed the OpenAI SDK's 10-minute non-streaming
    # read-timeout on complex prompts (same constraint as the Anthropic
    # SDK's streaming requirement for long requests). Use the streaming
    # context manager so the SDK keeps the connection alive on each event.
    with client.responses.stream(**kwargs) as stream:
        for _ in stream:
            pass
        resp = stream.get_final_response()
    text = resp.output_text or ""

    u = getattr(resp, "usage", None)
    if u is None:
        return text, {}
    in_tok = getattr(u, "input_tokens", 0) or 0
    out_tok = getattr(u, "output_tokens", 0) or 0
    cached = 0
    in_details = getattr(u, "input_tokens_details", None)
    if in_details is not None:
        cached = getattr(in_details, "cached_tokens", 0) or 0
    reasoning = None
    out_details = getattr(u, "output_tokens_details", None)
    if out_details is not None:
        reasoning = getattr(out_details, "reasoning_tokens", None)
    visible = (out_tok - (reasoning or 0)) if reasoning else out_tok
    return text, {
        # Match call_openai's shape: input excludes cached, "output_tokens"
        # is visible only, reasoning lives in "thoughts_tokens".
        "input_tokens":      in_tok - cached,
        "output_tokens":     visible,
        "thoughts_tokens":   reasoning,
        "total_tokens":      getattr(u, "total_tokens", in_tok + out_tok),
        "cache_read_tokens": cached,
    }


def call_openai(ctx, settings, system_prompt, user_content):
    # Pro models only live on /v1/responses; route them to that adapter.
    if _is_responses_only(settings.model):
        return call_openai_responses(ctx, settings, system_prompt, user_content)

    client = ctx["client"]
    effort = OPENAI_EFFORT.get(settings.thinking, "medium")

    # GPT-5.x and o-series are reasoning models: use reasoning_effort, no temp.
    is_reasoning = settings.model.startswith(("gpt-5", "o1", "o3", "o4"))

    kwargs = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _openai_message_content(user_content)},
        ],
        "max_completion_tokens": settings.max_output_tokens,
    }
    if is_reasoning:
        kwargs["reasoning_effort"] = effort
    elif settings.temperature is not None:
        kwargs["temperature"] = settings.temperature

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""

    u = getattr(resp, "usage", None)
    if u is None:
        return text, {}
    reasoning = None
    cached = 0
    out_details = getattr(u, "completion_tokens_details", None)
    if out_details is not None:
        reasoning = getattr(out_details, "reasoning_tokens", None)
    in_details = getattr(u, "prompt_tokens_details", None)
    if in_details is not None:
        cached = getattr(in_details, "cached_tokens", 0) or 0

    prompt_tok = getattr(u, "prompt_tokens", 0) or 0
    completion = getattr(u, "completion_tokens", 0) or 0
    visible    = (completion - (reasoning or 0)) if reasoning else completion
    return text, {
        # OpenAI's prompt_tokens already includes cached_tokens — split them
        # so cost calc uses cached_input rate for the cached portion.
        "input_tokens":      prompt_tok - cached,
        "output_tokens":     visible,
        "thoughts_tokens":   reasoning,
        "total_tokens":      getattr(u, "total_tokens", prompt_tok + completion),
        "cache_read_tokens": cached,
    }


# ---- Claude Code CLI adapter ---------------------------------------------
# Drives the locally-installed `claude` CLI (`-p` print mode) instead of the
# HTTP SDK, so evaluation runs against a logged-in Pro/Max subscription with
# no API key. Modeled on TaskSolver/tasksolver/claude_code.py.

import json as _json
import os as _os
import subprocess as _subprocess
from glob import glob as _glob

# Labels like "claude-code-sonnet-4-6" mean "run via the Claude Code CLI using
# Sonnet 4.6". The CLI's --model only accepts bare family aliases
# (sonnet/opus/fable) or full model ids (claude-sonnet-4-6), so strip the
# "claude-code-" prefix down to the real id passed to --model.
_CLI_MODEL_ALIASES = {
    "claude-code-sonnet-4-6": "claude-sonnet-4-6",
    "claude-code-opus-4-7":   "claude-opus-4-7",
    "claude-code-opus-4-8":   "claude-opus-4-8",
    "claude-code-fable-5":    "claude-fable-5",
    "claude-code-haiku-4-5":  "claude-haiku-4-5",
}


def _resolve_cli_model(model):
    """Map a 'claude-code-*' label to the real --model id the CLI accepts.
    Plain ids (sonnet, claude-sonnet-4-6, …) pass through unchanged. The
    sentinel 'claude-code' (no family) -> None = CLI default model."""
    if model in (None, "claude-code"):
        return None
    if model in _CLI_MODEL_ALIASES:
        return _CLI_MODEL_ALIASES[model]
    if model.startswith("claude-code-"):
        return "claude-" + model[len("claude-code-"):]
    return model


def _claude_command():
    """Resolve the `claude` binary. Prefer a Homebrew cask (macOS); fall back
    to whatever is on PATH (Linux/dev installs)."""
    for path in reversed(sorted(_glob("/opt/homebrew/Caskroom/claude-code/*/claude"))):
        if _os.access(path, _os.X_OK):
            return path
    return "claude"


CLI_WORKSPACE_ROOT = "/tmp/agent_generation"


def _flatten_to_prompt(system_prompt, user_content, workspace):
    """Collapse the provider-neutral content into a single CLI prompt string.

    Images are written into `workspace` (the CLI's cwd) and referenced by
    their bare filename, so the model's Read tool stays inside the workspace
    and never has to touch a path outside it (which the headless CLI would
    deny — the cause of the silent "images unreadable -> fall back to a
    generic object" failure).

    Returns (prompt, image_names) — `image_names` are the reference files
    written into the workspace, so the caller can exclude them from the
    copy-back (only model-generated files belong in results/)."""
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    image_names = []
    if isinstance(user_content, str):
        text_parts = [user_content] if user_content else []
    else:
        text_parts = []
        for i, p in enumerate(user_content, 1):
            if p["type"] == "text":
                text_parts.append(p["text"])
            elif p["type"] == "image":
                ext = (p.get("mime", "image/png").split("/")[-1] or "png")
                # Prefer the original filename; fall back to a stable index.
                fname = p.get("name") or f"reference_{i}.{ext}"
                dst = _os.path.join(workspace, fname)
                with open(dst, "wb") as f:
                    f.write(p["data"])
                image_names.append(fname)
            else:
                raise ValueError(f"unknown part type: {p['type']!r}")
    if image_names:
        parts.append("The reference image(s) are saved in your current working "
                     "directory. Use the Read tool to inspect each one (open by "
                     "its filename) before answering.")
        parts.extend(f"Image {i}: {name}" for i, name in enumerate(image_names, 1))
    parts.extend(text_parts)
    return "\n\n".join(parts), image_names


def call_claude_code(ctx, settings, system_prompt, user_content,
                     instance=None, cli_out_dir=None):
    """Run one inference via the Claude Code CLI in print mode.

    Each call runs in its own fresh workspace `/tmp/agent_generation/<random>/`
    (cwd), with any reference images copied in. The CLI is confined to that
    workspace (`--add-dir` names only it) so it can Read the images without
    a permission denial while being unable to reach other files. The
    workspace is removed after the call.

    Returns (text, usage) like the SDK adapters. Token counts come from the
    CLI's JSON `usage` block; the CLI's own `total_cost_usd` is surfaced under
    the `_cli_cost_usd` usage key (popped by the caller, not summed)."""
    import shutil as _shutil
    import tempfile as _tempfile

    model = ctx.get("cli_model")
    # Unique per-call workspace under /tmp/agent_generation so concurrent
    # workers never collide. mkdtemp creates it fresh and atomically.
    _os.makedirs(CLI_WORKSPACE_ROOT, exist_ok=True)
    workspace = _tempfile.mkdtemp(prefix="ws_", dir=CLI_WORKSPACE_ROOT)

    prompt, ref_image_names = _flatten_to_prompt(system_prompt, user_content, workspace)

    def build(tool_flag):
        cmd = [_claude_command(), "-p", prompt,
               "--output-format", "json",
               tool_flag, "Read",
               "--add-dir", workspace,
               "--permission-mode", "acceptEdits"]
        if model:
            cmd += ["--model", model]
        return cmd

    timeout = getattr(settings, "cli_timeout", None) or 1200
    try:
        completed = _subprocess.run(build("--tools"), check=False, cwd=workspace,
                                    capture_output=True, text=True, timeout=timeout)
        # Older CLIs use --allowedTools instead of --tools.
        if completed.returncode != 0 and "unknown option" in (completed.stderr or "").lower():
            completed = _subprocess.run(build("--allowedTools"), check=False, cwd=workspace,
                                        capture_output=True, text=True, timeout=timeout)

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        # Copy the model-generated files back to results/<model>/<instance>/
        # before teardown. The reference images we copied IN are excluded —
        # only what the model produced belongs in results/. `out_dir` is passed
        # per-call by the caller (thread-safe; ctx is shared across workers).
        out_dir = cli_out_dir
        if out_dir:
            skip = set(ref_image_names)
            _os.makedirs(out_dir, exist_ok=True)
            for entry in _os.listdir(workspace):
                if entry in skip:
                    continue
                src = _os.path.join(workspace, entry)
                dst = _os.path.join(out_dir, entry)
                if _os.path.isdir(src):
                    _shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    _shutil.copy2(src, dst)
    finally:
        _shutil.rmtree(workspace, ignore_errors=True)

    if completed.returncode != 0:
        raise RuntimeError(f"claude CLI exit {completed.returncode}: "
                           f"{stderr or stdout or '(no output)'}")

    try:
        parsed = _json.loads(stdout)
    except _json.JSONDecodeError:
        # Non-JSON stdout: treat the whole thing as the answer.
        return stdout, {}

    # A non-zero api_error_status (e.g. 404 bad model) comes back with rc 0.
    if parsed.get("api_error_status") or parsed.get("is_error"):
        raise RuntimeError(f"claude CLI error (api_error_status="
                           f"{parsed.get('api_error_status')}): "
                           f"{parsed.get('result')}")

    text = (parsed.get("result") or "").strip()
    u = parsed.get("usage") or {}
    in_tok    = u.get("input_tokens", 0) or 0
    cache_r   = u.get("cache_read_input_tokens", 0) or 0
    cache_w   = u.get("cache_creation_input_tokens", 0) or 0
    out_tok   = u.get("output_tokens", 0) or 0
    usage = {
        "input_tokens":          in_tok,
        "output_tokens":         out_tok,
        "thoughts_tokens":       None,
        "total_tokens":          in_tok + cache_r + cache_w + out_tok,
        "cache_read_tokens":     cache_r,
        "cache_creation_tokens": cache_w,
    }
    if parsed.get("total_cost_usd") is not None:
        usage["_cli_cost_usd"] = parsed["total_cost_usd"]
    return text, usage


_GEMINI_CLI_MODEL_ALIASES = {
    "gemini-cli": None,
    "gemini-cli-pro": "gemini-3-pro-preview",
    "gemini-cli-flash": "gemini-3-flash-preview",
    "gemini-cli-3-pro": "gemini-3-pro-preview",
    "gemini-cli-3-flash": "gemini-3-flash-preview",
    "gemini-cli-3-pro-preview": "gemini-3-pro-preview",
    "gemini-cli-3-flash-preview": "gemini-3-flash-preview",
}


def _resolve_gemini_cli_model(model):
    if model in (None, "gemini-cli"):
        return None
    if model in _GEMINI_CLI_MODEL_ALIASES:
        return _GEMINI_CLI_MODEL_ALIASES[model]
    if model.startswith("gemini-cli-"):
        suffix = model[len("gemini-cli-"):]
        if suffix.startswith("gemini-"):
            return suffix
        return f"gemini-{suffix}"
    return model


def _gemini_command():
    return _os.environ.get("GEMINI_CLI_COMMAND", "gemini")


def _extract_gemini_cli_text(parsed):
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("response", "result", "text", "content", "message", "output"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, (dict, list)):
                text = _extract_gemini_cli_text(value)
                if text:
                    return text
        candidates = parsed.get("candidates")
        if isinstance(candidates, list):
            text = _extract_gemini_cli_text(candidates)
            if text:
                return text
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            text = _extract_gemini_cli_text(item)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def _flatten_to_gemini_prompt(system_prompt, user_content, workspace):
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if isinstance(user_content, str):
        text_parts = [user_content] if user_content else []
    else:
        text_parts = []
        for i, p in enumerate(user_content, 1):
            if p["type"] == "text":
                text_parts.append(p["text"])
            elif p["type"] == "image":
                ext = (p.get("mime", "image/png").split("/")[-1] or "png")
                fname = p.get("name") or f"reference_{i}.{ext}"
                dst = _os.path.join(workspace, fname)
                with open(dst, "wb") as f:
                    f.write(p["data"])
                parts.append(f"Image {i}: @{fname}")
            else:
                raise ValueError(f"unknown part type: {p['type']!r}")
    if not isinstance(user_content, str):
        parts.append("Inspect every referenced image before answering.")
    parts.extend(text_parts)
    return "\n\n".join(parts)


def call_gemini_cli(ctx, settings, system_prompt, user_content,
                    instance=None, cli_out_dir=None):
    import shutil as _shutil
    import tempfile as _tempfile

    _os.makedirs(CLI_WORKSPACE_ROOT, exist_ok=True)
    workspace = _tempfile.mkdtemp(prefix="gemini_ws_", dir=CLI_WORKSPACE_ROOT)
    try:
        prompt = _flatten_to_gemini_prompt(system_prompt, user_content, workspace)
        cmd = [_gemini_command(), "-p", prompt, "--output-format", "json"]
        model = ctx.get("cli_model")
        if model:
            cmd += ["--model", model]

        timeout = getattr(settings, "cli_timeout", None) or 1200
        completed = _subprocess.run(cmd, check=False, cwd=workspace,
                                    capture_output=True, text=True, timeout=timeout)
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
    finally:
        _shutil.rmtree(workspace, ignore_errors=True)

    if completed.returncode != 0:
        output = stderr or stdout or "(no output)"
        raise RuntimeError(f"gemini CLI exit {completed.returncode}: {output}")

    try:
        parsed = _json.loads(stdout)
    except _json.JSONDecodeError:
        return stdout, {}

    text = _extract_gemini_cli_text(parsed).strip()
    usage = {
        "input_tokens":          0,
        "output_tokens":         0,
        "thoughts_tokens":       None,
        "total_tokens":          0,
        "cache_read_tokens":     0,
        "cache_creation_tokens": 0,
    }
    return text, usage


# ---- Antigravity (agy) CLI adapter -----------------------------------------
# Parity with 3D-CoT's
#   python inference_geometry_oneshot.py --generator_type agy-gemini-3-pro
# (TaskSolver's pyagy.AgyModel): the same pyagy client runs `agy --print`
# under a PTY in a git workspace, reusing the local Antigravity login
# (~/.gemini/antigravity-cli/) — no API key. Prompt shape mirrors
# AgyModel.prepare_payload: an image-hint block + "Image i: <path>" lines
# first, then all text (the system prompt folded in — agy has no separate
# system channel), joined with blank lines. settings.temperature / thinking /
# max_output_tokens / seed are ignored — the CLI owns generation.


def _resolve_agy_model(model):
    """`agy` / `antigravity` -> None (agy's default model); `agy-<model>` ->
    the suffix verbatim (`agy-gemini-3-pro` -> `gemini-3-pro`, the same rule
    as tasksolver/agent.py); bare ids (`gemini-3-pro`) pass through."""
    if model in (None, "agy", "antigravity"):
        return None
    if model.startswith("agy-"):
        suffix = model[len("agy-"):]
        if not suffix:
            raise SystemExit(
                f"Empty agy model suffix in {model!r}; use `agy` or "
                "`agy-<model>` (e.g. `agy-gemini-3-pro`)."
            )
        return suffix
    return model


def _flatten_to_agy_prompt(system_prompt, user_content, image_dir):
    """AgyModel.prepare_payload parity: the image hint + `Image i: <path>`
    lines come first, then every text part, all joined with blank lines."""
    strings, image_paths = [], []
    if system_prompt:
        strings.append(system_prompt)
    if isinstance(user_content, str):
        if user_content:
            strings.append(user_content)
    else:
        for i, p in enumerate(user_content, 1):
            if p["type"] == "text":
                strings.append(p["text"])
            elif p["type"] == "image":
                ext = (p.get("mime", "image/png").split("/")[-1] or "png")
                fname = p.get("name") or f"reference_{i}.{ext}"
                dst = _os.path.join(image_dir, fname)
                with open(dst, "wb") as f:
                    f.write(p["data"])
                image_paths.append(dst)
            else:
                raise ValueError(f"unknown part type: {p['type']!r}")
    parts = []
    if image_paths:
        parts.append("The visual inputs are saved as local image files. Use the Read "
                     "tool to inspect them when answering.")
        parts.extend(f"Image {i}: {p}" for i, p in enumerate(image_paths, 1))
    parts.extend(strings)
    return "\n\n".join(parts)


def call_agy(ctx, settings, system_prompt, user_content):
    import shutil as _shutil
    import tempfile as _tempfile

    agy_ask = ctx["agy_ask"]
    _os.makedirs(CLI_WORKSPACE_ROOT, exist_ok=True)
    # Holds the reference images only; pyagy creates its own git workspace
    # for the agy run (workspace=None), exactly like AgyModel without an
    # explicit workspace. Absolute paths keep the images reachable from there.
    image_dir = _tempfile.mkdtemp(prefix="agy_img_", dir=CLI_WORKSPACE_ROOT)
    try:
        prompt = _flatten_to_agy_prompt(system_prompt, user_content, image_dir)
        # 1800 = the print_timeout tasksolver/agent.py passes to AgyModel.
        timeout = getattr(settings, "cli_timeout", None) or 1800
        r = agy_ask(prompt, model=ctx["agy_model"], timeout=timeout)
        if not r.text:
            # Same failure surface as AgyModel._finish.
            raise RuntimeError(
                "agy --print returned no output "
                f"(exit_status={r.exit_status}, workspace={r.workspace}). "
                "Ensure agy is logged in (~/.gemini/antigravity-cli/) and reachable. "
                f"Transcript head:\n{r.transcript[:500]}"
            )
    finally:
        _shutil.rmtree(image_dir, ignore_errors=True)

    u = r.usage
    return r.text, {
        "input_tokens":          u.prompt_tokens,
        "output_tokens":         u.candidates_tokens,
        # Gemini 3 thinks dynamically (agy sends no explicit thinkingConfig);
        # the wire capture's raw usage carries the actual spend.
        "thoughts_tokens":       (u.raw or {}).get("thoughtsTokenCount"),
        "total_tokens":          u.total_tokens,
        "cache_read_tokens":     0,
        "cache_creation_tokens": 0,
    }


# ---- Codex CLI adapter ------------------------------------------------------
# Parity with 3D-CoT's
#   python inference_geometry_oneshot.py --generator_type codex-gpt-5-codex
# (TaskSolver's pycodex.CodexModel): the same pycodex client runs the bundled,
# wirecap-instrumented `codex exec` and reads the decoded turn from its
# capture JSONL. Auth: an explicit api_key is forwarded as OPENAI_API_KEY;
# otherwise the inherited environment, else the local `codex login`
# (~/.codex/auth.json). Prompt shape mirrors CodexModel.prepare_payload: an
# image-hint block + "Image i: <path>" lines first, then all text (the system
# prompt folded in — `codex exec` has no separate system channel).
# settings.temperature / thinking / max_output_tokens / seed are ignored —
# the CLI owns generation.
#
# settings.codex_bin / $CODEX_BIN swap in an external codex CLI (a bare name
# resolves on PATH) for models the backend gates on a newer client than the
# bundled build (e.g. gpt-5.6-sol). External binaries lack the wirecap bridge,
# so that mode runs `codex exec --json` and parses the event stream instead of
# the capture JSONL.


def _resolve_codex_model(model):
    """`codex` -> None (codex's default model); `codex-<model>` -> the suffix
    verbatim (`codex-gpt-5-codex` -> `codex exec -m gpt-5-codex`, the same
    rule as tasksolver/agent.py); bare ids (`gpt-5-codex`) pass through."""
    if model in (None, "codex"):
        return None
    if model.startswith("codex-"):
        suffix = model[len("codex-"):]
        if not suffix:
            raise SystemExit(
                f"Empty codex model suffix in {model!r}; use `codex` or "
                "`codex-<model>` (e.g. `codex-gpt-5-codex`)."
            )
        return suffix
    return model


def _resolve_codex_bin(settings):
    """Optional external codex CLI: settings.codex_bin (config) else $CODEX_BIN.
    A bare name resolves on PATH; None -> the bundled wirecap-patched codex."""
    import shutil as _shutil
    spec = getattr(settings, "codex_bin", None) or _os.environ.get("CODEX_BIN")
    if not spec:
        return None
    spec = _os.path.expanduser(spec)
    path = _shutil.which(spec) if _os.sep not in spec else spec
    if not path or not (_os.path.isfile(path) and _os.access(path, _os.X_OK)):
        raise SystemExit(f"codex_bin {spec!r} not found or not executable.")
    return _os.path.abspath(path)


def _parse_codex_json_events(transcript):
    """Parse `codex exec --json` stdout. Returns (text, usage_dict, errors):
    the last agent_message text, the summed turn.completed usage (pycodex
    Usage field names), and any error-event messages."""
    text, errors = "", []
    usage = {"input_tokens": 0, "cached_input_tokens": 0,
             "output_tokens": 0, "reasoning_output_tokens": 0}
    for line in transcript.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
        except ValueError:
            continue
        kind = obj.get("type")
        if kind == "item.completed":
            item = obj.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                text = item["text"]
        elif kind == "turn.completed":
            m = obj.get("usage") or {}
            for k in usage:
                usage[k] += m.get(k) or 0
        elif kind == "error":
            errors.append(obj.get("message") or _json.dumps(obj))
    return text, usage, errors


def _flatten_to_codex_prompt(system_prompt, user_content, image_dir):
    """CodexModel.prepare_payload parity: the image hint + `Image i: <path>`
    lines come first, then every text part, all joined with blank lines."""
    strings, image_paths = [], []
    if system_prompt:
        strings.append(system_prompt)
    if isinstance(user_content, str):
        if user_content:
            strings.append(user_content)
    else:
        for i, p in enumerate(user_content, 1):
            if p["type"] == "text":
                strings.append(p["text"])
            elif p["type"] == "image":
                ext = (p.get("mime", "image/png").split("/")[-1] or "png")
                fname = p.get("name") or f"reference_{i}.{ext}"
                dst = _os.path.join(image_dir, fname)
                with open(dst, "wb") as f:
                    f.write(p["data"])
                image_paths.append(dst)
            else:
                raise ValueError(f"unknown part type: {p['type']!r}")
    parts = []
    if image_paths:
        parts.append("The visual inputs are saved as local image files; "
                     "read them when answering.")
        parts.extend(f"Image {i}: {p}" for i, p in enumerate(image_paths, 1))
    parts.extend(strings)
    return "\n\n".join(parts)


def call_codex(ctx, settings, system_prompt, user_content):
    import shutil as _shutil
    import tempfile as _tempfile

    codex_ask = ctx["codex_ask"]
    _os.makedirs(CLI_WORKSPACE_ROOT, exist_ok=True)
    # Per-call workspace, unlike CodexModel (serial, shared scratch repo):
    # workers here run concurrently and pycodex truncates + re-reads
    # `codex-capture.jsonl` inside the workspace on every call, so a shared
    # workspace would cross-read turns. `codex exec` runs with
    # --skip-git-repo-check, so a plain temp dir suffices. The reference
    # images live in the same dir, reachable by absolute path from the prompt.
    workspace = _tempfile.mkdtemp(prefix="codex_ws_", dir=CLI_WORKSPACE_ROOT)
    try:
        prompt = _flatten_to_codex_prompt(system_prompt, user_content, workspace)
        # 1800 = the timeout tasksolver/agent.py passes to CodexModel.
        timeout = getattr(settings, "cli_timeout", None) or 1800
        kwargs = {"model": ctx["codex_model"], "workspace": workspace,
                  "timeout": timeout}
        # The reference images live inside this /tmp workspace and the model
        # is told to read them itself, but codex's view_image/shell tooling
        # runs through the bwrap fs-sandbox helper: under the default
        # read-only policy the images are unreachable, and on kernels without
        # user namespaces (e.g. WSL2) bwrap cannot start at all — the model
        # then silently answers blind. Full access keeps the read tools
        # working; isolation comes from the per-call workspace cwd.
        flags = ["--sandbox", "danger-full-access"]
        external_bin = ctx.get("codex_bin")
        if external_bin:
            # No wirecap bridge in an external codex: the capture JSONL stays
            # empty, so ask for the machine-readable event stream instead.
            kwargs["codex_bin"] = external_bin
            flags.append("--json")
        kwargs["extra_flags"] = flags
        if ctx.get("codex_env"):
            kwargs["extra_env"] = ctx["codex_env"]
        r = codex_ask(prompt, **kwargs)
        if external_bin:
            from pycodex import Usage as _CodexUsage
            text, u, errors = _parse_codex_json_events(r.transcript)
            if not text:
                raise RuntimeError(
                    "codex exec --json returned no agent message "
                    f"(exit_status={r.exit_status}, codex_bin={external_bin}, "
                    f"workspace={r.workspace}). "
                    "Ensure codex is authenticated (OPENAI_API_KEY or `codex login`). "
                    + (f"Errors: {'; '.join(errors[:3])}" if errors else
                       f"Transcript head:\n{r.transcript[:500]}")
                )
            usage = _CodexUsage(
                total_tokens=u["input_tokens"] + u["output_tokens"], **u)
        else:
            if not r.text:
                # Same failure surface as CodexModel._finish.
                raise RuntimeError(
                    "codex exec returned no output "
                    f"(exit_status={r.exit_status}, workspace={r.workspace}). "
                    "Ensure codex is authenticated (OPENAI_API_KEY or `codex login`). "
                    f"Transcript head:\n{r.transcript[:500]}"
                )
            text, usage = r.text, r.usage
    finally:
        _shutil.rmtree(workspace, ignore_errors=True)

    # codex TokenUsage semantics match the OpenAI /v1/responses shape:
    # input_tokens includes cached, output_tokens includes reasoning. Split
    # them the same way call_openai_responses does.
    reasoning = usage.reasoning_output_tokens or 0
    return text, {
        "input_tokens":          usage.input_tokens - usage.cached_input_tokens,
        "output_tokens":         usage.output_tokens - reasoning,
        "thoughts_tokens":       reasoning or None,
        "total_tokens":          usage.total_tokens,
        "cache_read_tokens":     usage.cached_input_tokens,
        "cache_creation_tokens": 0,
    }


_PROVIDER_FUNCS = {
    "gemini":     call_gemini,
    "anthropic":  call_anthropic,
    "openai":     call_openai,
    "claude_code": call_claude_code,
    "gemini_cli": call_gemini_cli,
    "agy":        call_agy,
    "codex":      call_codex,
}


def build_provider_ctx(settings):
    """Instantiate the SDK client (and optional rate limiter) for the
    chosen provider. ``settings.rpm`` / ``settings.tpm`` (either may be
    ``None``) configure the shared leaky-bucket limiter."""
    if settings.provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise SystemExit("google-genai not installed. Run: pip install google-genai")
        ctx = {"client": genai.Client(api_key=settings.api_key), "types": types}
    elif settings.provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise SystemExit("anthropic not installed. Run: pip install anthropic")
        ctx = {"client": anthropic.Anthropic(api_key=settings.api_key)}
    elif settings.provider == "openai":
        try:
            import openai
        except ImportError:
            raise SystemExit("openai not installed. Run: pip install openai")
        ctx = {"client": openai.OpenAI(api_key=settings.api_key)}
    elif settings.provider == "claude_code":
        # No SDK client — the adapter shells out to the local `claude` CLI.
        # Resolve the --model id once here from the configured model label.
        ctx = {"client": None, "cli_model": _resolve_cli_model(settings.model)}
    elif settings.provider == "gemini_cli":
        # No SDK client — the adapter shells out to the local `gemini` CLI.
        # Auth comes from the Gemini CLI login / subscription state.
        ctx = {"client": None, "cli_model": _resolve_gemini_cli_model(settings.model)}
    elif settings.provider == "agy":
        # No SDK client — pyagy drives the local Antigravity `agy` CLI
        # (bundled in the 3D-CoT tasksolver wheel). Auth comes from the
        # agy login; AGY_BIN/AGY_SHIM env vars override the bundled binary.
        try:
            from pyagy import ask as agy_ask
        except ImportError:
            raise SystemExit(
                "pyagy not importable. It ships inside the 3D-CoT tasksolver "
                "wheel — run from the 3D-CoT pixi env (e.g. `pixi run python "
                "tasks/image_to_3d/run.py ...`)."
            )
        ctx = {"client": None, "agy_ask": agy_ask,
               "agy_model": _resolve_agy_model(settings.model)}
    elif settings.provider == "codex":
        # No SDK client — pycodex drives the local `codex` CLI (bundled in
        # the 3D-CoT tasksolver wheel; settings.codex_bin / $CODEX_BIN swap in
        # an external binary — see call_codex). Auth comes from
        # settings.api_key / OPENAI_API_KEY, else the local `codex login`.
        try:
            from pycodex import ask as codex_ask
        except ImportError:
            raise SystemExit(
                "pycodex not importable. It ships inside the 3D-CoT tasksolver "
                "wheel — run from the 3D-CoT pixi env (e.g. `pixi run python "
                "tasks/image_to_3d/run.py ...`)."
            )
        ctx = {"client": None, "codex_ask": codex_ask,
               "codex_model": _resolve_codex_model(settings.model),
               "codex_bin": _resolve_codex_bin(settings)}
        if settings.api_key:
            ctx["codex_env"] = {"OPENAI_API_KEY": settings.api_key}
    else:
        raise SystemExit(f"Unknown provider: {settings.provider!r}")

    rpm = getattr(settings, "rpm", None)
    tpm = getattr(settings, "tpm", None)
    if rpm or tpm:
        ctx["rate_limiter"] = RateLimiter(rpm=rpm, tpm=tpm)
    return ctx


def call_provider(ctx, settings, system_prompt, user_content,
                  instance=None, out_dir=None):
    """Dispatch to the right adapter, with rate-limited send + retry.

    `instance` / `out_dir` are only meaningful for CLI adapters
    (they name its per-call /tmp workspace and where to copy results back).
    They're passed per-call rather than via the shared `ctx` because workers
    run concurrently. SDK adapters ignore them.

    Quota / availability errors (429, 503, rate_limit, etc.) retry
    indefinitely (capped at ``QUOTA_RETRY_CAP``) without consuming the
    standard ``RETRY_BACKOFF_S`` budget; other transient errors burn
    one slot from that budget per attempt.
    """
    fn = _PROVIDER_FUNCS[settings.provider]
    limiter = ctx.get("rate_limiter")
    extra = ({"instance": instance, "cli_out_dir": out_dir}
             if settings.provider in ("claude_code", "gemini_cli") else {})

    quota_attempts = 0
    transient_attempts = 0
    while True:
        handle = limiter.acquire() if limiter else None
        try:
            text, usage = fn(ctx, settings, system_prompt, user_content, **extra)
            if limiter:
                limiter.reconcile(handle, usage.get("total_tokens"))
            return text, usage
        except Exception as e:
            if limiter:
                limiter.reconcile(handle, 0)  # refund the failed reservation
            if _is_fatal_quota(e):
                # Out-of-credit / billing-cap is permanent; retrying just
                # burns ~5h of backoff per worker. Fail fast.
                raise
            if _is_quota(e):
                quota_attempts += 1
                if quota_attempts > QUOTA_RETRY_CAP:
                    raise
                idx = min(quota_attempts - 1, len(QUOTA_BACKOFF_S) - 1)
                time.sleep(QUOTA_BACKOFF_S[idx])
                continue
            if _is_other_transient(e):
                if transient_attempts >= len(RETRY_BACKOFF_S):
                    raise
                time.sleep(RETRY_BACKOFF_S[transient_attempts])
                transient_attempts += 1
                continue
            raise
