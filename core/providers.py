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


def _flatten_to_prompt(system_prompt, user_content):
    """Collapse the provider-neutral content into a single CLI prompt string.
    Images are saved to disk and referenced by path (the CLI Read tool can
    open them), mirroring the TaskSolver adapter's behavior."""
    import tempfile
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    image_paths = []
    if isinstance(user_content, str):
        text_parts = [user_content] if user_content else []
    else:
        text_parts = []
        for p in user_content:
            if p["type"] == "text":
                text_parts.append(p["text"])
            elif p["type"] == "image":
                ext = (p.get("mime", "image/png").split("/")[-1] or "png")
                fd, path = tempfile.mkstemp(suffix="." + ext, prefix="ccimg_")
                with _os.fdopen(fd, "wb") as f:
                    f.write(p["data"])
                image_paths.append(path)
            else:
                raise ValueError(f"unknown part type: {p['type']!r}")
    if image_paths:
        parts.append("The visual inputs are saved as local image files. Use the "
                     "Read tool to inspect them when answering.")
        parts.extend(f"Image {i}: {pth}" for i, pth in enumerate(image_paths, 1))
    parts.extend(text_parts)
    return "\n\n".join(parts)


def call_claude_code(ctx, settings, system_prompt, user_content):
    """Run one inference via the Claude Code CLI in print mode.

    Returns (text, usage) like the SDK adapters. Token counts come from the
    CLI's JSON `usage` block; the CLI's own `total_cost_usd` is surfaced under
    the `_cli_cost_usd` usage key (popped by the caller, not summed)."""
    prompt = _flatten_to_prompt(system_prompt, user_content)
    model = ctx.get("cli_model")

    def build(tool_flag):
        cmd = [_claude_command(), "-p", prompt,
               "--output-format", "json",
               tool_flag, "Read",
               "--permission-mode", "acceptEdits"]
        if model:
            cmd += ["--model", model]
        return cmd

    timeout = getattr(settings, "cli_timeout", None) or 1200
    completed = _subprocess.run(build("--tools"), check=False,
                                capture_output=True, text=True, timeout=timeout)
    # Older CLIs use --allowedTools instead of --tools.
    if completed.returncode != 0 and "unknown option" in (completed.stderr or "").lower():
        completed = _subprocess.run(build("--allowedTools"), check=False,
                                    capture_output=True, text=True, timeout=timeout)

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
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


_PROVIDER_FUNCS = {
    "gemini":     call_gemini,
    "anthropic":  call_anthropic,
    "openai":     call_openai,
    "claude_code": call_claude_code,
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
    else:
        raise SystemExit(f"Unknown provider: {settings.provider!r}")

    rpm = getattr(settings, "rpm", None)
    tpm = getattr(settings, "tpm", None)
    if rpm or tpm:
        ctx["rate_limiter"] = RateLimiter(rpm=rpm, tpm=tpm)
    return ctx


def call_provider(ctx, settings, system_prompt, user_content):
    """Dispatch to the right adapter, with rate-limited send + retry.

    Quota / availability errors (429, 503, rate_limit, etc.) retry
    indefinitely (capped at ``QUOTA_RETRY_CAP``) without consuming the
    standard ``RETRY_BACKOFF_S`` budget; other transient errors burn
    one slot from that budget per attempt.
    """
    fn = _PROVIDER_FUNCS[settings.provider]
    limiter = ctx.get("rate_limiter")

    quota_attempts = 0
    transient_attempts = 0
    while True:
        handle = limiter.acquire() if limiter else None
        try:
            text, usage = fn(ctx, settings, system_prompt, user_content)
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
