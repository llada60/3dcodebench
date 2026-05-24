"""Build the user-message payload for a render-failure retry attempt.

Used by the multi_turn_debug loop in `run_inference.process_one`. The model
gets the original task description, its own previous code, and the Blender
stderr / render-log error, then is asked to output a corrected full script.

Design choices:
  - Stateless: each retry is a fresh `call_provider` call, NOT a
    conversation continuation. Works uniformly across Anthropic / Gemini /
    OpenAI providers and side-steps multi-turn billing surprises.
  - Provider-neutral: returns the same shape `inputs.load_user_content`
    returns (str for text_to_3d, list-of-parts for image_to_3d).
"""

from pathlib import Path
from typing import Union

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompt" / "render_feedback_user_template.txt"

# Hard cap on the error text fed back to the model. Blender tracebacks +
# scene dumps can be tens of KB; truncating preserves head + tail (the
# parts that actually identify the bug).
_MAX_ERROR_CHARS = 3000
_HEAD_FRAC = 0.7  # keep 70 % of the budget at the head, 30 % at the tail


def _truncate(text: str, max_chars: int = _MAX_ERROR_CHARS) -> str:
    if not text:
        return "(no error text recorded)"
    text = text.rstrip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * _HEAD_FRAC)
    tail = max_chars - head - 80
    return f"{text[:head]}\n\n... [truncated {len(text) - head - tail} chars] ...\n\n{text[-tail:]}"


def _render_log_error_text(render_log: dict) -> str:
    """Pull the most informative error string from the render_log dict."""
    for key in ("error", "stderr", "traceback"):
        v = render_log.get(key)
        if v:
            return _truncate(str(v))
    return "(render_log.json had no error / stderr field)"


def _format_original_task(original_user_content: Union[str, list]) -> str:
    """Render the original task back into the feedback prompt as text.

    text_to_3d: the description string.
    image_to_3d: the leading text instruction + image filenames (the actual
    bytes get re-attached as image parts further down).
    """
    if isinstance(original_user_content, str):
        return original_user_content
    text_lines, image_names = [], []
    for part in original_user_content:
        if part["type"] == "text":
            text_lines.append(part["text"])
        elif part["type"] == "image":
            image_names.append(part.get("name", "<image>"))
    out = "\n".join(text_lines).strip()
    if image_names:
        out += f"\n\n(Reference images attached below: {', '.join(image_names)})"
    return out


def build_render_feedback_content(
    original_user_content: Union[str, list],
    prev_code: str,
    render_log: dict,
    *,
    attempt_num: int,
    max_attempts: int,
):
    """Construct the user-message payload for a render-retry call.

    Returns the same provider-neutral shape as `load_user_content`:
      - text_to_3d  -> str
      - image_to_3d -> list[part]; original images are RE-ATTACHED so the
        retry call sees them too (the retry is stateless).
    """
    template = _TEMPLATE_PATH.read_text()
    body = template.format(
        original_task_block=_format_original_task(original_user_content),
        prev_code=prev_code.rstrip(),
        status=render_log.get("status", "?"),
        n_meshes=render_log.get("n_meshes", 0),
        n_views_rendered=render_log.get("n_views_rendered", 0),
        attempt_num=attempt_num,
        max_attempts=max_attempts,
        error_text=_render_log_error_text(render_log),
    )

    if isinstance(original_user_content, str):
        return body

    parts = [{"type": "text", "text": body}]
    for part in original_user_content:
        if part["type"] == "image":
            parts.append(part)
    return parts


def brief_error(render_log: dict, max_chars: int = 300) -> str:
    """One-line-ish error summary for the per-attempt history field."""
    err = render_log.get("error") or ""
    first = err.strip().splitlines()[0] if err.strip() else ""
    if len(first) > max_chars:
        first = first[:max_chars] + "..."
    return first
