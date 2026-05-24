"""visual_feedback (self-critique) helpers.

Single-call critic+fixer: model is shown its own previous code + 4 rendered
views + the original description, and is asked to either say NEEDS_FIX: NO
(stop) or NEEDS_FIX: YES + assessment + corrected full Python script.

Provider-neutral payloads: returns the same shape `inputs.load_user_content`
returns (str for text-only, list-of-parts for image-bearing). The render PNGs
are appended as image parts using the same `{"type":"image","mime":...,"data":bytes,"name":...}`
schema the existing providers adapter understands.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Union, List

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt"
_USER_TEMPLATE = {
    "text_to_3d":  _PROMPT_DIR / "visual_critique_user_template_text.txt",
    "image_to_3d": _PROMPT_DIR / "visual_critique_user_template_image.txt",
}
_SYSTEM_PROMPT = {
    "text_to_3d":  _PROMPT_DIR / "visual_critique_system_prompt_text.txt",
    "image_to_3d": _PROMPT_DIR / "visual_critique_system_prompt_image.txt",
}


def critique_system_prompt(task: str = "text_to_3d") -> str:
    """Task-specific system prompt. text_to_3d frames the comparison as
    'render vs description'; image_to_3d frames it as 'render vs reference
    images' (so the model is told to actually compare to the attached refs,
    not to a non-existent 'description')."""
    path = _SYSTEM_PROMPT.get(task)
    if path is None:
        raise ValueError(f"unknown task for visual_critique system prompt: {task!r}")
    return path.read_text()


def _format_original_task(original_user_content: Union[str, list]) -> str:
    """Render the original task back into the critique prompt as text.
    text_to_3d -> the description string; image_to_3d -> the leading
    instruction + image filenames (the actual reference image bytes are
    re-attached further down)."""
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
        out += f"\n\n(Original reference image(s) attached below: {', '.join(image_names)})"
    return out


def _load_render_pngs(render_dir: Path) -> List[dict]:
    """Read the canonical 4 turntable PNGs (Image_005 / 015 / 025 / 035) into
    image parts for the provider adapter. Skips any that don't exist; returns
    whatever's available (could be 0..4 PNGs).
    """
    parts = []
    for fname in ("Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"):
        p = render_dir / fname
        if not p.is_file():
            continue
        parts.append({
            "type": "image",
            "mime": "image/png",
            "data": p.read_bytes(),
            "name": fname,
        })
    return parts


def build_critique_user_content(
    original_user_content: Union[str, list],
    prev_code: str,
    render_dir: Path,
    *,
    iter_num: int,
    max_iter: int,
    task: str = "text_to_3d",
):
    """Construct the critique user message, dispatched on `task`.

    text_to_3d: text body cites the description; only the model's renders
    are attached as images.

    image_to_3d: text body explicitly says "compare your renders to the
    reference images"; the original 4 reference images are re-attached
    FIRST, then the model's 4 renders, so the model sees them in the
    expected order (refs[0..3], then renders[0..3]).
    """
    template_path = _USER_TEMPLATE.get(task)
    if template_path is None:
        raise ValueError(f"unknown task for visual_critique user template: {task!r}")
    template = template_path.read_text()

    if task == "text_to_3d":
        body = template.format(
            original_task_block=_format_original_task(original_user_content),
            prev_code=prev_code.rstrip(),
            iter_num=iter_num,
            max_iter=max_iter,
        )
    else:  # image_to_3d
        body = template.format(
            prev_code=prev_code.rstrip(),
            iter_num=iter_num,
            max_iter=max_iter,
        )

    parts: List[dict] = [{"type": "text", "text": body}]

    # image_to_3d: re-attach original reference images FIRST so the model
    # sees them in the order described by the prompt body.
    if not isinstance(original_user_content, str):
        for part in original_user_content:
            if part["type"] == "image":
                parts.append(part)

    # Then attach the renders the model is critiquing.
    parts.extend(_load_render_pngs(render_dir))
    return parts


_NEEDS_FIX_RE = re.compile(r"^\s*NEEDS_FIX\s*:\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)
_ASSESSMENT_RE = re.compile(r"<assessment>(.*?)</assessment>", re.IGNORECASE | re.DOTALL)
# Strict: <code>...</code>. Tolerates an optional markdown fence inside.
_CODE_STRICT_RE = re.compile(r"<code>(.*?)</code>", re.IGNORECASE | re.DOTALL)
# Lax fallback: model often opens <code> then closes with ``` instead of </code>.
# Match `<code>` (optional opening fence) (...content...) (optional closing fence)
# until either </code> or end of input.
_CODE_LAX_RE = re.compile(
    r"<code>\s*(?:```(?:python|py)?\s*\n)?(.*?)(?:\s*```)?\s*(?:</code>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Last-resort: a plain ```python ... ``` fenced block anywhere in the response.
_BARE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_LEADING_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n?", re.IGNORECASE)
_TRAILING_FENCE_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)


def _clean_code_block(code: str) -> str:
    """Strip stray markdown fences and surrounding whitespace from a code chunk."""
    code = code.strip()
    code = _LEADING_FENCE_RE.sub("", code)
    code = _TRAILING_FENCE_RE.sub("", code)
    return code.strip()


def _extract_code(text: str) -> Optional[str]:
    """Pull a code block out of the response, tolerant of a few common
    formatting deviations the model makes."""
    m = _CODE_STRICT_RE.search(text)
    if m:
        return _clean_code_block(m.group(1))
    m = _CODE_LAX_RE.search(text)
    if m and m.group(1).strip():
        return _clean_code_block(m.group(1))
    # Last resort — bare ``` fenced block (no <code> tags at all).
    m = _BARE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def parse_critique_response(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse the model's critique response.

    Returns:
        ("DONE",    assessment_text, None)        — model said NEEDS_FIX: NO
        ("FIX",     assessment_text, new_code)    — model said NEEDS_FIX: YES + provided code
        ("MALFORMED", raw_text, None)             — couldn't find required parts; caller decides

    Robustness:
      - If NEEDS_FIX line is missing but a code block exists, treat as FIX
        (model implied a fix).
      - If `<code>` opens but closes with ``` instead of `</code>`, accept it.
      - If only a bare ```python fenced block (no <code> tags), accept that too.
      - If NEEDS_FIX: YES but no extractable code → MALFORMED.
    """
    text = text or ""

    decision_m = _NEEDS_FIX_RE.search(text)
    assessment_m = _ASSESSMENT_RE.search(text)
    code = _extract_code(text)

    assessment = assessment_m.group(1).strip() if assessment_m else None

    if decision_m:
        decision = decision_m.group(1).upper()
        if decision == "NO":
            return ("DONE", assessment, None)
        # YES path — we need code to actually fix anything.
        if code:
            return ("FIX", assessment, code)
        return ("MALFORMED", assessment or text[:500], None)

    # No explicit NEEDS_FIX line — implicit FIX if a code block is present.
    if code:
        return ("FIX", assessment, code)
    return ("MALFORMED", text[:500], None)


def has_baseline_renders(out_dir: Path) -> bool:
    """True iff at least one canonical turntable PNG exists in out_dir/renders/."""
    rd = out_dir / "renders"
    if not rd.is_dir():
        return False
    return any((rd / n).is_file()
               for n in ("Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"))
