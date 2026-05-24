"""Per-instance user content + provider response cleanup.

The loader returns a *provider-neutral* representation; each SDK adapter
in `utils/providers.py` translates it to its own schema.

Shape:
    text_to_3d   -> str  (the prompt text)
    image_to_3d  -> list[part]
        part = {"type": "text",  "text": str}
             | {"type": "image", "mime": str, "data": bytes, "name": str}

This module only knows about the on-disk layout under
`eval/data/<instance>/`:
    prompt_description.txt   (text_to_3d)
    prompt_instruction.txt   (text_to_3d, alternative)
    images/*.png|.jpg|...    (image_to_3d, all images used as multi-view input)
"""

import ast
import re
from pathlib import Path

# A single short user-side instruction for the image task. The system prompt
# already carries the heavy task definition; this just frames the images.
IMAGE_TASK_USER_TEXT = (
    "Reconstruct the object shown in the following reference image(s) as a "
    "Blender 5.0 Python script. Treat all images as views of the SAME object."
)

DEFAULT_IMAGE_SUBDIR = "images"
_IMAGE_EXTS = {".png": "image/png",
               ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg",
               ".webp": "image/webp"}

# When include_description=True, the image_to_3d user message becomes:
# "<text description>\n\n<this combined-mode instruction>" + image parts.
COMBINED_TEXT_IMAGE_INSTRUCTION = (
    "A reference image of the target object is attached below. "
    "Use BOTH the text description above AND the reference image to reconstruct "
    "the object as a Blender 5.0 Python script. The text description is the "
    "authoritative semantic spec (object identity, parts, materials); the "
    "reference image provides additional visual context (proportions, layout, style)."
)


def load_user_content(instance_dir: Path, task: str, prompt_type: str = "description",
                      max_images: int = None, image_subdir: str = DEFAULT_IMAGE_SUBDIR,
                      include_description: bool = False):
    """Return the user-message payload for one eval instance.

    `task` selects the input source; `prompt_type` is only consulted for
    text_to_3d (description vs instruction). `max_images` (image_to_3d only)
    truncates the sorted image list to the first N entries — used by the
    images_amount_ablation to feed 1/2/3/4 views in canonical order
    (Image_005, _015, _025, _035).

    `image_subdir` (image_to_3d only) selects which subdirectory under
    `instance_dir` holds the reference images. Defaults to "images" (the
    Infinigen turntable renders); can be set to e.g. "nano_banana_pro" to
    use a generated reference image instead.

    `include_description` (image_to_3d only) prepends the original text
    description (read from prompt_<prompt_type>.txt) to the user message
    BEFORE the image part, enabling a "text + image" combined input.
    Without this flag, image_to_3d gets only the generic instruction +
    image (the model has to infer semantics from the image alone).
    """
    instance_dir = Path(instance_dir)
    if task == "text_to_3d":
        path = instance_dir / f"prompt_{prompt_type}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name} in {instance_dir}")
        return path.read_text().strip()

    if task == "image_to_3d":
        img_dir = instance_dir / image_subdir
        if not img_dir.is_dir():
            raise FileNotFoundError(f"missing {image_subdir}/ in {instance_dir}")
        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in _IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f"no images in {img_dir}")
        if max_images is not None and max_images > 0:
            images = images[:max_images]
        if include_description:
            desc_path = instance_dir / f"prompt_{prompt_type}.txt"
            if not desc_path.exists():
                raise FileNotFoundError(
                    f"--include-description set but missing {desc_path.name} "
                    f"in {instance_dir}"
                )
            description = desc_path.read_text().strip()
            text_part = (
                "Original text description of the target object:\n\n"
                f"{description}\n\n"
                f"{COMBINED_TEXT_IMAGE_INSTRUCTION}"
            )
        else:
            text_part = IMAGE_TASK_USER_TEXT
        parts = [{"type": "text", "text": text_part}]
        for p in images:
            parts.append({
                "type": "image",
                "mime": _IMAGE_EXTS[p.suffix.lower()],
                "data": p.read_bytes(),
                "name": p.name,
            })
        return parts

    raise ValueError(f"unknown task: {task!r}")


def serialize_user_content(user_content) -> str:
    """Plain-text rendering for `prompt.txt` next to each output script."""
    if isinstance(user_content, str):
        return user_content
    text_lines = []
    image_names = []
    for part in user_content:
        if part["type"] == "text":
            text_lines.append(part["text"])
        elif part["type"] == "image":
            image_names.append(part.get("name", "<image>"))
    out = "\n".join(text_lines)
    if image_names:
        out += f"\n\n[reference images: {', '.join(image_names)}]"
    return out


_FENCE_LINE_RE = re.compile(r"^\s*```(?:python|py)?\s*$")


def strip_code_fence(text: str) -> str:
    """Sanitize a model response into runnable Python.

    Always drops standalone ``` / ```python / ```py fence lines. If the
    surrounding text still doesn't parse, search for the smallest pair
    of (head, tail) line trims that yields a parseable Python block —
    this catches prose preambles ("This script generates...") and
    postambles ("Hope this helps!") that the model adds despite the
    system prompt's "no prose" rule. Falls back to fence-stripped text
    if no trim parses (caller should detect and retry).
    """
    lines = [ln for ln in text.splitlines() if not _FENCE_LINE_RE.match(ln)]
    end_nl = text.endswith("\n")
    n = len(lines)

    def _emit(s):
        s = s.lstrip("\n")
        if end_nl and not s.endswith("\n"):
            s += "\n"
        return s

    base = "\n".join(lines)
    try:
        ast.parse(base)
        return _emit(base)
    except SyntaxError:
        pass

    # Search (skip_head, skip_tail) pairs by ascending total trim. Cap
    # the search depth at min(n, 60) lines on each side to keep this
    # bounded for pathological outputs.
    cap = min(n, 60)
    for total in range(1, 2 * cap + 1):
        for skip_head in range(min(total, cap) + 1):
            skip_tail = total - skip_head
            if skip_tail > cap or skip_head + skip_tail > n:
                continue
            stop = n - skip_tail if skip_tail else n
            candidate = "\n".join(lines[skip_head:stop])
            if not candidate.strip():
                continue
            try:
                ast.parse(candidate)
                return _emit(candidate)
            except SyntaxError:
                continue
    return _emit(base)
