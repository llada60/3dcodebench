"""Text-to-3D evaluation.

Feed each category's text description (`prompt_description.txt`) to the
chosen LLM, ask it to emit Blender 5.0 Python, then render + score the
resulting GLB against the reference.

Usage:
    python tasks/text_to_3d/run.py --config configs/gemini_3_1_pro.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.runner import main as _main  # noqa: E402

if __name__ == "__main__":
    # The yaml is expected to set: task=text_to_3d, prompt_type=description (or instruction).
    _main()
