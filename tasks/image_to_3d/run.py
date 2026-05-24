"""Image-to-3D evaluation.

Same loop as text_to_3d, but conditions the LLM on a reference image
(rendered from the ground-truth factory) instead of a text description.

Usage:
    python tasks/image_to_3d/run.py --config configs/gemini_3_1_pro.yaml --task image_to_3d
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.runner import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
