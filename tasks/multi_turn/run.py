"""Multi-turn error-feedback loop.

After the single-shot attempt, retry up to T=3 times. Each retry is
stateless and feeds the previous code + the Blender traceback back to the
LLM via the multi_turn_feedback_template.

Usage:
    python tasks/multi_turn/run.py --config configs/gemini_3_1_pro.yaml \
        --max-feedback-rounds 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.runner import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
