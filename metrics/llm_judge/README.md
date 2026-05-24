# LLM-as-judge

Uses a strong LLM (Gemini 3 Pro by default) to score generated artifacts
either pairwise (A/B) or absolutely (1-5 Likert).

| File | Purpose |
|---|---|
| `judge.py` | Run the judge over a results tree. Supports `--mode code` or `--mode image`. |
| `pull_pairs.py` | Build A/B comparison pairs across two models for pairwise judging. |
| `summarize.py` | Aggregate absolute scores into per-model means. |
| `summarize_ab.py` | Aggregate pairwise verdicts into win-rates with bootstrap CIs. |
| `prompts/code_judge.txt` | System prompt for code-quality judging. |
| `prompts/image_judge.txt` | System prompt for rendered-image judging. |

Typical usage:

```bash
export GEMINI_API_KEY=...
python metrics/llm_judge/judge.py        --mode image \
    --results-dir results/gemini-3.1-pro-preview
python metrics/llm_judge/summarize.py    --judge-dir judgments/gemini-3.1-pro-preview
```
