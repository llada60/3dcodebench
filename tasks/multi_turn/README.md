# Multi-turn error-feedback

After the single-shot attempt, retry up to `T=3` times. **Stateless**: each
retry is a fresh API call that includes only the previous code and the
Blender traceback (no chat history). The goal is to measure how much
recovery a model can squeeze out of a bare traceback.

## How a retry round is built

For each failing instance, we re-run the system prompt with a user message
assembled from `prompts/multi_turn_feedback_template.txt`:

```
The previous attempt failed when running in Blender 5.0.

--- PREVIOUS CODE ---
{prev_code}

--- BLENDER TRACEBACK (last 40 lines) ---
{traceback_tail}

Please return the FULL corrected script, not a diff.
```

`core/feedback.py::brief_error()` truncates the traceback to the last
~40 lines and strips Blender's noisy frames (`bpy/_external/...`, etc.)
to keep the prompt under the typical context budget.

## Running

```bash
python tasks/multi_turn/run.py --config configs/gemini_3_1_pro.yaml \
                               --max-feedback-rounds 3
```

The runner first executes the single-shot pass (skipping any instance that
already has a passing `<Category>_seed0.glb`), then loops up to
`--max-feedback-rounds` over the still-failing ones.

Outputs:
```
results/<model>/<Category>_seed0/
├── <Category>_seed0.py            # final version (may be from any round)
├── <Category>_seed0.glb           # final export (only if any round passed)
├── <Category>_seed0.json          # usage + cost (aggregated across rounds)
└── _multi_turn_history.json       # per-round code + traceback for debugging
```

## Measuring uplift

Compare the executability rate before and after the loop:

```bash
python metrics/executability.py --results-dir results/<model> --history
```

The `--history` flag reads `_multi_turn_history.json` and reports
`single` vs `after_T1` vs `after_T2` vs `after_T3` pass rates.

For the &Delta;-metrics reported in the paper (post-loop SigLIP-2 / Chamfer /
Uni3D), restrict the comparison to each method's own success set:

```bash
python metrics/image_similarity.py --results-dir results/<model> \
                                   --reference-dir benchmark/categories \
                                   --filter own-success
```

The `--filter own-success` flag includes only instances that passed in *both*
the single-shot and the post-loop pass for that specific model -- so a
|&Delta;| near zero means recovered scripts produce shapes of comparable
quality to originally-passing ones.

## Debugging a single instance

If you want to step through a specific failure interactively:

```bash
CAT=Chameleon_seed0
python tasks/multi_turn/run.py --config configs/gemini_3_1_pro.yaml \
                               --instances $CAT \
                               --max-feedback-rounds 1 \
                               --verbose
# Then inspect the history:
cat results/gemini-3.1-pro-preview/$CAT/_multi_turn_history.json | jq .
```

`--verbose` prints the full feedback prompt sent on each retry, which is the
fastest way to spot prompt-engineering issues.
