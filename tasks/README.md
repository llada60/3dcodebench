# Tasks

Each subdirectory is one **evaluation setting**. They share `core/runner.py`
but expose distinct entry points so it's clear what's being measured.

| Task | What the LLM sees | What we measure |
|---|---|---|
| `text_to_3d/` | A natural-language description of the object | Single-shot Blender Python correctness + visual fidelity |
| `image_to_3d/` | A rendered reference image of the object | Same, but conditioned on an image |
| `multi_turn/` | The previous attempt's code + the Blender traceback | Recovery rate after `T=3` stateless retries |
| `coding_agent/` | A free-form work directory and tool access (Claude Code / Codex / Gemini CLI / agy) | End-to-end agent harness pass rate |

## Running

Every task uses the same CLI signature, driven by a YAML config from `configs/`:

```bash
# pick a config (one per model); export the matching API key
export GEMINI_API_KEY=...
python tasks/text_to_3d/run.py    --config configs/gemini_3_1_pro.yaml
python tasks/image_to_3d/run.py   --config configs/gemini_3_1_pro.yaml --task image_to_3d
python tasks/multi_turn/run.py    --config configs/gemini_3_1_pro.yaml --max-feedback-rounds 3
```

Outputs land at `results/<model>/<Category>_seed0/`:
`<Category>_seed0.py` (generated code), `*.glb` (export), `*.json` (logs).

## Coding-agent task

`tasks/coding_agent/` wraps the real CLI agents (no Python entry point):

```bash
bash tasks/coding_agent/run_claude_agent.sh ArmChair_seed0
bash tasks/coding_agent/run_codex_agent.sh  ArmChair_seed0
bash tasks/coding_agent/run_gemini_agent.sh ArmChair_seed0
bash tasks/coding_agent/run_agy_agent.sh    ArmChair_seed0
```

Each script provisions a scratch dir, drops the prompt + reference, and lets
the agent iterate. See the scripts themselves for required env vars.
