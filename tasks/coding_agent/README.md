# Coding-agent harness

Wraps the four CLI agents the paper benchmarks. Each gets a scratch
directory with the prompt + reference, then runs autonomously with file +
shell tools. We measure end-to-end pass rate of the final `*.glb`.

| Agent | Underlying model | CLI | Script |
|---|---|---|---|
| **Claude Code** | Claude Opus / Sonnet | [`claude`](https://docs.anthropic.com/en/docs/claude-code) | `run_claude_agent.sh` |
| **Codex CLI** | GPT-5.x | [`codex`](https://github.com/openai/codex) | `run_codex_agent.sh` |
| **Gemini CLI** | Gemini 3 Pro / Flash | [`gemini`](https://github.com/google-gemini/gemini-cli) | `run_gemini_agent.sh` |
| **agy (Antigravity CLI)** | Gemini 3.5 Flash (High) | [`agy`](https://google.github.io/antigravity/) | `run_agy_agent.sh` |

All four `run_*_agent.sh` scripts share the same contract:

```
$ bash tasks/coding_agent/<script>.sh <Category_seed0> [--text|--image]
```

## Common scratch-dir layout

Each script provisions:

```
agent_runs/<model>/<Category>_seed0/
├── prompt.md                  # task spec (description or image variant)
├── reference.glb              # ground-truth from benchmark/categories/<Cat>_seed0/
├── reference.png              # rendered preview (image variant only)
├── README_agent.md            # 1-paragraph instructions: "produce <Category>_seed0.py and call blender to export <Category>_seed0.glb"
└── (agent writes here)
```

The agent is told to:
1. Produce `<Category>_seed0.py` (a Blender 5.0 script that builds the object).
2. Run `blender --background --python <Category>_seed0.py` to verify it executes.
3. Export `<Category>_seed0.glb` and exit.

We then bake the produced `.py` independently for scoring -- agents that
silently faked the GLB don't count.

## Per-agent setup

### Claude Code (`run_claude_agent.sh`)
```bash
npm install -g @anthropic-ai/claude-code   # one-time
export ANTHROPIC_API_KEY=sk-ant-...
bash tasks/coding_agent/run_claude_agent.sh ArmChair_seed0
```
The script invokes `claude --model claude-opus-4-7 --print "$PROMPT"` and
caps wall-time at 30 min.

### Codex CLI (`run_codex_agent.sh`)
```bash
npm install -g @openai/codex                # one-time
export OPENAI_API_KEY=sk-...
bash tasks/coding_agent/run_codex_agent.sh ArmChair_seed0
```

### Gemini CLI (`run_gemini_agent.sh`)
```bash
npm install -g @google/gemini-cli           # one-time
export GEMINI_API_KEY=AIza...
bash tasks/coding_agent/run_gemini_agent.sh ArmChair_seed0
```

### agy / Antigravity (`run_agy_agent.sh`)
agy is OAuth-only and has a per-3.5h quota (~80 runs):
```bash
# Follow https://google.github.io/antigravity/ to install + agy login
bash tasks/coding_agent/run_agy_agent.sh ArmChair_seed0
```

## Batching across the benchmark

Each script also has an `_img.sh` variant for the image-conditioned task.
To sweep all 212 categories, wrap the per-instance script in a `for` loop or
`parallel` (no batch wrapper is shipped to keep the repo cluster-agnostic):

```bash
ls benchmark/categories | parallel -j 4 \
    bash tasks/coding_agent/run_claude_agent.sh {}
```

## Scoring agent outputs

Treat `agent_runs/<model>/` like any results tree:

```bash
python metrics/executability.py    --results-dir agent_runs/claude-opus-4-7
python metrics/image_similarity.py --results-dir agent_runs/claude-opus-4-7 \
                                   --reference-dir benchmark/categories
```

The paper reports executability + image-grounded scores for all four agents
side-by-side; see `tasks/README.md` for the canonical metric table.
