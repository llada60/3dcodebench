<p align="center">
  <img src="assets/logo.png" alt="3DCodeBench" width="800"/>
</p>

**Benchmarking Agentic Procedural 3D Modeling Via Code.**
*Yipeng Gao, Lei Shu, Genzhi Ye, Xi Xiong, Ameesh Makadia, Meiqi Guo, Laurent Itti, Jindong Chen*

| Project page | Paper | Online arena |
|---|---|---|
| [3dcodebench.com](https://www.3dcodebench.com) | [arXiv:2606.01057](https://arxiv.org/abs/2606.01057) | [3dcodebench.com/arena](https://www.3dcodebench.com/arena) |

## News

- [06/01/2026] Paper released on arXiv: [3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code](https://arxiv.org/abs/2606.01057).

3DCodeBench measures how well frontier models can **write Blender 5.0 Python
that procedurally builds a specific 3D object**. The benchmark covers 212
categories — chairs, plants, sea creatures, coral, kitchen hardware, … — each
with a ground-truth factory script, a text description, and a structured
instruction. We evaluate single-shot, multi-turn, and full coding-agent
settings, and score outputs on executability, image similarity (SigLIP-2 /
DINOv3), 3D-shape distance (Chamfer / Uni3D), and LLM-as-judge.

## Built on Infinigen

3DCodeBench builds on the [Infinigen](https://github.com/princeton-vl/infinigen)
procedural generation ecosystem. The benchmark categories and the broader
3DCodeData corpus are distilled from Infinigen / Infinigen Indoors procedural
assets, then converted into standalone Blender 5.0 scripts for evaluating
text-, image-, and agent-driven procedural 3D modeling via code.

If you use the released benchmark, dataset, or generated factories, please cite
3DCodeBench and the relevant Infinigen works listed below.

## Repository layout

```
3dcodebench/
├── benchmark/categories/   212 categories, each = factory.py + 2 prompt txts
├── tasks/                  one entry per eval setting
│   ├── text_to_3d/         description → Blender Python
│   ├── image_to_3d/        rendered image → Blender Python
│   ├── multi_turn/         T=3 retry loop with traceback feedback
│   └── coding_agent/       Claude Code / Codex / Gemini CLI / agy wrappers
├── metrics/                executability, SigLIP/DINOv3, Chamfer, Uni3D, LLM judge
├── core/                   shared runner, provider abstraction, render/export
├── configs/                one YAML per model (API key from env)
├── prompts/                system + template prompts
├── data_pipeline/          how 3DCodeData was curated: operators + key notes
├── doc/                    method docs (e.g. the two renderers)
├── CONTRIBUTING.md         how to add new categories
└── LICENSE
```

Each subdirectory has its own README. Start with [`tasks/README.md`](tasks/README.md)
to see what to run, then [`metrics/README.md`](metrics/README.md) for scoring.

## Quickstart

### Install

```bash
git clone https://github.com/gaoypeng/3dcodebench.git
cd 3dcodebench
pip install -r requirements.txt

# Blender 5.0 must be installed separately (https://www.blender.org/download/).
# The render / GLB-export / runner steps read $BLENDER (falling back to
# `blender` on PATH); every Blender-driven script also takes --blender to override.
export BLENDER=/path/to/blender-5.0/blender

# API keys -- set the ones for the providers you'll call:
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

### Fetch the benchmark data

The 212 reference categories live on HuggingFace, not in this repo. Pull
them into `benchmark/categories/` (where every task expects to find them):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download YipengGao/3DCode \
    --repo-type dataset --include "3DCodeBench/*" \
    --local-dir /tmp/3dcode_dl
mkdir -p benchmark/categories
mv /tmp/3dcode_dl/3DCodeBench/* benchmark/categories/
```

The broader **3DCodeData** corpus (212 factories &times; 60 seeds =
**12,720 instances**) is in the same dataset under `3DCodeData/` -- use
`--include "3DCodeData/*"` instead. Each instance ships two self-contained
Blender 5.0 scripts (full-material + geometry-only), 2 caption variants, 4
multi-view WebP renders, and two exported meshes (a baked textured GLB + a
white-mode geometry GLB for shape scoring). See
[benchmark/README.md](benchmark/README.md) for details, and
[data_pipeline/](data_pipeline/) for how it was curated.

### Raw model logs (optional)

Our own inference runs — the **raw outputs of every evaluated model and coding
agent** (**81,605** generated Blender scripts across 82,042 trials, plus 2,767
agent transcripts) — are released alongside the benchmark under
`3DCodeBench_ModelLogs/` in the same dataset. Use them to reproduce our
leaderboard numbers or for error / cost analysis without re-running inference:

```bash
huggingface-cli download YipengGao/3DCode \
    --repo-type dataset --include "3DCodeBench_ModelLogs/**" --local-dir model_logs
```

See [**Dataset organization**](#dataset-organization) below for the full layout
and how to load the generated code.

### Reference images for image-to-3D

The `3DCodeBench/` eval set ships **code + two prompts per category, no images**.
`text_to_3d` and the code-only tasks therefore work straight after the download
above. **`image_to_3d` additionally needs reference views** at
`benchmark/categories/<inst>/images/Image_0{05,15,25,35}.png` — the four
canonical azimuths {45°, 135°, 225°, 315°} the benchmark was selected at.

Render them once from each ground-truth factory using the same camera
convention as the scorers (see the camera path in
[`core/render.py`](core/render.py); [`data_pipeline/operators/renderer.py`](data_pipeline/operators/renderer.py)
produces the equivalent multi-view PNGs). Put the four PNGs under each
instance's `images/` subdir. (Alternatively, point the runner at a different
reference-image folder via the `image_subdir` config field — e.g. a generated
reference image instead of the turntable renders.)

### Outputs

`results/<model>/<Category>_seed0/`:
* `<Category>_seed0.py` -- generated Blender Python
* `<Category>_seed0.glb` -- exported mesh
* `<Category>_seed0.json` -- usage + cost + traceback (on failure)

### Run inference

```bash
# Task 1: single-shot text-to-3D
python tasks/text_to_3d/run.py    --config configs/gemini_3_1_pro.yaml

# Task 2: single-shot image-to-3D
#   Conditions the model on reference views instead of text. These must
#   already exist at benchmark/categories/<inst>/images/Image_0{05,15,25,35}.png
#   (see "Reference images for image-to-3D" below — the benchmark ships code +
#   prompts only, so you render the references once from each factory .py).
python tasks/image_to_3d/run.py   --config configs/gemini_3_1_pro.yaml --task image_to_3d

# Task 3: multi-turn error-feedback loop (T=3 retries on failed instances)
python tasks/multi_turn/run.py    --config configs/gemini_3_1_pro.yaml \
                                  --max-feedback-rounds 3

# Task 4: coding-agent harness (Claude Code / Codex / Gemini CLI / agy)
#   See tasks/coding_agent/README.md for per-CLI setup.
bash tasks/coding_agent/run_claude_agent.sh ArmChair_seed0
```

### Score the outputs

```bash
# Scorers take --model (the sub-folder name under --results-root) plus,
# where a reference mesh/image is needed, --data-root (the benchmark set).
MODEL=gemini-3.1-pro-preview
ROOT=results
DATA=benchmark/categories

# Geometry-free scorers (no GPU required):
python metrics/executability.py    --model $MODEL --results-root $ROOT
python metrics/shape_chamfer.py    --model $MODEL --results-root $ROOT --data-root $DATA
python metrics/failure_taxonomy.py --roots $ROOT/$MODEL

# Image-grounded scorers (need GPU + SigLIP-2 / DINOv3 weights):
python metrics/image_similarity.py --model $MODEL --results-root $ROOT --data-root $DATA \
                                   --encoder siglip2
python metrics/image_similarity.py --model $MODEL --results-root $ROOT --data-root $DATA \
                                   --encoder dinov3

# 3D-3D scorer (needs Uni3D weights):
python metrics/shape_uni3d.py      --model $MODEL --results-root $ROOT --data-root $DATA

# LLM-as-judge (pairwise or absolute):
python metrics/llm_judge/judge.py  --judge gemini-3.1-pro-preview --mode image
```

> **Both `shape_chamfer.py` and `shape_uni3d.py` compare exported GLBs, so run
> [`core/export_glb.py`](core/export_glb.py) on both the model results and the
> reference set first** — e.g. `python core/export_glb.py --model $MODEL
> --results-root $ROOT` (and once for the reference factories under
> `--data-root`). Pass `--help` to any scorer for the full flag list.

See [`metrics/README.md`](metrics/README.md) for the full setup of SigLIP-2,
DINOv3, and Uni3D (model weights, conda env, GPU notes).

## Dataset organization

Everything lives in **one HuggingFace dataset, [`YipengGao/3DCode`](https://huggingface.co/datasets/YipengGao/3DCode)**,
under three top-level folders:

| Folder | What | Size |
|---|---|---|
| `3DCodeBench/` | The eval set — 212 categories, one canonical seed each: reference factory `.py` + `prompt_description.txt` + `prompt_instruction.txt`. No images (you render references yourself, see above). | 212 categories |
| `3DCodeData/` | The broader corpus — 212 factories × 60 seeds. Each instance ships a full-material script, a geometry-only `_geo.py`, 2 captions, 4 WebP renders, and two GLBs (textured + white-mode). Also exposed as `3DCodeData/data/train.parquet` for fast loading. | 12,720 instances |
| `3DCodeBench_ModelLogs/` | **Our raw inference logs** — every model's generated code, prompt, and per-call metadata, plus full coding-agent transcripts. | 82,042 trials |

### 📁 `3DCodeBench_ModelLogs/`

```
3DCodeBench_ModelLogs/
├── data/                       # 16 live settings, ONE PARQUET EACH
│   ├── text_to_3D.parquet      #   row = one trial; the generated code is the `code` column
│   ├── image_to_3D.parquet
│   ├── text_to_3D_agent.parquet
│   ├── *_multi_turn_debug.parquet / *_with_api_doc.parquet / *_visual_feedback*.parquet
│   ├── *_from_nbp*.parquet
│   └── thinking_ablation.parquet / temperature_ablation.parquet / images_amount_ablation.parquet
├── agent_logs/                 # 2,767 raw coding-agent transcripts
│   └── <setting>/<model>/<Object>_seed0/
│       ├── agent_stdout.log    #   full agent trajectory (tool calls, turns, stdout)
│       ├── agent_meta.json     #   num_turns, cost_usd, tokens, duration, exit code
│       └── agent_prompt.txt
├── deprecated/                 # 3 superseded/broken early runs — do NOT use for numbers
└── inputs/                     # shared inputs: 212 objects × (2 prompt txts + 4 PNG views)
```

> **The generated code is stored *inside* the parquet files, in the `code` column —
> not as loose `.py` files.** One row = one trial; pick a parquet by `setting`, a row by
> `(model, instance)`, and read `code`. This is intentional: it keeps the release light,
> lets you filter/aggregate by model or object in one line, and ships token/cost/status
> metadata next to every script.

**Where is everything:**

| You want… | Where it is |
|---|---|
| **Output code** (model-generated script) | `code` column of `data/<setting>.parquet` — one row per trial |
| Each multi-turn / visual-feedback attempt | `attempt_codes` column (JSON string → `list[str]`) |
| The exact prompt sent | `prompt` column |
| **Text input** (description / spec) | `inputs/<Object>_seed0/prompt_description.txt` · `prompt_instruction.txt` |
| **Image input** (4 reference views) | `inputs/<Object>_seed0/images/Image_0{05,15,25,35}.png` |
| Coding-agent full transcript | `agent_logs/<setting>/<model>/<Object>_seed0/agent_stdout.log` |
| Model / object / outcome / cost | `model`, `instance`, `status`, `cost_usd`, `*_tokens` columns |

**Scale:** 16 live settings hold **78,956** generated scripts; `deprecated/` adds **2,649**;
multi-turn / visual-feedback settings additionally keep **every** intermediate attempt in the
`attempt_codes` column (≈ 6,300 more) — **81,605 final scripts / ≈ 87,900 counting attempts**,
across up to 12 models × 212 objects plus the large ablation sweeps, with **2,767** agent
transcripts.

**Parquet columns** (28, identical across every setting; the two `*_agent` settings add 4 →
32): `setting`, `sub_task`, `model`, `instance`, `factory`, `seed` (usually null — the seed
index is in the instance name), `prompt` (exact input), **`code`** (the generated script),
`code_chars`, `n_attempts`, `attempt_codes` (a **JSON-encoded string** of `list[str]`, one per
attempt — multi-turn/visual-feedback only; `json.loads` it), `status`, `error`,
`input_tokens` / `output_tokens` / `thoughts_tokens` / `total_tokens`, `cache_read_tokens`,
`cache_creation_tokens`, `cost_usd`, `latency_s`, `parse_attempts`, `provider`, `temperature`,
`thinking`, `task` (`text_to_3d`/`image_to_3d`), `prompt_type`, `max_images`. Agent settings
add `num_turns`, `agent_exit`, `time_limit_s`, `max_budget`. `prompt`/`code`/`status`/tokens
are always populated; the rest are filled only where the provider reported them (nullable
numeric columns are `float` with `NaN` where missing).

### Loading the model outputs

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Load one setting's parquet (each row is one trial)
f = hf_hub_download("YipengGao/3DCode",
                    "3DCodeBench_ModelLogs/data/text_to_3D.parquet", repo_type="dataset")
df = pd.read_parquet(f)

# 2. Success rate per model
print(df.assign(ok=df.status.eq("OK")).groupby("model").ok.mean().sort_values())

# 3. Pull one trial's generated code and write it back to a .py file
row = df[(df.model == "gpt-5.5") & (df.instance == "Beetle_seed0")].iloc[0]
open("Beetle_seed0.py", "w").write(row.code)

# 4. Every intermediate attempt of a multi-turn run
import json
mt = pd.read_parquet(hf_hub_download("YipengGao/3DCode",
        "3DCodeBench_ModelLogs/data/text_to_3D_multi_turn_debug.parquet", repo_type="dataset"))
attempts = json.loads(mt.iloc[0].attempt_codes)   # list[str], one per turn
```

The `3DCodeData/` corpus loads the same way via the `datasets` library — see the
[dataset card](https://huggingface.co/datasets/YipengGao/3DCode) for details, or:

```python
from datasets import load_dataset
ds = load_dataset("YipengGao/3DCode", "3DCodeData", split="train")  # factory, code, code_geo, captions, preview
```

## Data pipeline

The factory scripts, renders, and meshes behind the **3DCodeData** corpus are
produced by the operators in [`data_pipeline/`](data_pipeline/) — validation,
multi-view rendering, reference-comparison grids, and seed-coverage summaries.
That directory also collects the **key curation notes**, most importantly
[`random_seed_setting.md`](data_pipeline/notes/random_seed_setting.md): Infinigen
uses two distinct seeds (raw `idx` for parameter sampling, `int_hash((idx,idx))`
for geometry), and conflating them silently produces objects with the wrong
proportions. See [`data_pipeline/README.md`](data_pipeline/README.md) for the
full picture.

For the two multi-view renderers — the reference/eval renderer
([`core/render.py`](core/render.py)) that produces the `Image_0X5` views and the
parts-colored curation renderer
([`data_pipeline/operators/renderer.py`](data_pipeline/operators/renderer.py)) —
their camera convention, scene setup, and outputs are documented in
[`doc/README.md`](doc/README.md).

## Contributing

The benchmark grows by **adding new categories**. If you have a procedural
Blender script for something we don't cover yet (a new vehicle, a building, a
musical instrument, …), please open a PR. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the format and review checklist.

PRs that add new **eval tasks** (e.g. a sketch-to-3D variant) or new
**metrics** (e.g. material-fidelity scorer) are also welcome — please open
an issue first to align on scope.

## Citation

Please cite 3DCodeBench, and also cite the Infinigen works that the procedural
asset library is based on:

```bibtex
@misc{gao2026threedcodebench,
  title  = {3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code},
  author = {Gao, Yipeng and Shu, Lei and Ye, Genzhi and Xiong, Xi and
            Makadia, Ameesh and Guo, Meiqi and Itti, Laurent and Chen, Jindong},
  year   = {2026},
  eprint = {2606.01057},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url    = {https://arxiv.org/abs/2606.01057}
}

@inproceedings{infinigen2023infinite,
  title={Infinite Photorealistic Worlds Using Procedural Generation},
  author={Raistrick, Alexander and Lipson, Lahav and Ma, Zeyu and Mei, Lingjie and Wang, Mingzhe and Zuo, Yiming and Kayan, Karhan and Wen, Hongyu and Han, Beining and Wang, Yihan and Newell, Alejandro and Law, Hei and Goyal, Ankit and Yang, Kaiyu and Deng, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12630--12641},
  year={2023}
}

@inproceedings{infinigen2024indoors,
  author    = {Raistrick, Alexander and Mei, Lingjie and Kayan, Karhan and Yan, David and Zuo, Yiming and Han, Beining and Wen, Hongyu and Parakh, Meenal and Alexandropoulos, Stamatis and Lipson, Lahav and Ma, Zeyu and Deng, Jia},
  title     = {Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {21783-21794}
}
```

## Acknowledgements

Categories are distilled from the [Infinigen](https://github.com/princeton-vl/infinigen)
procedural asset ecosystem, including Infinigen and Infinigen Indoors from the
Princeton Vision & Learning Lab.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The factory
scripts under `benchmark/categories/` retain Infinigen's BSD-3-Clause license.
