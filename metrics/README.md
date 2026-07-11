# Metrics

Each metric is a standalone scorer that consumes a results tree
(`results/<model>/<Category>_seed0/`) and emits per-instance JSON/CSV plus an
aggregate summary. Mix and match -- nothing in here depends on the runner.

| Metric | Measures | Inputs | GPU? | Extra setup |
|---|---|---|---|---|
| `executability.py` | Did the generated script run end-to-end in Blender 5.0 and produce a non-empty mesh? | `<results>/*.py` | no | -- |
| `failure_taxonomy.py` | Classify Blender tracebacks into buckets (geometry, modifier, node-tree, API mismatch, ...). | traceback strings | no | -- |
| `shape_chamfer.py` | Two-way Chamfer distance between point clouds sampled from generated vs reference GLBs (10k pts each). | two `.glb` paths | no | `pip install trimesh scipy` |
| `image_similarity.py` | SigLIP-2 / DINOv3 cosine between rendered views of the generated GLB and the reference. | rendered PNGs | yes | see [SigLIP-2 / DINOv3 setup](#siglip-2--dinov3-image_similaritypy) |
| `text_image_similarity.py` | SigLIP-2 cosine between the text prompt and rendered output. Only meaningful for `text_to_3d`. | prompt + rendered PNGs | yes | same env as above |
| `shape_uni3d.py` | Uni3D 3D&harr;3D cosine on sampled point clouds. | two `.glb` paths | yes | see [Uni3D setup](#uni3d-shape_uni3dpy) |
| `llm_judge/` | LLM-as-judge over code or rendered images (pairwise or absolute). | generated code or rendered PNGs | no | provider API key |

## Common arguments

```
--model <name>             sub-folder name under --results-root      # required for every metric
--results-root <path>      results/                                  # default: <repo>/results
--data-root <path>         benchmark/categories/                     # reference set for similarity / chamfer / uni3d
--instances <a> <b> ...    restrict to specific instances (default: all)
```

Per-metric summaries are written to `<results-root>/<model>/_metrics/<name>.json`.
Pass `--help` to any scorer for its full flag list (`executability.py` and
`image_similarity.py` add `--workers` / `--batch-size`; `failure_taxonomy.py`
instead takes `--roots <dir> ...` and `--out`; `llm_judge/judge.py` takes
`--judge` / `--mode`).

---

## `executability.py`

Aggregates the per-instance `renders/render_log.json` files written by
[`core/render.py`](../core/render.py) (which is what actually spawns Blender
5.0). A script "executes" iff Blender ran it without raising **and** at least
one mesh object exists afterwards. Pure Python, no Blender needed here — run
`core/render.py` first.

```bash
python metrics/executability.py --model gemini-3.1-pro-preview --results-root results
```

Output: `results/<model>/_metrics/executability.json` (pass rate, status
breakdown, top error fingerprints).

---

## `shape_chamfer.py`

Pure-Python: samples each GLB to a 10k-point cloud (`trimesh.sample_surface`),
computes mean two-way nearest-neighbour distance (scipy `cKDTree`). Normalised
so each cloud sits in a unit sphere first.

Both meshes must already be exported as GLBs (see
[`core/export_glb.py`](../core/export_glb.py)) — generated at
`<results-root>/<model>/<inst>/glb/<inst>.glb`, reference at
`<data-root>/<inst>/glb/<inst>.glb`.

```bash
pip install trimesh scipy
python metrics/shape_chamfer.py --model gemini-3.1-pro-preview --results-root results \
                                --data-root benchmark/categories
```

Output: per-instance distance + mean / median / std.

---

## SigLIP-2 / DINOv3 (`image_similarity.py`)

Both rely on `transformers` + a GPU. We recommend a dedicated conda env so the
heavier vision deps don't pollute the runner env:

```bash
# Create a vision env (CUDA 12.x assumed; pick a matching wheel if different):
conda create -n 3dcb-vision python=3.11 -y
conda activate 3dcb-vision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40 pillow numpy tqdm

# SigLIP-2 weights pull on first call (HuggingFace cache):
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/siglip2-base-patch16-224')"

# DINOv3 weights similarly:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-base')"
```

Run:

`--model` is the results sub-folder; the encoder is chosen with `--encoder`
(`siglip2` | `dinov2` | `dinov3`, default `siglip2`):

```bash
python metrics/image_similarity.py --model gemini-3.1-pro-preview --results-root results \
                                   --data-root benchmark/categories \
                                   --encoder siglip2
```

Switch to DINOv3:
```bash
python metrics/image_similarity.py --model gemini-3.1-pro-preview --results-root results \
                                   --data-root benchmark/categories --encoder dinov3
```

It compares the rendered views of the generated GLB against the reference views
under `<data-root>/<inst>/images/`.

---

## Uni3D (`shape_uni3d.py`)

Uses **Uni3D-Giant** for the point-cloud embedding and **OpenCLIP
EVA02-E-14-plus** for the text/image side. The Uni3D-Giant checkpoint
(`BAAI/Uni3D :: modelzoo/uni3d-g/model.pt`) is pulled automatically via
`hf_hub_download` on first run; you only need the Uni3D repo on disk, pointed to
by `UNI3D_REPO` (default `<repo>/external/Uni3D`).

```bash
# Same vision env as above.
pip install open_clip_torch ftfy regex
git clone https://github.com/baaivision/Uni3D external/Uni3D
export UNI3D_REPO=$PWD/external/Uni3D
```

Run:
```bash
python metrics/shape_uni3d.py --model gemini-3.1-pro-preview --results-root results \
                              --data-root benchmark/categories \
                              --n-points 8192
```

Output: per-instance 3D&harr;3D cosine (and a text/image-vs-shape CLIP score
unless `--no-clip`).

---

## LLM-as-judge (`llm_judge/`)

See [`llm_judge/README.md`](llm_judge/README.md). Two modes:

* **Absolute** -- judge gives each output a 1-5 score on the relevant rubric
  (`prompts/code_judge.txt` or `prompts/image_judge.txt`).
* **Pairwise (AB)** -- `pull_pairs.py` builds (modelA, modelB) pairs across
  two runs; `judge.py --mode image-ab` votes which of A/B better matches the
  reference; `summarize_ab.py` aggregates into win-rates + bootstrap CIs.

---

## Aggregating across metrics

The runner already writes per-instance JSON. To build the main paper table
(executability + image-grounded + 3D-shape + cost):

```bash
python scripts/aggregate.py results/gemini-3.1-pro-preview \
                            results/claude-opus-4-7 \
                            results/gpt-5.5 \
                            --out aggregate.csv
```

(That helper is paper-specific and lives outside this repo for now; the
per-metric JSONL files above are enough to build your own table.)
