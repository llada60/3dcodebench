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
--results-dir <path>       results/<model>/        # required for every metric
--reference-dir <path>     benchmark/categories/   # required for similarity / chamfer / uni3d
--workers <int>            parallel rendering / scoring threads (default 8)
--out <path>               per-instance JSONL output (default: <results>/metric_<name>.jsonl)
```

---

## `executability.py`

Spawns Blender 5.0 per instance, captures the exit code and the export, and
records a per-instance `{ok: bool, traceback: str}`. The script runs in a
sandbox temp dir; `BLENDER` env var must point to Blender 5.0.

```bash
export BLENDER=/path/to/blender-5.0/blender
python metrics/executability.py --results-dir results/gemini-3.1-pro-preview --workers 8
```

Output: `results/<model>/executability.jsonl` + a summary `executability.json`.

---

## `shape_chamfer.py`

Pure-Python: samples each GLB to a 10k-point cloud (`trimesh.sample_surface`),
computes mean two-way nearest-neighbour distance (scipy `cKDTree`). Normalised
so each cloud sits in a unit sphere first.

```bash
pip install trimesh scipy
python metrics/shape_chamfer.py --results-dir results/gemini-3.1-pro-preview \
                                --reference-dir benchmark/categories
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

```bash
python metrics/image_similarity.py --results-dir results/gemini-3.1-pro-preview \
                                   --reference-dir benchmark/categories \
                                   --model siglip2-base \
                                   --views 5            # cosine averaged across N camera views
```

Switch to DINOv3:
```bash
python metrics/image_similarity.py ... --model dinov3
```

The renderer launches Blender headless to render N orbiting views of each GLB
into `<results>/<Category>_seed0/views/view_NN.png` (cached across runs).

---

## Uni3D (`shape_uni3d.py`)

Uni3D embeddings need the Uni3D checkpoint and a CLIP text/image encoder for
the joint space.

```bash
# Same vision env as above.
pip install open_clip_torch ftfy regex

# Download Uni3D weights (Uni3D-base, ~430MB):
mkdir -p $HOME/.cache/uni3d
wget -O $HOME/.cache/uni3d/uni3d_base.pt \
    https://huggingface.co/BAAI/Uni3D/resolve/main/uni3d_base.pt
export UNI3D_CKPT=$HOME/.cache/uni3d/uni3d_base.pt
```

Run:
```bash
python metrics/shape_uni3d.py --results-dir results/gemini-3.1-pro-preview \
                              --reference-dir benchmark/categories \
                              --ckpt $UNI3D_CKPT \
                              --n-points 10000
```

Output: per-instance 3D&harr;3D cosine; `--mode 3d-text` also supported for
prompt&harr;mesh.

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
