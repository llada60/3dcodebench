# Benchmark data

The 212 reference categories that the evaluation in `tasks/*` runs against
**live on HuggingFace, not in this repository**:

**https://huggingface.co/datasets/YipengGao/3DCode** (folder: `3DCodeBench/`)

Each category is a `<Name>_seed0/` directory with three files:

```
3DCodeBench/<Category>_seed0/
├── <Category>_seed0.py          ← reference Blender 5.0 factory (ground truth)
├── prompt_description.txt       ← short, single-paragraph caption
└── prompt_instruction.txt       ← long, structured spec
```

## Downloading

The runner expects the data at `benchmark/categories/<Category>_seed0/...`.
Two one-liners that produce that layout:

```bash
# Option 1: huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download YipengGao/3DCode \
    --repo-type dataset --include "3DCodeBench/*" \
    --local-dir /tmp/3dcode_dl
mkdir -p benchmark/categories
mv /tmp/3dcode_dl/3DCodeBench/* benchmark/categories/

# Option 2: git clone (uses git-lfs; only fetch the benchmark folder)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/YipengGao/3DCode /tmp/3dcode_dl
cd /tmp/3dcode_dl && git lfs pull --include "3DCodeBench/**" && cd -
mkdir -p benchmark/categories
cp -r /tmp/3dcode_dl/3DCodeBench/* benchmark/categories/
```

## Why on HuggingFace?

The eval set + the broader **3DCodeData** corpus (212 factories &times; 60 seeds
= 12,720 instances, each with full-material + geometry-only scripts, 2 caption
variants, multi-view renders, and textured + white-mode GLBs) are versioned
together over there. HF handles the size + the partial-download tooling better
than git/GitHub LFS, and lets non-coders cite a stable dataset URL. See
[`../data_pipeline/`](../data_pipeline/) for how the corpus was curated.

## Adding a new category

The benchmark grows via PRs to the HuggingFace dataset, not this repo. See the
[`CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root for the format and
review checklist; once accepted, the maintainers move your category into the
HF dataset.
