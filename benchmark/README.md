# Benchmark categories

212 procedural-3D categories drawn from the Infinigen object library, each
distilled into three files:

```
benchmark/categories/<Category>_seed0/
├── <Category>_seed0.py          ← reference Blender Python factory (ground truth)
├── prompt_description.txt       ← short text description used by text_to_3d
└── prompt_instruction.txt       ← long structured spec used by text_to_3d (instruction variant)
```

`<Category>_seed0.py` is a self-contained Blender 5.0 script that, when run
headless (`blender --background --python …`), builds the reference mesh and
exports a GLB. It is the ground truth for chamfer / SigLIP / Uni3D scoring.

The two `.txt` files are the human-readable inputs the LLM sees:
- `prompt_description.txt` — one-paragraph caption (e.g. *"A 3D model of an
  upholstered armchair rendered from an elevated three-quarter perspective…"*)
- `prompt_instruction.txt` — multi-section structured spec covering geometry,
  proportions, parts, and finish hints.

For the image-to-3D task, the rendered reference image is produced on demand
from the factory by `core/render.py`; it is not checked in to keep the repo
lightweight.

## Adding a new category

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root for the full
workflow. The short version:

1. Drop a self-contained `<Name>_seed0.py` factory into a fresh subdirectory.
2. Write a one-paragraph `prompt_description.txt`.
3. (Optional) Write a structured `prompt_instruction.txt`.
4. Verify it runs: `blender --background --python <Name>_seed0/<Name>_seed0.py`.
5. Open a PR.
