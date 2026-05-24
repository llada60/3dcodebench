# Contributing to 3DCodeBench

Thanks for considering a contribution. The most useful PRs are **new
categories** (new objects the benchmark covers), but new tasks, metrics, and
bug fixes are equally welcome.

## Adding a new category

Each category lives at `benchmark/categories/<Name>_seed0/` and contains three
files:

```
benchmark/categories/<Name>_seed0/
├── <Name>_seed0.py          ← reference Blender 5.0 factory (the ground truth)
├── prompt_description.txt   ← short, single-paragraph caption
└── prompt_instruction.txt   ← long, structured spec (geometry + parts + finish)
```

### 1. Write the factory

`<Name>_seed0.py` must be a **self-contained** Blender 5.0 Python script:

- No imports outside the standard library and `bpy` / `bmesh` / `mathutils`.
- No reads from external files; no network access.
- Runs in `<5 minutes` of CPU time on a single core.
- Produces at least one mesh in the active scene.
- Deterministic for `seed=0` (we won't run other seeds at submission time).

A minimal skeleton:

```python
import bpy, bmesh, math
from mathutils import Vector

# 1. wipe the scene
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for x in list(coll):
        coll.remove(x)

# 2. build your mesh
me = bpy.data.meshes.new("MyThing")
ob = bpy.data.objects.new("MyThing", me)
bpy.context.collection.objects.link(ob)
bm = bmesh.new()
# ... add verts/edges/faces ...
bm.to_mesh(me); bm.free()
```

Verify it runs headless:

```bash
blender --background --python benchmark/categories/<Name>_seed0/<Name>_seed0.py
```

### 2. Write the prompts

`prompt_description.txt` should be **one paragraph** that a human could read
and visualize the object from. Match the existing tone — see
`ArmChair_seed0/prompt_description.txt` for the canonical style.

`prompt_instruction.txt` adds a more structured spec. Cover:
- Overall geometry / silhouette
- Major parts and their proportions
- Materials / finish hints (optional)

### 3. Open the PR

Title format: `Add category: <Name>`. Include:
- A 200×200 render of the factory output (drop it inline as a screenshot).
- One-line rationale of why the object is interesting (e.g. *"requires
  combining boolean ops with a curve modifier"*).

We'll review for: script runs cleanly, prompts are unambiguous, no overlap
with an existing category.

## Adding a new task

Add a `tasks/<your_task>/` directory with:
- `run.py` — the entry point (typically thin wrapper around `core.runner`)
- `README.md` — what the task measures, what it feeds the LLM, how to score

Open an issue describing the task before sending a PR so we can sanity-check
scope.

## Adding a new metric

Add `metrics/<your_metric>.py`. It should accept `--results-dir <path>` and
emit per-instance JSON/CSV plus an aggregate summary. Document the metric in
`metrics/README.md`.

## Code style

We don't enforce a formatter; just match the surrounding code. Prefer:
- Small, focused files (one responsibility per `.py`).
- Type hints on public functions.
- Comments only when *why* is non-obvious.
