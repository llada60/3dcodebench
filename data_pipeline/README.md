# Data pipeline

How the **3DCodeData** corpus on HuggingFace
([`YipengGao/3DCode`](https://huggingface.co/datasets/YipengGao/3DCode), folder
`3DCodeData/`) was curated — the operators that build it and the notes that
document the non-obvious decisions.

> These scripts assume an `objects_blender/`-style source tree: standalone
> Blender 5.0 factory scripts distilled from
> [Infinigen](https://github.com/princeton-vl/infinigen), one directory per
> category, each with per-seed variants. That source tree is **not** bundled
> here — this directory documents the *process*, not a turnkey rebuild.

## Pipeline at a glance

```
Infinigen factory                                          3DCodeData/ on HF
─────────────────                                          ─────────────────
  distill to a self-contained                  ┌─ <Name>_<NNN>.py        (full-material)
  Blender 5.0 script, per seed   ──────────▶   ├─ <Name>_<NNN>_geo.py    (geometry-only)
  (no infinigen imports)                        │
        │                                        ├─ <Name>_<NNN>.glb      (baked textured mesh)
        ├─ validate (executes? non-empty mesh?) ├─ <Name>_<NNN>_geo.glb  (white-mode mesh)
        ├─ render multi-view + compare to ref    ├─ renders/*.webp        (4 canonical views)
        └─ caption (VLM, image + code)           └─ captions/*.txt        (2 variants)
```

Each instance ships in two flavors — a **full-material** script (`<Name>_<NNN>.py`)
and a **geometry-only** script (`<Name>_<NNN>_geo.py`) — plus a baked **textured**
GLB and a **white-mode geometry** GLB used for Chamfer / Uni3D shape scoring.

## `operators/` — core scripts

| Script | What it does |
|---|---|
| `validate_blender_scripts.py` | Executes each standalone `_bpy.py` in Blender background mode and checks it produces a non-empty mesh (≥1 MESH object with >0 verts). Parallel workers + timeout. |
| `renderer.py` | Renders a script as a 360° turntable GIF (5-color per-part palette) + 8 octant PNGs. Multi-seed, recursive directory scan. |
| `viz_factory.py` | Renders factory variants and composes side-by-side comparison grids against the reference Infinigen renders, to eyeball seed-vs-groundtruth alignment. |
| `sum_factories.py` | Scans an output tree and summarizes variants + seed counts per category; flags categories with fewer than 60 seeds. |

All three Blender-driven scripts read the interpreter path from `$BLENDER`
(default points at a placeholder — set it to your Blender 5.0 binary).

## `notes/` — key decisions

| Note | Why it matters |
|---|---|
| [`random_seed_setting.md`](notes/random_seed_setting.md) | **The most load-bearing note.** Infinigen uses *two* seeds: `np.random.seed(idx)` for `__init__` parameter sampling (dimensions/proportions) and `np.random.seed(int_hash((idx, idx)))` for `create_asset` geometry construction. Mixing them up gives an object with the *wrong proportions* for the same seed index. Standalone scripts must keep both (`FACTORY_SEED = idx`, `SEED = int_hash`). |
| [`code_simplification.md`](notes/code_simplification.md) | The staged process for turning raw transpiled factories into clean, diverse, self-contained training scripts — with the golden rule that mesh output must be **identical** (vertex count + positions within 1e-4) before and after every transform. |
| [`texture_conversion.md`](notes/texture_conversion.md) | How geometry-only factories become full-material scripts: faithful raw-bpy replication of the Infinigen shader (no templates, no approximation), per-part material binding, single self-contained file. |

## Relationship to the benchmark

The `3DCodeBench/` eval set (212 canonical seeds) and the broader `3DCodeData/`
corpus (212 × 60 = 12,720 instances) are versioned together on HuggingFace. The
eval harness in [`../tasks/`](../tasks) scores model output against the
benchmark; this pipeline is what produced the underlying factory scripts,
renders, and meshes.
