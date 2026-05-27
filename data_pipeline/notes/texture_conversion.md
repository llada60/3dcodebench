# `objects_blender_texture/` Conversion Specification

Convert `objects_blender/<cat>/<Factory>.py` (geometry only) into `objects_blender_texture/<cat>/<Factory>.py` (geometry + materials). Core constraints and methodology below.

## Hard Requirements

1. **1:1 alignment with `objects_blender/`**: Directory structure and file list use `objects_blender/` as the source of truth — do not reference `objects_captions_seed/`.
2. **Self-contained single script**: Each factory script must contain both geometry and material code, depending only on `bpy / bmesh / numpy / math / random / colorsys`. **No infinigen imports**, no dependency on a shared `_materials.py` in the same directory.
3. **No templates**: Each factory is independently hand-converted. Different categories have different material loading patterns, part structures, and seeding conventions — mechanical inlining of a shared `_materials_template.py` is forbidden. `tmp/_materials_template.py` serves only as a reference for the appliance category and must not be mechanically inlined into new categories.
   - **Faithful shader replication is non-negotiable**: No matter how complex the infinigen source shader is (deeply nested procedural textures, multi-layer mixes, custom node groups, etc.), the textured script MUST faithfully reproduce it in raw bpy node tree code. Simplification, approximation, or substitution with a "close enough" shader is forbidden. If a shader is too complex to inline cleanly, still inline it fully — correctness of the visual result outweighs script length or readability.
4. **Geometry code unchanged**: Reuse the geometry logic from the `objects_blender/` source file verbatim. The only permitted insertions are `obj.data.materials.append(MAT_X)` before `join()` calls; the only permitted geometry modification is splitting `join([a, b])` into "append material to each part first, then join".
5. **Per-part multi-material**: The original infinigen factory typically binds different materials to body/door/handle/glass/screen etc. The textured script must replicate this granularity.

## Seeding Alignment with infinigen

Reference `infinigen/core/placement/factory.py:30-78` and `feedback_factory_seed.md`:

- **Materials** use the **raw `SEED`** (corresponding to infinigen's `with FixedSeed(factory_seed)` in `__init__`)
- **Geometry** uses the `objects_blender/` script's existing re-seed behavior (corresponding to infinigen's `FixedSeed(int_hash((factory_seed, i)))` in `create_asset`) — do not modify
- **Material sampling must happen before `build()`** (at module level), placed in the Phase 1 block after `seed_all(SEED)`. The `sample_params(seed)` inside `build()` will re-seed numpy.random for geometry state, but materials are already captured in module-level `MAT_*` variables and are unaffected.
- **Material sampling order must replicate the call order from infinigen's `get_material_params()`** (same numpy.random consumption sequence).
- **wear_tear 50/50 probability gates are also decided in Phase 1**, immediately after material sampling: `WEAR_DO_SCRATCH = np.random.uniform() <= 0.5; WEAR_DO_EDGEWEAR = ...`. Application happens at the end of `build()` after join.

## Script Structure Skeleton

```python
import os, math, bpy, bmesh, random, colorsys
import numpy as np

# === Inlined material library (hand-crafted for THIS factory's needs) ===
# Include only the material constructors this factory actually uses.
# Copy shader logic from infinigen/assets/materials/<type>/<name>.py, translated
# to raw bpy node tree construction. Each category typically needs different
# shaders — DO NOT copy a library from another category wholesale.
def seed_all(seed): np.random.seed(seed); random.seed(seed)
def weighted_sample(reg): classes, w = zip(*reg); w = np.array(w, float); return classes[int(np.random.choice(len(classes), p=w/w.sum()))]
def make_X(...): ...   # raw-bpy shader functions specific to this factory
REGISTRY = { "<key>": [(make_X, weight), ...], ... }
def get_material(key, name=None): return weighted_sample(REGISTRY[key])(name=name or key.title())
def apply_scratches(obj): ...    # if factory uses wear_tear
def apply_edge_wear(obj): ...

# === Original geometry helpers (copied verbatim from objects_blender/) ===
def sel_none(): ...
def box(...): ...
def sample_params(seed=0): np.random.seed(seed); ...   # KEEP AS-IS

# === Phase 1 — Material sampling on raw SEED (mirrors infinigen __init__) ===
SEED = int(os.environ.get('SEED', 0))
seed_all(SEED)
MAT_X = get_material("<key>", "<slot_name>")   # mirror call order from infinigen factory
...
WEAR_DO_SCRATCH  = np.random.uniform() <= 0.5
WEAR_DO_EDGEWEAR = np.random.uniform() <= 0.5

# === Geometry with per-part material assignment ===
def build(seed=SEED):
    p = sample_params(seed)   # re-seeds numpy for geometry; materials already captured
    ...
    body.data.materials.append(MAT_X)   # BEFORE any join()
    ...
    obj = join([body, door, ...])
    if obj:
        if WEAR_DO_SCRATCH:  apply_scratches(obj)
        if WEAR_DO_EDGEWEAR: apply_edge_wear(obj)
    return obj

build(SEED)
```

## Pre-Conversion Checklist (per factory)

1. Read `infinigen/assets/objects/<cat>/<file>.py` → extract `get_material_params()` (or equivalent `__init__` material sampling) to get the material slot → registry key mapping and call order.
2. Read `infinigen/assets/composition/material_assignments.py` → get the `(class, weight)` list for each registry key.
3. Read `infinigen/assets/materials/<type>/<name>.py` → translate the needed shader functions into raw bpy node tree code.
4. Read `objects_blender/<cat>/<file>.py` → confirm the independent object variable names and join nesting hierarchy in `build()`, and decide where to insert `.data.materials.append(...)`.
5. **Validate with the texture-aware renderers below** — a hand-converted factory is only considered done after its shader output matches the infinigen reference render on the required number of seeds.

## Validation Renderers

Two dedicated texture-aware renderers live in `data_pipeline_operators/` —
unlike `renderer.py` / `viz_factory.py`, these scripts **do not strip or
recolour materials**. They preserve the exact shader node trees each factory
constructs and light them with a neutral studio (white world + brighter
3-point area lights) so the rendered image is a faithful preview of the
shader output.

### Case A — Single-file factory with runtime `SEED` (current layout)

Use `renderer_texture.py`. Required: render **at least 5 seeds** and visually
confirm the materials match infinigen's reference render at the corresponding
seed. Any seed where the material diverges (wrong colour, missing roughness,
extra/missing procedural detail, wrong metal/dielectric response) blocks
acceptance — fix the shader and re-render.

```bash
# 5 seeds, turntable GIF + 8 octant stills per seed
python data_pipeline_operators/renderer_texture.py \
    objects_blender_texture/<cat>/<Factory>.py \
    -o renders_texture/<cat> -s 5

# Per-category sweep (all factories in one category, 5 seeds each)
python data_pipeline_operators/renderer_texture.py \
    objects_blender_texture/<cat> \
    -o renders_texture/<cat> -s 5
```

Compare each seed's rendered stills against
`output_factories/<cat>/<Factory>/<Factory>_<NNN>/Image_*.png`. Material
correspondence must hold for **every** rendered seed, not just seed 0.

### Case B — Per-seed batch scripts (future diversified layout)

Once a factory has been expanded into `objects_blender_texture/<cat>/<Factory>/<Factory>_NNN.py`
(one file per seed, seed hardcoded, no env var), use `viz_factory_texture.py`
for batch validation. It produces comparison grids with **10 seeds per page**,
code-rendered views on the left and yaw-aligned infinigen reference views on
the right, plus an `OK / CHK` silhouette-Dice badge per row so regressions are
easy to spot.

```bash
# Stream mode (recommended): render → grid → discard temp files
python data_pipeline_operators/viz_factory_texture.py \
    --factory <Factory> --seeds 0-49 --stream --per-page 10

# Whole category at once
python data_pipeline_operators/viz_factory_texture.py \
    --category <cat> --seeds 0-49 --stream --per-page 10
```

Row-by-row acceptance rule: for each row in the grid, the 4 code renders
(left) must visually correspond to the 4 infinigen references (right) — same
silhouette **and** same shader appearance. The `OK 0.xx` / `CHK 0.xx` badge
under each seed label is a silhouette-Dice score; `< 0.60` means the row
needs manual inspection. Material mismatches are not caught by the Dice
score — a row can read `OK` on geometry while still having a wrong shader, so
always eyeball the colour / roughness / metallic response per row.

### Why two scripts and not one

`renderer_texture.py` is optimised for **single-factory deep inspection**:
turntable GIF + 8 octants per seed is the right artefact for debugging a
shader you just hand-wrote. `viz_factory_texture.py` is optimised for
**batch regression**: a wall of 10 rows × 8 thumbnails on one page lets you
scan 50 seeds in a couple of seconds and immediately see which row diverges.
Keep them separate — do not merge.

## Blender 5.0 Key Adaptations

- `ShaderNodeTexMusgrave` removed → use `ShaderNodeTexNoise` (`musgrave_dimensions` → `noise_dimensions`)
- Principled BSDF socket: `Specular IOR Level` (4.x+) vs `Specular` (3.x) — use fallback
- Noise output `Fac` → `Factor` — use `noise.outputs.get("Fac") or noise.outputs[0]`
- Voronoi 4D may not be available — fallback to `3D`
- Mix node RGBA: `inputs[0]=Factor, inputs[6]=A, inputs[7]=B, outputs[2]=Result`
- `NodeSocketVirtual` removed → map to `NodeSocketFloat`
- `bpy.ops.mesh.triangulate` unavailable headless → use `bmesh.ops.triangulate()`
- EEVEE engine id: `BLENDER_EEVEE_NEXT` → `BLENDER_EEVEE`

## Key Lessons from Prior Conversions

Read before starting a new category.

### Seeding & regeneration discipline

- **Raw seed vs int_hash**: `__init__` (materials, factory-level RNG) uses the raw `factory_seed`; `create_asset` (geometry) uses `int_hash((factory_seed, i))`. Never mix them. The material sampling in Phase 1 must use the raw `SEED`; the geometry `sample_params()` already embeds the correct re-seed and must not be modified.
- **Fix in place, never regenerate**: When a converted factory has a bug, patch it directly in `objects_blender_texture/<cat>/<file>.py`. Do NOT re-run a generic template/converter — that destroys hand-tuned per-factory choices (shader details, part-to-slot mapping, fallback logic).

### Shader / node-tree pitfalls (Blender 5.0)

- **`CaptureAttribute` starts empty**: In 5.0, `capture_items` has size 0. You MUST call `node.capture_items.new('FLOAT', 'Value')` to register a channel, otherwise the node captures nothing (all outputs = 0). #1 cause of silent "flat result" bugs.
- **RandomValue output indices are data-type-dependent**: `FLOAT_VECTOR→[0]`, `FLOAT→[1]`, `INT→[2]`, `BOOLEAN→[3]`. For BOOLEAN, `outputs[0]` is the inactive vector output and always evaluates False — use `outputs[3]` explicitly.
- **Voronoi / Random hashing changed in 5.0**: Identical seeds produce different patterns vs 4.4. Do not rely on byte-exact reproduction across versions.
- **Point Density Texture removed entirely** in 5.0 — find an alternative (noise/voronoi-driven mask) if the infinigen source uses it.

### Batch validation fixes (apply proactively when copying helpers)

- `mesh.calc_normals()` removed in 5.0 → use `mesh.update()`.
- `v.select_set(np.bool_)` fails → wrap with `bool(...)`.
- `bpy.ops.object.material_slot_remove()` fails in background (poll error) → use `obj.data.materials.clear()`.
- `bpy.ops.mesh.primitive_*_add()` places at 3D cursor by default → ALWAYS pass `location=(0,0,0)` explicitly and reset cursor in `clear_scene()`: `bpy.context.scene.cursor.location = (0, 0, 0)`.
- GeoNodes modifier ordering: "Geometry input must be the first" warning silently produces no mesh — after adding sockets, call `ng.interface.move(geom_socket, 0)`.

### Workspace hygiene

- **Temp / debug scripts go in a scratch dir (e.g. `tmp/`)**, NOT inside `objects_blender_texture/<cat>/`. Keep category folders clean: they should contain only the final converted factory scripts.
- **Clean intermediate renders**: After `viz_factory`-style previews, `rm -rf` the per-seed directories; keep only the `grid_page` PNG as the human-visible verification artifact.

### Naming & readability

- Use semantic names in inlined helpers. Avoid cryptic single-letter/abbreviated locals (`_j`, `rr`, `ra`, `dt`). Prior conversions accumulated these and made debugging painful — don't repeat.
