# Random Seed setting

Please write the setting of random seed for infinigen original assets and objects_blender code definition and how to get the seed scripts for each factory. The setting of random seed is very important to promise the generated seed code correponds to the rendered images.

## Summary

To ensure a standalone script (e.g., `DishwasherFactory_000.py`) produces the same geometry as infinigen's rendered reference for the same seed index, the random seed for **parameter sampling** must be set to the **raw index value** (0, 1, 2, ...), NOT `int_hash((idx, idx))`.

Infinigen's factory system uses two different seeds at two stages: the `__init__` stage samples parameters (dimensions, proportions) using `np.random.seed(idx)`, while the `create_asset` stage constructs geometry using `np.random.seed(int_hash((idx, idx)))`. The standalone scripts in `objects_blender/` currently store `SEED = int_hash((idx, idx))` (e.g., 543568399 for idx=0). This `SEED` value is correct for geometry-level randomness, but it must NOT be used for parameter sampling. For parameter sampling, scripts must use `FACTORY_SEED = idx` (e.g., 0 for `_000.py`, 5 for `_005.py`). Using the wrong seed causes the standalone script to generate objects with different proportions (e.g., a dishwasher that is short and wide instead of tall and slim).

## How Infinigen's Seed System Works

1. **Factory instantiation**: `generate_individual_assets.py` creates each factory with `fac = Factory(idx)`, where `idx` (0, 1, 2, ...) is the **raw factory_seed**.

2. **Two seed contexts in the factory lifecycle**:
   - `__init__`: Uses `with FixedSeed(factory_seed)` — sets `np.random.seed(idx)` (raw index). Parameter sampling (`sample_parameters()`) happens HERE.
   - `spawn_asset(i)`: Uses `with FixedSeed(int_hash((factory_seed, i)))` — sets `np.random.seed(int_hash((idx, idx)))`. Geometry node construction (`create_asset()`) happens HERE.

3. **`int_hash` function**: `int_hash((a, b))` uses MD5 to produce a deterministic large integer from a tuple. Example: `int_hash((0, 0)) = 543568399`.

4. **`FixedSeed` context manager**: Sets `np.random.seed(value)` on entry, restores previous state on exit. Defined in `infinigen/core/util/math.py`.

## Critical Rule: Never Mix the Two Seeds

- **Parameter sampling** (dimensions, proportions, counts): Use `np.random.seed(factory_seed)` = `np.random.seed(idx)`.
- **Geometry construction** (node tree evaluation, mesh operations): Use `np.random.seed(int_hash((factory_seed, idx)))`.

Mixing them up causes the standalone script to generate **different dimensions** than the infinigen reference for the same seed index. For example, DishwasherFactory with idx=0:

| Seed method | Depth | Width | Height | H/W ratio |
|---|---|---|---|---|
| `np.random.seed(0)` (correct) | 1.176 | 1.040 | 1.098 | 1.06 (tall) |
| `np.random.seed(543568399)` (wrong) | 0.875 | 1.070 | 0.762 | 0.71 (short) |

## How to Set Seeds in Standalone Scripts

### Pattern for factories with `sample_params()` (e.g., appliances, tables)

```python
SEED = 543568399          # int_hash((idx, idx)), for geometry-level randomness
FACTORY_SEED = 0          # raw idx, for __init__ parameter sampling

def sample_params():
    np.random.seed(FACTORY_SEED)      # Must use raw idx, NOT int_hash
    depth = 1 + np.random.normal(0, 0.1)
    width = 1 + np.random.normal(0, 0.1)
    ...

def build():
    p = sample_params()               # Parameters match infinigen's __init__
    np.random.seed(SEED)              # Switch to int_hash for geometry ops
    ...
```

### Pattern for transpiled GeoNodes scripts (e.g., corals, code_seed)

These scripts extract parameter values from Blender node trees. Random calls are minimal. Use `np.random.seed(FACTORY_SEED)` at the top if random sampling exists.

### Pattern for hand-written procedural scripts (e.g., creatures)

These scripts implement geometry procedurally. If they sample parameters from random distributions at the top level, they should use `np.random.seed(idx)` (raw factory seed), not `int_hash`.

## How to Verify Correctness

1. Run the standalone script and measure bounding box dimensions.
2. Run the infinigen factory with the same idx and compare dimensions.
3. If proportions differ significantly (e.g., H/W ratio off by >10%), the seed is likely wrong.

Verification command:
```bash
blender --background --python-expr "
import runpy, bpy
from mathutils import Vector
# ... clear scene, run script ...
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
# print bounding box dimensions
"
```

## Affected Factories (Audit Results)

| Category | `sample_params(seed)` bug | Status |
|---|---|---|
| appliances/DishwasherFactory (59 seeds) | Yes — used int_hash | Fixed |
| appliances/BeverageFridgeFactory (59 seeds) | Yes — used int_hash | Fixed |
| appliances/MicrowaveFactory (56 seeds) | Yes — used int_hash | Fixed |
| appliances/Monitor, Oven, TV | No `sample_params()` function | Not affected |
| All other categories (creatures, corals, etc.) | No `sample_params(seed)` pattern | Not affected |

## Reference: Seed Flow Diagram

```
generate_individual_assets.py
  │
  ├─ fac = Factory(idx)                    # factory_seed = idx
  │   └─ __init__(factory_seed=idx)
  │       └─ with FixedSeed(factory_seed)  # np.random.seed(idx) ← PARAMS HERE
  │           └─ sample_parameters()
  │
  └─ fac.spawn_asset(idx)
      └─ with FixedSeed(int_hash((factory_seed, idx)))  # ← GEOMETRY HERE
          └─ create_asset()
```

