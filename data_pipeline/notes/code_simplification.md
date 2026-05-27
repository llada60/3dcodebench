# Factory Script Code Simplification Guide

> **Purpose**: Transform seed-variant factory scripts into high-quality, diverse Blender Python training data. Input scripts are organized as `{scripts_root}/{category}/{Factory}/{Factory}_{idx}.py`, where `{scripts_root}` may vary (e.g., `objects_blender/` or `objects_blender_code_seed/`).
>
> **Audience**: Coding agents (Gemini, Claude, etc.) that will process scripts one-at-a-time.
>
> **Golden Rule**: The 3D mesh output must be **identical** before and after every transformation. Verify vertex count and positions match within tolerance (1e-4).

---

## Overview: Three-Stage Pipeline

```
Stage 1: BAKE ✓ DONE  Stage 2: PRUNE         Stage 3: DIVERSIFY + SIMPLIFY
─────────────────     ─────────────────      ────────────────────────────────
Replace all random    Remove dead branches   Rewrite in unique coding style,
calls with concrete   and unused functions   simplify aggressively, vary
literal values        for this specific      naming/structure/API choices
                      seed's chosen path
```

Each stage preserves the 3D output exactly. Stage 1 is mechanical. Stage 2 requires understanding control flow. Stage 3 is creative but shape-preserving.

### Stage 1:

Stage 1 applied to scripts in `objects_blender_code_seed/`. All categories except the 4 slow Gray-Scott RD coral factories are fully baked. **Agents working on Stage 2/3 should use these already-baked scripts as input**, not the original `objects_blender/` templates.

**What the automated baking produced:**

1. All `np.random.*`, `rng.*`, `uniform()`, `normal()`, `randint()`, `choice()` calls replaced:
   - Single-call lines → inline literal: `w = 0.18473`
   - Multi-call lines (function called in a loop) → `_bake_Lxxx = iter([v1, v2, ...]); next(_bake_Lxxx)`
2. `SEED` and `FACTORY_SEED` constants removed
3. `sample_params()` function body replaced / removed

**What the automated baking LEFT in place (Stage 2 must clean up):**

- `FixedSeed` class definition + `with FixedSeed(...):` usage — still syntactically present but has no effect (all random calls inside are now baked)
- `int_hash()`, `md5_hash()`, `hashlib` — only used by FixedSeed
- Thin wrapper functions like `def log_uniform(lo, hi): return np.exp(next(_bake_Lxx))`
- `from numpy.random import normal, uniform, randint` — these functions are no longer called
- Module-level `_bake_Lxxx = iter([...])` declarations (inserted before function defs)
- The factory class scaffolding (e.g., `class MonocotGrowthFactory`) — leave unless simplifying to flat function

**The `_bake_L` pattern:**
```python
# Module level (before function def, for multi-call sites):
_bake_L517 = iter([0.012415, 0.012415, 0.012415, 0.012415])

def build_leaf(...):
    ...
    width = leaf_width + float(next(_bake_L517))  # same value used 4 times here
```

```python
# Inside function body (for loops within the function):
def make_seed_particles(rng, count):
    _bake_L767 = iter([0.900, 0.366, 0.890, 1.038, ...])  # 60 values for 60 seeds
    for i in range(count):
        sx = float(next(_bake_L767))
```

### Recommended Processing Mode: Per-Seed Agent

Process each seed variant script **individually** with a coding agent, not via batch generation scripts. A single agent reads the original factory script, understands its geometry construction, computes the exact baked values for that seed, and produces one simplified/diversified output.

**Why not batch generation?** A batch script that generates all 60 variants from a single template amplifies any formula error across every seed:
- Hollow cube walls using `W` instead of `W-2*dt` → all 60 scripts have overlapping geometry
- Handle standoff at `length+width` instead of `length` → all 60 scripts misplace the handle
- These bugs are hard to catch visually because the overlapping parts are hidden inside the joined mesh

**Per-seed agent advantages:**
- The agent reads and understands the ORIGINAL verified factory script each time
- Formula derivations are traced from the actual code, not re-implemented in a separate template
- Errors are isolated to one seed, not propagated to all 60
- Style diversity emerges naturally from independent agent runs
- The agent can verify its output by running the script and comparing vertex counts

---

## Before Processing Any Script

Before making any changes, read the source factory file end-to-end and identify:

0. **Seeding calls checking** Count the **total number of random values produced at runtime** (not just the number of source lines with `np.random.*`). Loop bodies multiply: a single `rng.uniform()` inside `for bi in range(40): for fi in range(20):` produces 800 values at runtime. If the total runtime count exceeds ~100, do **not** bake — keep the random seed instead, because baking would store hundreds or thousands of literal values, hurting code readability far more than it helps. If the total is a few tens of values or fewer, bake them.

   Decision rule in practice:
   - Any `np.random.*` / `rng.*` call inside a loop → count it as `call × loop_iterations`. If that product alone exceeds ~100, the whole script is KEEP SEED.
   - Flat (non-loop) random calls are cheap to bake regardless of count.
   - When in doubt, prefer KEEP SEED over baking with ugly arrays.

   **KEEP SEED scripts — hard constraint for Stage 3:** If a script was processed as KEEP SEED (i.e., it has `np.random.seed(literal_N)` at the top but still contains `np.random.*` / `rng.*` calls throughout), then Stage 3 **must not touch any random call**. Specifically:
   - Do **NOT** add `_bake_Lxxx = iter([...])` / `next(_bake_Lxxx)` patterns.
   - Do **NOT** pre-run the RNG and cache its outputs in any form.
   - Do **NOT** replace `rng.normal(...)`, `rng.uniform(...)`, `np.random.choice(...)`, or any other sampling call with a literal or an iterator.
   - The only permitted changes are: renaming functions/variables, adding docstrings, restructuring control flow, adjusting whitespace/comments.

   Violation example (WRONG — must never appear in a KEEP SEED script after Stage 3):
   ```python
   rng = np.random.default_rng(543568399 + 50)
   _bake_L306 = iter([-0.18778, 0.0024091, ...])   # ← FORBIDDEN
   ...
   rot_x = float(next(_bake_L306))                  # ← FORBIDDEN
   ```

   Correct form (unchanged rng call in KEEP SEED script):
   ```python
   rng = np.random.default_rng(543568399 + 50)
   ...
   rot_x = float(rng.normal(0, 0.2))               # ← correct, leave as-is
   ```

1. **Seeding mechanism** — look for `SEED`, `FACTORY_SEED`, `np.random.seed()`, `int_hash()`, `FixedSeed`, `get_state()`/`set_state()`, or an explicit `RandomState` instance. This determines how to compute baked values and whether multiple independent RNG streams exist.

2. **Hard dependencies** — check imports for `scipy`, `trimesh`, `bmesh`, `mathutils`. These cannot be removed even after baking random calls, because they are used for geometry construction (BSpline evaluation, convex hull, mesh editing).

3. **Branching type** — distinguish between Python `if/else` branches (Stage 2 prunable) and GeoNodes switch nodes (not prunable at the Python level — the node graph itself must be simplified, or left as-is).

4. **Array-structured random data** — if random values populate a `np.array()` used for NURBS control points or curve data, bake the random VALUES inside the array but preserve the `np.array()` structure. The shape and type are required by the geometry construction code.

5. **Non-deterministic operations** — cloth simulation and text-to-mesh conversion produce slightly different vertex counts across Blender versions or runs. Accept ±5 vertex tolerance for scripts containing these operations.

6. **Custom math helpers** — identify functions like `log_uniform`, `clip_gaussian`, `wrap_gaussian`, `quadratic_interp`, `catmull_rom_eval` that are not random calls but are used in geometry computation. Keep these unless fully inlined.

---

## Stage 1: Numerical Baking

### Goal
Eliminate ALL randomness. Every `np.random.*` call becomes a literal value. Remove `np.random.seed()`, `SEED`, `FACTORY_SEED`, and `sample_params()`.

### Baking Rules

| Random Call | Baking Rule | Before | After |
|---|---|---|---|
| `np.random.normal(mu, sigma)` | float, ~5 significant figures | `1 + np.random.normal(0, 0.1)` | `1.0051` |
| `np.random.uniform(a, b)` | float, ~5 significant figures | `np.random.uniform(0.05, 0.1)` | `0.072346` |
| `np.random.randint(a, b)` | exact integer | `np.random.randint(2, 4)` | `3` |
| `np.random.choice(list)` | exact chosen value | `np.random.choice(["a","b"])` | `"a"` |
| `bool(uniform() < threshold)` | exact bool | `bool(np.random.uniform() < 0.4)` | `True` |
| `np.random.normal(size=N)` | literal array, each elem ~5 sig figs | `np.random.normal(1, 0.1, 3)` | `[0.98424, 1.0051, 0.97823]` |
| `np.random.uniform(size=(n,m))` | literal nested list, each elem ~5 sig figs | | `[[0.12345, 0.34562], ...]` |
| `log_uniform(a, b)` | float, ~5 significant figures | `log_uniform(0.005, 0.01)` | `0.0070926` |
| `clip_gaussian(mu, sigma, lo, hi)` | final returned value, ~5 sig figs | `clip_gaussian(1.75, 0.75, 0.9, 3)` | `1.8234` |
| `np.random.dirichlet(alpha)` | literal array, each elem ~5 sig figs | | `[0.31423, 0.45214, 0.23363]` |
| `wrap_gaussian(mu, sigma, lo, hi)` | float, ~5 significant figures | | `0.67123` |

### Precision Rule: Preserve Topology, Minimize Decimals

**Principle**: Round each baked value to the **shortest representation** that preserves the exact 3D output — same vertex count, same topology, no visible geometric shifts. Precision is not just about spatial error tolerance; it depends on how each value is used downstream.

**Three sensitivity levels exist for baked values:**

1. **Position/size values** (box location, scale): Tolerant — even 1mm error is invisible. These can often use 3–4 decimal places.
2. **Values in arithmetic chains** (derived positions computed from multiple params via add/multiply): Rounding errors accumulate across the chain, so each input needs more relative precision than the final tolerance suggests.
3. **Topology-sensitive values** (anything that feeds into `floor()`, `int()`, loop counts, Boolean modifiers, bevel, bridge_edge_loops): Even tiny rounding can change the integer result or modifier output, producing a completely different mesh. These need the most precision.

Since you often cannot easily tell which category a value belongs to (a base parameter may flow into all three), the safest approach is to maintain sufficient **relative** precision uniformly.

**Empirical finding** (tested on 60 MonitorFactory seeds with uniform and adaptive rounding):
- Fixed 2 significant figures: **always breaks** geometry
- Fixed 4 decimal places: fails ~5% of seeds (modifier-sensitive cases)
- Fixed 5 decimal places: fails ~2% (still insufficient for some small values)
- **5 significant figures (adaptive)**: **100% match** — the only strategy that passed all seeds
- Fixed 6 decimal places: fails ~2% — paradoxically worse than 5 sig figs, because for small values (e.g., 0.008) 6dp gives only ~4 sig figs

The key insight is that **adaptive** rounding (significant figures) outperforms **uniform** rounding (fixed decimal places). Significant figures keep relative precision constant across all magnitudes — large values get fewer decimals, small values get more:
```
1.2345678  → 1.2346    (4 decimal places)
0.076328   → 0.076329  (6 decimal places)
0.0082091  → 0.0082091 (7 decimal places)
```

This matters because small values (radii, margins, thickness factors) tend to participate in the most sensitive computation chains — they get multiplied, divided, or passed into modifiers where relative error is what counts.

**Practical recommendation**: Start from full-precision values (as computed by `sample_params`), then reduce decimal places per-value while verifying vertex count after each change. Use ~5 significant figures as a reasonable starting point. The goal is the shortest literal numbers that still produce identical geometry — this keeps the code clean and natural-looking without introducing errors.

**Exception**: Integers are always exact (e.g., `randint(2,4)` → `3`).

#### Why 2 Significant Figures Violates the Tolerance

All geometry values are in **meters**. The problem with 2 sig figs is that it rounds to a *relative* precision (±5% of the value), so two values of similar magnitude can round in opposite directions by amounts far exceeding 1 mm:

```python
# Exact values for BeverageFridge seed 4:
D        = 1.0051   →  sig2 = 1.0    (rounds DOWN: 10.051 → 10)
door_cx  = 1.0547   →  sig2 = 1.1    (rounds UP:   10.547 → 11)

# Gap between body right face and door left face:
door_cx - dt/2 - D = 1.1 - 0.099/2 - 1.0 = 0.05 m = 5 cm  ← FAR EXCEEDS 1mm TOLERANCE!
```

With 4 decimal places (error < 0.05 mm), values stay consistent:
```python
D        = 1.0051   →  4dp = 1.0051
door_cx  = 1.0547   →  4dp = 1.0547

# Gap with 4dp:
1.0547 - 0.0993/2 - 1.0051 = 0.0000  ← well within tolerance
```

#### Compute Derived Values from EXACT Params, Then Round

```python
# ✓ CORRECT: compute from exact params, round final result
D, W, H, dt = compute_params(seed)   # exact floats
door_cx = round(D + dt/2, 4)         # round the FINAL derived value

# ✗ WRONG: round base params first, then compute positions
D_r  = round(D, 2)     # → 1.0  (wrong!)
dt_r = round(dt, 2)    # → 0.099
door_cx = D_r + dt_r/2 # → 1.0495 ≠ 1.0547 (already wrong, and will round further)
```

#### Math Constants: Use Symbolic Form
Replace all literal pi values with `math.pi`:
- `1.5707963...` → `math.pi / 2`
- `3.14159265...` → `math.pi`
- `6.28318530...` → `2 * math.pi`

Always ensure `import math` is present when using `math.pi`.

### How to Compute Baked Values

To get the correct baked value for a specific seed variant:

```python
# Run the original script's sample_params with the correct seed
import numpy as np, hashlib

def int_hash(x, max_val=(2**32 - 1)):
    m = hashlib.md5()
    for s in x:
        m.update(str(s).encode("utf-8"))
    return abs(int(m.hexdigest(), 16)) % max_val

# For FACTORY_SEED scripts (Type A: appliances, some bathroom):
#   seed = idx  (raw index)
# For SEED scripts (Type B: everything else):
#   seed = int_hash((idx, idx))

# Then call sample_params(seed) and read all values
```

**CRITICAL**: You MUST run the original script to get exact values. Do NOT manually compute `np.random.normal()` — the RNG state depends on the exact sequence of calls.

**Per-script processing**: Each seed variant is a separate, independent task. The agent reads the original factory script, bakes values for that one seed, and produces one output file. Do not attempt to write a single script that processes or generates multiple seeds at once — per-seed isolation is required for both correctness and style diversity.

### Tricky Patterns

#### Pattern 1: Data-Dependent Control Flow
Some factories branch on a random value, and each branch consumes different subsequent random calls. The RNG sequence diverges.

```python
# TVFactory example:
has_depth_extrude = bool(np.random.uniform() < 0.4)  # consumes 1 call
if has_depth_extrude:
    depth_extrude = depth * np.random.uniform(2, 5)    # consumes 1 MORE call
# If has_depth_extrude is False, this uniform() is never called,
# so all subsequent random values shift!
```

**Rule**: Always run the FULL `sample_params()` with the correct seed. Never try to compute values piecemeal.

#### Pattern 2: Hash-Based Seeding
Some factories re-seed the RNG partway through using `int_hash()`:

```python
np.random.seed(int_hash((SEED, 0)))   # first sub-seed
# ... generate body params ...
np.random.seed(int_hash((SEED, 1)))   # second sub-seed
# ... generate limb params ...
```

**Rule**: Each `int_hash` re-seed creates an independent RNG stream. All values from each stream must be captured separately.

#### General Rule for All Other RNG Complexity

Many factories use additional patterns that make the RNG state non-trivial to reason about manually: state save/restore (`get_state`/`set_state`), retry loops (`clip_gaussian` calls normal up to 21 times), `FixedSeed` context managers, variable loop counts, multi-dimensional arrays, and explicit `RandomState` instances.

**Rule for all of these**: Do NOT attempt to manually trace the RNG state. Always run the full original script with the correct seed and capture the final values. The source of complexity is irrelevant — only the output values matter.

#### Discarded Values Still Consume RNG State

Some scripts generate a random value and then overwrite it, or call a function whose return value is unused. The random call still advances the RNG state, so skipping it shifts all subsequent values. Always run the full function — never skip calls that seem unused.

### What to Remove After Baking

1. `SEED = ...` and `FACTORY_SEED = ...` constants
2. `np.random.seed(...)` calls
3. `sample_params()` function definition (inline the baked values)
4. `int_hash()`, `md5_hash()`, `FixedSeed` class definitions (if no longer used)
5. `log_uniform()`, `clip_gaussian()`, `wrap_gaussian()` helper definitions
6. `import hashlib` (if int_hash removed)
7. `import numpy as np` — **only if numpy is COMPLETELY unused** after baking. Check for: `np.array`, `np.linalg`, `np.clip`, `np.cross`, `np.dot`, `np.linspace`, `np.zeros`, `np.concatenate`, etc.

---

## Stage 2: Dead Code Elimination

### Goal
Remove all code paths that are NOT executed for this specific seed's parameter values. After baking, branching variables become constants — trace the execution path and delete everything else.

### How to Identify Branches

After Stage 1, every former random variable is now a constant. Scan the baked script for constants used in `if/else` or string comparisons — these are dead branch candidates.

**Structural branch** (worth eliminating): the baked value selects an object TYPE, routing execution into geometrically distinct code paths (different functions, different mesh construction). Each dead path typically includes one or more function definitions that become entirely unused.

**Dimensional variation** (skip Stage 2): the baked value only scales or positions geometry — every code path still executes. No elimination needed.

If a script has no constant-gated `if/else` after baking, skip Stage 2 entirely.

### Baked Boolean Elimination (CRITICAL)

When a script has baked boolean constants like `has_extrude = True` or `has_stand = False`, ALL conditional blocks using them must be resolved:

```python
# BEFORE (has_extrude = True, has_stand = False):
has_extrude = True
has_stand = False

if has_base:
    obj = make_base()
else:
    obj = make_bowl()   # dead code if has_base=True

if has_extrude:
    extrude_back(obj)   # always runs

if has_stand:
    obj = add_stand(obj)  # never runs

# AFTER:
obj = make_base()
extrude_back(obj)
# add_stand removed entirely — AND its function definition too
```

**Rules**:
1. If `has_X = True`: remove the `if has_X:` guard, keep the body, delete any `else:` branch
2. If `has_X = False`: remove the entire `if has_X:` block (including body), keep any `else:` branch
3. After removing dead branches, also remove the function DEFINITIONS that are no longer called (e.g., `def add_stand()` if `has_stand = False`)
4. Remove the boolean constant declarations themselves (`has_stand = False` line)
5. Also resolve string-type branches: if `sink_type = 'drop-in'`, remove code for 'vessel' and 'undermount'
6. Remove `if obj:` / `if body:` / `if rack:` guards where the function always returns non-None
7. Please don't include any "seed" or "idx" in the comments.

### Additional Dead Code to Remove

After branch elimination:
1. **Unused parameter keys**: If `p["single_leg_w_fac"]` is never used, remove it from the dict
2. **The parameter dict itself**: After baking + pruning, `sample_params()` is gone. Don't create a replacement dict — inline values directly where used
3. **Conditional guards on deterministic values**: `if body: parts.append(body)` where body is always non-None → remove the `if`, keep `parts.append(body)`
4. **Unused imports**: After removing code, check if `math`, `numpy`, `hashlib` are still needed
5. **Boolean constant declarations**: Remove `has_extrude = True` — the value is already inlined

### Cleanup Specific to Automated-Baked Scripts

When processing a script from `objects_blender_code_seed/` (Stage 1 already done):

**A. Remove seed infrastructure imports:**
```python
# REMOVE (no longer called):
from numpy.random import normal, uniform, randint, choice
```

**B. Remove FixedSeed blocks (dedent body):**
```python
# BEFORE:
with FixedSeed(int_hash(("collection", i))):
    obj = build_fn(i, **kwargs)
    coll.objects.link(obj)

# AFTER:
obj = build_fn(i, **kwargs)
coll.objects.link(obj)
```

**C. Remove FixedSeed class, int_hash, md5_hash, hashlib:**
After all `with FixedSeed(...)` blocks are dedented, delete the class definition
and its helpers. Also remove `import random` and `import hashlib` if only used
by FixedSeed.

**D. Inline thin wrappers:**
```python
# BEFORE:
_bake_L41 = iter([3.9134, -2.0348, -1.9530, -2.2739])
def log_uniform(low, high):
    return np.exp(next(_bake_L41))
# ...called 4 times at: scale = log_uniform(0.5, 2.0)

# AFTER (precompute exp):
_bake_L41 = iter([49.93, 0.1311, 0.1418, 0.1034])
# ...call site:
scale = next(_bake_L41)
```

**E. Constant iterator collapse:**
When a `_bake_Lxxx` iterator has all identical values, replace with a single literal:
```python
# BEFORE:
_bake_L517 = iter([0.012415, 0.012415, 0.012415, 0.012415])
# ...all 4 next() calls get 0.012415

# AFTER:
# (remove _bake_L517 declaration)
# ...all 4 call sites use: 0.012415
```
Only safe when all values differ by < 1e-6.

---

## Stage 3: Code Diversity & Simplification

### Goal
Rewrite each script in a **unique** coding style while keeping the 3D output identical. This is critical for training data quality — if all 60 seeds produce structurally identical code, the dataset has low diversity.

### 3.1 Simplification Rules (Apply to ALL scripts)

These simplifications preserve the 3D output:

#### Inline Single-Use Variables
```python
# Before:
dt = 0.072
rack_w = D - dt * 2.1
rack = make_rack(rack_w, ...)

# After (if dt only used here):
rack = make_rack(0.85 - 0.072 * 2.1, ...)
# Or pre-compute: 0.85 - 0.15 = 0.70
rack = make_rack(0.70, ...)
```

#### Pre-Compute Derived Constants
When all inputs are now literal, compute the result:
```python
# Before:
D = 0.85; W = 1.1; dt = 0.072
top = gn_cube(size=(D + dt, W, dt), pos=(0, 0, H))

# After:
top = gn_cube(size=(0.92, 1.1, 0.072), pos=(0, 0, 1.0))
```

#### Collapse Trivial Wrappers
```python
# Before:
def gn_cube(size, pos):
    sx, sy, sz = size
    px, py, pz = pos
    return box(sx, sy, sz, (sx*0.5+px, sy*0.5+py, sz*0.5+pz))

top = gn_cube(size=(0.92, 1.1, 0.072), pos=(0, 0, 1.0))

# After (inline the wrapper):
top = box(0.92, 1.1, 0.072, (0.46, 0.55, 1.0))
```

#### Merge Sequential Transforms
```python
# Before:
handle.rotation_euler = (0, math.pi/2, 0)
apply_tf(handle, rot=True)
handle.rotation_euler = (-math.pi/2, 0, 0)
apply_tf(handle, rot=True)
handle.location = (1.0, 0.11, 0.90)
apply_tf(handle, loc=True)

# After (combine into single rotation + location):
handle.rotation_euler = (combined_x, combined_y, combined_z)
handle.location = (1.0, 0.11, 0.90)
apply_tf(handle, loc=True, rot=True)
```

#### Remove Redundant None Checks
After baking, if a function always returns a valid object:
```python
# Before:
body = gn_hollow_cube(...)
if body:
    parts.append(body)

# After (body is always non-None with these params):
parts.append(gn_hollow_cube(...))
```

#### Remove Unnecessary Comments
```python
# Before:
# ======== 1. Body: hollow cube ========
# Size=(Depth, Width, Height), Thickness=DoorThickness
body = gn_hollow_cube(size=(0.85, 1.1, 1.0), ...)

# After (code is self-evident):
body = gn_hollow_cube(size=(0.85, 1.1, 1.0), ...)
```

### 3.2 Coding Style Rules

**Each seed variant must look structurally distinct from all others for the same factory.** The agent processes each script independently — read the current seed's code, rewrite it freshly, and do not copy the structure of previously written variants. The goal is for the 60 outputs to read as if written by 60 different programmers.

**Rules**:

1. **Meaningful names**: All function and variable names must be semantically descriptive. No cryptic abbreviations (`ds`, `sa`, `bx`, `tf`, `jn`). No single-letter names except `i`/`j` in loops and `x`/`y`/`z` for coordinates. Diversity in naming means choosing *different* meaningful names, not shorter ones (`deselect_all` vs `clear_selection` vs `reset_selection` — all valid, all different).

2. **Structural variety**: Vary the overall code structure across seeds. Some scripts use flat procedural sequences with no helpers; others decompose into small functions; others use a class or data table with a generic builder loop. No two scripts for the same factory should have the same structure.

3. **API variety**: Vary how geometry is created (`bpy.ops`, `bpy.data`, or `bmesh` direct construction), how transforms are applied, and how objects are joined.

4. **Computation style**: Some scripts pre-compute all derived values into literals; others keep symbolic expressions (e.g., `depth / 2`); others use a mix.

5. **Loop vs unroll**: Some scripts loop over repeated geometry; others unroll loops into explicit calls.

6. **Code organization**: Vary whether the script uses a `main()` function, `build()` call, top-level execution, or `if __name__ == "__main__":` guard.

7. **Comment style**: Vary freely — no comments, inline comments, section headers, or docstrings on functions.

---

## Lessons Learned

### Never Regex-Rename Blender API Calls
The `join` helper function and `bpy.ops.object.join()` share the word "join". A naive regex rename of `join` → `merge_objects` will break `bpy.ops.object.merge_objects()` (which doesn't exist).
- **Rule**: Only rename functions that are defined with `def funcname(` in the same file
- **Rule**: Never rename inside `bpy.ops.*`, `bpy.context.*`, `bpy.data.*` chains

### Don't Strip Imports Blindly
Always check which imports are actually used before removing. Baking random calls does not mean `bmesh`, `math`, or `numpy` are unused — they may be needed for geometry construction.

### Base Factory Files Are Read-Only
Files like `BeverageFridgeFactory.py` (without `_NNN` suffix) are the TEMPLATES — never modify them. Only modify the seed variant files (`_000.py` through `_059.py`).

---

## Verification Protocol

After transforming each script, verify the 3D output matches:

### Quick Check (Per-Script)
```bash
# 1. Run original script and count geometry
blender --background --python original_script.py

# 2. Run transformed script and count geometry
blender --background --python transformed_script.py

# 3. Compare vertex and face counts — must match exactly
```

### Thorough Check (Spot-Check a Few Per Category)
```python
# Export vertex positions from both scripts, compare:
import numpy as np
# ... run both scripts, collect vertex coords ...
assert np.allclose(verts_original, verts_transformed, atol=1e-4)
```

### Visual Check
Use `data_pipeline_operators/viz_factory.py` to render both original and transformed scripts. Compare the 4-view images visually. Any visible difference means the transformation is wrong.

### Common Failure Modes
1. **Wrong seed value** — completely different shape. Always verify seed formula.
2. **Missing random call** — RNG sequence shifts, all subsequent values wrong.
3. **Modifier order changed** — different mesh topology.
4. **Transform apply order changed** — different vertex positions.
5. **bmesh face winding changed** — normals flipped, visual artifacts.
6. **Integer truncation** — `int(2.9)` = 2, not 3. Be careful with `int(uniform(...))`.
7. **GeoNodes input socket order** — wrong values routed to wrong inputs in Blender 5.0.

---

## Workflow for Coding Agent

### When working from already-baked scripts (objects_blender_code_seed/)

Stage 1 is pre-done. Start from Step 4:

1. **Read** the baked script end-to-end. Note: `_bake_Lxxx` iterators replace all random calls; `FixedSeed`/`int_hash` are present but harmless.
2. **Run** the baked script in Blender to confirm it executes and produces a mesh.
3. **Record** the baseline vertex count and (optionally) vertex positions for verification.
4. **Apply Stage 2**: Remove FixedSeed, int_hash, thin wrappers, dead branches (see "Cleanup Specific to Automated-Baked Scripts").
5. **Apply Stage 3**: Rewrite in a unique coding style (see Style rules).
6. **Verify** vertex count and positions match the baseline from Step 3.
7. **Save** the transformed script, overwriting `{Factory}_{idx}.py`.

### When working from original templates (objects_blender/ — legacy workflow)

1. **Read** the script end-to-end. Identify: seeding mechanism, all random calls, branching points.
2. **Run** the original script in Blender to get baseline vertex count.
3. **Compute** all baked values by executing `sample_params()` (or equivalent) with the correct seed.
4. **Apply Stage 1**: Replace every random call with its baked literal value. Remove seed infrastructure.
5. **Apply Stage 2**: Trace the execution path. Delete unreachable branches and their helper functions.
6. **Verify** vertex count matches original after Stages 1+2.
7. **Apply Stage 3**: Rewrite the script in a unique coding style distinct from other seeds for the same factory.
8. **Verify** vertex count matches original after Stage 3.
9. **Save** the transformed script, overwriting the original `{Factory}_{idx}.py` file.

---

## Per-Category Reference

Quick reference for each of the 21 categories in `objects_blender_code_seed/`. For each category:
- **Approach** — primary geometry technique
- **Hard deps** — imports that CANNOT be removed even after baking (needed for geometry)
- **Stage 2 branching** — known variable-type branching (deadcode-eligible after baking)
- **Gotchas** — known tricky patterns or API issues

### appliances (6 factories, 360 scripts)
- **Approach**: bmesh direct mesh construction; GeoNodes for MonitorFactory screen
- **Hard deps**: `bmesh`, `math`; `numpy` for array operations
- **Stage 2 branching**: TVFactory had `leg_type` branching (2-leg vs single); OvenFactory had `use_gas` branching (gas grate vs electric heater). In baked scripts these are resolved constants — trace the if/else and remove dead branch.
- **Gotchas**: BeverageFridgeFactory has rack grid with variable loop count (`rack_h_amount`, `rack_w_amount`) — these are baked integers so the loop runs a fixed number of times; no special handling needed.

### bathroom (5 factories, 300 scripts)
- **Approach**: GeoNodes (NodeWrangler) for sinks and fixtures; bmesh for simpler geometry
- **Hard deps**: `bpy`, `mathutils`; `numpy` for normal computation
- **Stage 2 branching**: BathroomSinkFactory and StandingSinkFactory have `sink_type` branching (drop-in / undermount / vessel). Remove the unused type's geometry construction blocks and their helper functions.
- **Gotchas**: GeoNodes input ordering may differ between Blender versions — use `input_kwargs` with socket names, not positional args.

### cactus (7 factories, 420 scripts)
- **Approach**: GeoNodes for shape deformation (columnar, globular, prickly pear base); bmesh for stem assembly
- **Hard deps**: `bpy`, `mathutils`
- **Stage 2 branching**: `XxxBaseCactusFactory` vs `XxxCactusFactory` pattern — Base generates the mesh; the non-Base version adds colored segments/bumps. In baked scripts there is no type selection; each factory is its own script.
- **Gotchas**: PrickyPearCactusFactory uses a paddle-chain algorithm sensitive to the baked branch angle.

### clothes (6 factories, 360 scripts)
- **Approach**: bpy.ops mesh operations, cloth/physics simulation, particle-based surface noise
- **Hard deps**: `bpy` only; no scipy or bmesh needed
- **Stage 2 branching**: None — clothing shapes are purely parametric (no type selection)
- **Gotchas**: Cloth simulation and particle noise produce slightly different vertex counts across Blender versions or run order. Accept ±5% vertex count tolerance for these scripts. Do NOT try to bake cloth simulation results.

### corals (25 factories, 1500 scripts)
- **Approach**: Varies heavily by factory type:
  - **GeoNodes + bmesh**: BushCoral, Elkhorn, Star, Table, Tube, Twig, StarBase
  - **bmesh only**: BushBase, LeatherBase, Leather, Honeycomb, DiffGrowth, FanCoral
  - **Laplacian growth** (slow, ~2 min/script): CauliflowerBase, CauliflowerCoral, FanBase
  - **Gray-Scott reaction-diffusion** (very slow, ~10 min/script): BrainBase, BrainCoral (both embed full RD simulation), HoneycombBase, ReactionDiffusionBase — **NOT YET BAKED**
  - **DiffGrowth**: iterative edge subdivision/attraction simulation, ~30 sec/script
- **Hard deps**: `bmesh`, `mathutils`; some use `scipy.spatial` for voxel grid
- **Stage 2 branching**: None within a single coral type; the type selection happens at a higher level (which factory class to use)
- **Gotchas**: BushCoralFactory may generate very large meshes (1.5M+ verts for some seeds) — this is normal. The 4 RD coral factories are excluded from baking due to timeout; their scripts in `objects_blender_code_seed/` contain unbaked `np.random.*` calls.

### creatures (14 factories, 840 scripts)
- **Approach**:
  - **NURBS/scipy**: Beetle, Bird, FlyingBird, Fish — spline-based body from control points using scipy BSpline evaluation
  - **bmesh + voxel remesh**: Carnivore, Herbivore, Snake, Lizard, Frog, Chameleon — ellipsoid+tube primitives, voxel-remeshed to smooth surface
  - **Limb skeleton + voxel**: Crab, Lobster, SpinyLobster, Crustacean, Dragonfly
  - **Ribbon mesh**: Jellyfish
- **Hard deps**: `scipy` CANNOT be removed for NURBS creatures; `numpy`, `mathutils`
- **Stage 2 branching**: BirdFactory has per-seed wing-fold/leg-style branching; FishFactory has fin-count selection. In baked scripts these are resolved — prune unused fin/wing code paths.
- **Gotchas**: DragonflyFactory: `phases` array must be `np.array([...])`, not a scalar. Dragonfly/Bird/Lobster require long exec time (~60s) — these were baked with extended timeouts. BeetleFactory uses `int_hash((SEED, 0))` sub-seeding for different body regions.

### decor (1 factory, 60 scripts)
- **Approach**: GeoNodes for water wave animation (AquariumTankFactory)
- **Hard deps**: `numpy`, GeoNodes node tree
- **Stage 2 branching**: None
- **Gotchas**: AquariumTankFactory seed 040 has a face-index out-of-range bug (a face index captured during baking exceeds the mesh face count at runtime). This is a known limitation — skip seed 040 or regenerate it.

### deformed_trees (4 factories, 240 scripts)
- **Approach**: GeoNodes + bmesh for trunk/branch geometry; splines for curve-based branches
- **Hard deps**: `bmesh`, `mathutils`, `numpy`
- **Stage 2 branching**: None — each factory (Fallen, Hollow, Rotten, Truncated) is a distinct type with its own geometry approach
- **Gotchas**: RottenTreeFactory uses Boolean modifiers — use `mod.solver = "FLOAT"` (not `"EXACT"`) for Blender 5.0 stability (EXACT solver destroys voxel-remeshed geometry). FallenTreeFactory uses CurveToMesh nodes; ensure `CurveCircle.inputs['Radius']` is set directly for correct tube width in Blender 5.0.

### elements (17 factories, 1020 scripts)
- **Approach**: DoorFactory uses GeoNodes; staircases and rack/rug use bpy.ops; PalletFactory uses bmesh
- **Hard deps**: `bpy`, `mathutils` for door GeoNodes; no hard deps for most staircase scripts
- **Stage 2 branching**: DoorFactory had `door_type` × `frame_style` branching (panel / lite / louvered × flat / cased). In baked scripts, prune the unused door type's construction function.
- **Gotchas**: Staircase factories (6 variants) are largely non-branching — just parametric geometry. NatureShelfTrinketsFactory assembles collections of small objects (books, vases, rocks) — loop count is baked.

### fruits (8 factories, 480 scripts)
- **Approach**: GeoNodes (Pineapple, Durian use node-driven surface displacement); bmesh + splines for Apple, Coconut, etc.
- **Hard deps**: `bmesh`, `mathutils`
- **Stage 2 branching**: None — each fruit is a separate factory
- **Gotchas**: `mesh.calc_normals()` is removed in Blender 5.0 — use `mesh.update()` instead. This affects Blackberry, Durian, Strawberry which use bmesh normal computation.

### grassland (5 factories, 300 scripts)
- **Approach**: bmesh + bpy.ops; FlowerFactory uses GeoNodes for petal deformation
- **Hard deps**: `bmesh`, `mathutils`
- **Stage 2 branching**: FlowerFactory has petal count variation (not type branching); FlowerPlantFactory assembles seed + petal rings
- **Gotchas**: FlowerPlantFactory retains `seed_rng = np.random.default_rng(int(rng.integers(...)))` and `petal_rng = np.random.default_rng(int(rng.integers(...)))` — these are dead code (the created sub-RNGs are never used; their outputs are already in `_bake_L` iterators). Safe to remove in Stage 2.

### lamp (5 factories, 300 scripts)
- **Approach**: bmesh direct construction for all lamp types (ceiling, desk, floor, standing)
- **Hard deps**: `bmesh`, `mathutils`, `math`
- **Stage 2 branching**: LampFactory had a shade-style selection; DeskLampFactory had arm-joint count variation. In baked scripts these are resolved integers/constants — remove dead geometry construction.
- **Gotchas**: LampFactory uses cylindrical UV unwrapping which fails silently if mesh topology changes — do not alter winding order during Stage 3 rewrites.

### leaves (7 factories, 420 scripts)
- **Approach**: GeoNodes for wave/curl deformation (NodeWrangler); splines for midrib and veins; bmesh for leaf blade polygon
- **Hard deps**: `mathutils`, `numpy` for control points; no scipy
- **Stage 2 branching**: None — leaf shape is purely parametric
- **Gotchas**: LeafFactoryWrapped embeds large inline string constants (`_MAPLE_MODULE`, `_BROADLEAF_MODULE`, `_GINKO_MODULE`) that contain Python code — do not mistake these string contents for live code. The bake tool already handles these correctly via `_STRING_ASSIGN_PAT`. `LeafFactoryV2` uses a Bezier-based midrib quite different from other variants.

### mollusk (17 factories, 1020 scripts)
- **Approach**: All use scipy.interpolate for spiral parametric surface (BSpline, splprep, splev, CubicSpline)
- **Hard deps**: `scipy` CANNOT be removed — it performs the spline surface evaluation that creates the spiral shell mesh
- **Stage 2 branching**: None within each factory — each mollusk type (Conch, Auger, Clam, Scallop, Mussel, Nautilus, Snail, Volute, Shell) is a separate factory. `XxxBaseFactory` generates the raw mesh; `XxxFactory` uses the base and may add surface details.
- **Gotchas**: All mollusk shapes depend on precise spline control point arrays (baked from `np.random.normal`) — round these to at least 5 significant figures. `SpiralFactory` inner class or `create_asset()` may call `scipy` multiple times; preserve all spline evaluation calls.

### monocot (16 factories, 960 scripts)
- **Approach**: GeoNodes + bmesh + splines for grass-like plants (grasses, kelp, reed, wheat, taro, etc.)
- **Hard deps**: `bmesh`, `mathutils`, `numpy` for curve evaluation
- **Stage 2 branching**: None within each factory — each monocot type is its own factory
- **Gotchas**: GrassesMonocotFactory uses per-blade instance transforms applied via `bmesh`; KelpMonocotFactory creates long ribbon geometry sensitive to the baked blade-count integer.

### rocks (4 factories, 240 scripts)
- **Approach**: GeoNodes displacement textures (Voronoi, Clouds) applied to subdivided mesh; `bpy.ops.mesh.primitive_ico_sphere_add` base
- **Hard deps**: None beyond `bpy` — all displacement is done via GeoNodes texture nodes
- **Stage 2 branching**: None — displacement parameters are purely scalar
- **Gotchas**: BlenderRockFactory uses a `LAYERS` table (list of tuples) defining stacked displacement passes — this is a baked data structure, not branching code. Voronoi texture hashing changed in Blender 5.0 — if verifying against Blender 4.4 output, expect visual differences (same topology, different surface noise pattern).

### trees (3 factories, 180 scripts)
- **Approach**: bmesh for branch/trunk geometry; curve-based branching recursion
- **Hard deps**: `bmesh`, `mathutils`, `math`
- **Stage 2 branching**: None — tree branching depth and angle are baked scalars
- **Gotchas**: Trees generate recursive geometry — loop counts (branch counts per level) are baked integers. Ensure these match exactly or vertex counts diverge significantly. TreeFactory calls BranchFactory internally.

### tropic_plants (6 factories, 360 scripts)
- **Approach**: bmesh + splines for fronds/leaves (CoconutTree, PalmTree); bmesh for trunk geometry
- **Hard deps**: `bmesh`, `mathutils`, `math`
- **Stage 2 branching**: None — frond geometry is parametric
- **Gotchas**: PalmTreeFactory uses a 3-part structure (trunk + sheaths + fan crown); the crown is built as a fan of leaflets around a central point. Frond curvature is defined by baked spline control points — preserve the `np.array([[...]])` structure even after baking values inside it. CoconutTreeFactory: leaf cross-section width uses `CurveToMesh` — apply the Blender 5.0 `CurveCircle.inputs['Radius']` fix.

### underwater (2 factories, 120 scripts)
- **Approach**: bmesh for both SeaweedFactory and UrchinFactory
- **Hard deps**: `bmesh`, `mathutils`
- **Stage 2 branching**: None
- **Gotchas**: UrchinFactory creates a base icosphere then extrudes spines as individual tube meshes from face centers — the spine count and positions are determined by baked integer parameters. SeaweedFactory uses bmesh curve path + extrusion.

### wall_decorations (5 factories, 300 scripts)
- **Approach**: bmesh direct mesh construction for all (Balloon, Mirror, RangeHood, WallArt, WallShelf)
- **Hard deps**: `bmesh`, `math`
- **Stage 2 branching**: WallArtFactory had artwork-type branching (painting / pattern / text). In baked scripts, prune unused art-type construction code.
- **Gotchas**: RangeHoodFactory uses boolean modifiers for vent cutout — apply the `mod.solver = "FLOAT"` fix for Blender 5.0 if boolean results look wrong.

### windows (1 factory, 60 scripts)
- **Approach**: bmesh direct mesh construction
- **Hard deps**: `bmesh`, `math`
- **Stage 2 branching**: WindowFactory originally had `has_shutter` and `has_curtain` boolean gates. In baked scripts these are resolved — remove shutter/curtain construction code (and their function definitions) if the baked booleans are False.
- **Gotchas**: WindowFactory generates multi-object output (frame + glass + shutters/curtains) — the final join step depends on which components were built. After Stage 2 pruning, verify the join list only includes actually-constructed parts.
