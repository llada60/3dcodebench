#!/usr/bin/env python3
"""
viz_factory.py — Render standalone factory scripts and generate comparison grids.

All-in-one: discovery, Blender rendering, and grid composition in a single file.
Auto-discovers factories from objects_blender/{category}/{Factory}/{Factory}_{NNN}.py.
Reference images from output_factories/ shown side-by-side when available.

Usage:
    # List all discovered factories
    python viz_factory.py --list
    python viz_factory.py --list --category appliances

    # Streaming mode (recommended): render → compose grid → discard temp files
    # Only the final grid PNGs are saved — no per-seed view files on disk.
    python viz_factory.py --factory TVFactory --stream
    python viz_factory.py --category appliances --stream
    python viz_factory.py --factory BeetleFactory --seeds 0-9 --stream

    # Saved mode: render per-seed view files, then compose grids
    python viz_factory.py --factory TVFactory
    python viz_factory.py --category creatures --seeds 0-19 --workers 2

    # Rebuild grids from already-saved view files (no re-rendering)
    python viz_factory.py --only-grid --factory TVFactory

    # Render only, skip grid generation
    python viz_factory.py --only-render --factory TVFactory --seeds 0-4

    # Tune render quality
    python viz_factory.py --factory OvenFactory --stream --samples 64 --resolution 768

Output:
    renders/compare/{category}/{Factory}_grid_page01.png  (always)
    renders/compare/{category}/{Factory}/{Factory}_{NNN}/view_001..004.png  (saved mode only)

Requires:
    - Blender 4.2 at /path/to/blender-5.0/blender
    - PIL/Pillow (in conda env)
"""

import sys, os

# ═══════════════════════════════════════════════════════════════════════════════
# Blender render mode — entered when called as:
#   blender --background --python viz_factory.py -- --blender-render ...
# ═══════════════════════════════════════════════════════════════════════════════

_BL_ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

if "--blender-render" in _BL_ARGV:
    import math, argparse, re
    import bpy
    from mathutils import Vector

    PALETTE = [
        (0.65, 0.06, 0.06),   # deep red
        (0.06, 0.30, 0.70),   # steel blue
        (0.75, 0.45, 0.0),    # amber
        (0.06, 0.50, 0.14),   # forest green
        (0.55, 0.08, 0.60),   # deep magenta
        (0.02, 0.55, 0.55),   # teal
        (0.72, 0.24, 0.04),   # burnt orange
        (0.35, 0.14, 0.65),   # indigo
    ]
    VIEWS = [(0,30,"view_001.png"), (90,30,"view_002.png"),
             (180,30,"view_003.png"), (270,30,"view_004.png")]

    # Per-category lighting/material overrides for better visibility.
    # Keys: category name → dict with optional overrides.
    _CATEGORY_OVERRIDES = {
        "deformed_trees": {
            "light_mult": 6.0,         # much brighter lights for bark
            "fill_mult": 6.0,          # strong fill for shadow areas
            "single_color": (0.72, 0.55, 0.40),  # warm bark tan
            "roughness": 0.72,         # organic bark look
            "alpha": 1.0,              # fully opaque
        },
        "trees": {
            "light_mult": 10.0,
            "fill_mult": 10.0,
            "single_color": (0.55, 0.72, 0.35),  # bright leaf green
            "roughness": 0.75,
            "alpha": 1.0,
        },
        "tropic_plants": {
            "light_mult": 8.0,
            "fill_mult": 8.0,
            "single_color": (0.50, 0.75, 0.30),  # tropical leaf green
            "roughness": 0.70,
            "alpha": 1.0,
        },
        "monocot": {
            "light_mult": 8.0,
            "fill_mult": 8.0,
            "single_color": (0.48, 0.70, 0.28),  # grass/leaf green
            "roughness": 0.72,
            "alpha": 1.0,
        },
        # Factory-specific overrides (checked via factory name in script path)
        "TreeFlowerFactory": {
            "light_mult": 0.4,
            "fill_mult": 0.3,
            "roughness": 0.55,
        },
        # Spiky fruits: the spike geometry creates heavy self-shadowing that
        # kills visibility under the default dark-red palette + 1× lighting.
        # Boost lights so ambient light fills in the deep valleys between
        # spikes; use warm tones matching the reference look, and lower
        # roughness slightly so spike tips catch a highlight. World strength
        # stays at 1× to keep the scene background dark for contrast.
        "FruitFactoryDurian": {
            "light_mult": 12.0,
            "fill_mult": 14.0,
            "single_color": (0.82, 0.72, 0.42),  # warm tan/green-brown
            "roughness": 0.55,
            "alpha": 1.0,
        },
        "FruitFactoryPineapple": {
            "light_mult": 12.0,
            "fill_mult": 14.0,
            "single_color": (0.88, 0.75, 0.45),  # warm yellow-beige
            "roughness": 0.55,
            "alpha": 1.0,
        },
        "decor": {
            # Glass-tank visualization: separate loose parts, apply low alpha to
            # the largest part (glass shell) so content inside is visible.
            "glass_tank": True,
            "glass_alpha": 0.05,       # nearly invisible glass walls
            "glass_roughness": 0.08,   # slight gloss
            "content_alpha": 0.98,     # fully opaque content inside
            "light_mult": 1.5,
        },
    }

    # Season-specific tint multipliers applied to base tree color
    _TREE_SEASON_COLORS = {
        "summer": (0.45, 0.68, 0.22),  # vivid green
        "autumn": (0.82, 0.42, 0.15),  # orange/red
        "spring": (0.95, 0.75, 0.85),  # pink (flowers)
        "winter": (0.55, 0.42, 0.30),  # brown (bare trunk)
    }

    def _detect_tree_season(script_path):
        """Read the script file to extract SEASON = 'xxx' (for TreeFactory scripts)."""
        try:
            with open(script_path) as f:
                content = f.read(4096)  # header only
            m = re.search(r"SEASON\s*=\s*['\"]([a-z]+)['\"]", content)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    def _detect_category(script_path):
        """Detect category and factory name from script path.
        Returns category string. Factory-specific overrides are checked
        by looking for factory name in _CATEGORY_OVERRIDES."""
        parts = os.path.normpath(script_path).split(os.sep)
        category = None
        for i, p in enumerate(parts):
            if p in ("objects_blender", "objects_blender_code_seed",
                     "objects_blender_texture", "objects_blender_texture_code_seed"):
                if i + 1 < len(parts):
                    category = parts[i + 1]
                # Check for factory-specific override (e.g. TreeFlowerFactory)
                if i + 2 < len(parts):
                    factory = parts[i + 2]
                    if factory in _CATEGORY_OVERRIDES:
                        return factory
                # Trees category: detect season and return a virtual category key
                if category == "trees" and i + 2 < len(parts):
                    factory = parts[i + 2]
                    if factory == "TreeFactory":
                        season = _detect_tree_season(script_path)
                        if season:
                            return f"trees_{season}"
        return category

    # Auto-generate per-season tree overrides based on the base "trees" dict
    for _season, _color in _TREE_SEASON_COLORS.items():
        _base = dict(_CATEGORY_OVERRIDES.get("trees", {}))
        _base["single_color"] = _color
        _CATEGORY_OVERRIDES[f"trees_{_season}"] = _base

    def _bl_parse():
        p = argparse.ArgumentParser()
        p.add_argument("--blender-render", action="store_true")
        p.add_argument("--script", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--output", required=True)
        p.add_argument("--samples", type=int, default=64)
        p.add_argument("--resolution", type=int, default=512)
        p.add_argument("--keep-materials", action="store_true")
        return p.parse_args(_BL_ARGV)

    def _bl_clear():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)
        for c in list(bpy.data.curves): bpy.data.curves.remove(c)
        for ng in list(bpy.data.node_groups): bpy.data.node_groups.remove(ng)
        bpy.context.scene.cursor.location = (0, 0, 0)

    def _bl_run_script(path):
        import runpy
        runpy.run_path(path, run_name="__main__")
        for o in list(bpy.context.scene.objects):
            if o.type in ("CAMERA", "LIGHT"):
                bpy.data.objects.remove(o, do_unlink=True)
        return [o for o in bpy.context.scene.objects if o.type == 'MESH']

    def _make_mat(name, color, alpha, roughness=0.65, transparent_mix=False):
        """Create a material.

        If transparent_mix=True, use a Transparent+Diffuse MixShader instead of
        Principled BSDF Alpha.  This avoids blinding specular highlights in
        CYCLES — perfect for glass-tank visualization where the glass should be
        nearly invisible so interior content is clearly visible.
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        if transparent_mix:
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            transp = nt.nodes.new("ShaderNodeBsdfTransparent")
            transp.inputs["Color"].default_value = (1, 1, 1, 1)
            diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
            diff.inputs["Color"].default_value = (*color, 1)
            diff.inputs["Roughness"].default_value = roughness
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.inputs["Fac"].default_value = alpha  # alpha fraction is diffuse
            nt.links.new(transp.outputs[0], mix.inputs[1])
            nt.links.new(diff.outputs[0], mix.inputs[2])
            nt.links.new(mix.outputs[0], out.inputs[0])
        else:
            bsdf = nt.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (*color, 1)
                bsdf.inputs["Roughness"].default_value = roughness
                bsdf.inputs["Metallic"].default_value = 0.0
                bsdf.inputs["Alpha"].default_value = alpha
        mat.use_backface_culling = False
        try: mat.surface_render_method = 'BLENDED'
        except: pass
        try: mat.blend_method = 'BLEND'
        except: pass
        return mat

    def _bl_color(objs, category=None):
        bpy.context.view_layer.update()
        ovr = _CATEGORY_OVERRIDES.get(category, {})

        # Glass-tank mode: separate loose parts, apply near-invisible alpha to the
        # largest part (outer shell) and opaque alpha to interior content.
        if ovr.get("glass_tank"):
            glass_alpha   = ovr.get("glass_alpha", 0.10)
            content_alpha = ovr.get("content_alpha", 0.85)
            # Separate by loose parts (always, regardless of vertex count)
            for b in list(objs):
                for o in bpy.context.scene.objects: o.select_set(False)
                b.select_set(True)
                bpy.context.view_layer.objects.active = b
                bpy.ops.mesh.separate(type="LOOSE")
            bpy.context.view_layer.update()
            objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            # NOTE: do NOT rejoin — the solidify glass creates 2 separate loose shells
            # (inner + outer cube surfaces), and cactus/mushroom add many tiny debris pieces.
            # Rejoining would collapse everything to one object getting glass material.

            def _bb_dims_vol(obj):
                bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
                xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
                dx = max(xs)-min(xs); dy = max(ys)-min(ys); dz = max(zs)-min(zs)
                return dx, dy, dz, dx*dy*dz

            # Classify by bounding-box volume (sorted descending):
            #   TOP 2  → glass shells (solidify on a closed cube = outer + inner surface)
            #   Flat+wide → belts (thin Z, XY footprint matches tank rim)
            #   Large enough (> 0.5% of glass vol) → content (colored)
            #   Tiny (< 0.5% of glass vol) → ignored (zero-volume debris from GeoNodes)
            dims_vols = [(obj,) + _bb_dims_vol(obj) for obj in objs]
            dims_vols.sort(key=lambda x: x[4], reverse=True)  # sort by volume desc
            glass_vol = dims_vols[0][4]
            gdx, gdy, gdz = dims_vols[0][1], dims_vols[0][2], dims_vols[0][3]
            GLASS_COLOR = (0.45, 0.65, 0.80)
            BELT_COLOR  = (0.25, 0.25, 0.35)
            glass_mat = _make_mat("glass", GLASS_COLOR, glass_alpha,
                                  roughness=0.8, transparent_mix=True)
            belt_mat  = _make_mat("belt", BELT_COLOR, 0.0, roughness=0.3,
                                  transparent_mix=True)  # invisible debris
            ci = 0
            for rank, (obj, dx, dy, dz, vol) in enumerate(dims_vols):
                if rank < 2:
                    # Top 2 largest = outer + inner glass shells from solidify
                    mat = glass_mat
                elif vol < glass_vol * 0.005:
                    # Near-zero volume debris → fully transparent (invisible)
                    mat = belt_mat
                else:
                    # Belt: flat frame (thin Z, wide XY footprint matching tank rim)
                    is_belt = (dz < 0.15 * gdz and dx * dy > 0.35 * gdx * gdy)
                    if is_belt:
                        mat = _make_mat("belt_vis", BELT_COLOR, min(content_alpha, 0.95), roughness=0.3)
                    else:
                        r, g, b = PALETTE[ci % len(PALETTE)]
                        mat = _make_mat(f"content_{ci}", (r, g, b), content_alpha)
                        ci += 1
                obj.data.materials.clear()
                obj.data.materials.append(mat)
            print(f"  [glass_tank] {len(dims_vols)} parts: glass×2, content×{ci}, glass_vol={glass_vol:.3f}")
            return objs

        # Single-color mode: skip separate-by-loose, assign one warm material to all
        if "single_color" in ovr:
            r, g, b = ovr["single_color"]
            alpha = ovr.get("alpha", 1.0)
            roughness = ovr.get("roughness", 0.65)
            mat = bpy.data.materials.new("viz_uniform")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
                bsdf.inputs["Roughness"].default_value = roughness
                bsdf.inputs["Metallic"].default_value = 0.0
                bsdf.inputs["Alpha"].default_value = alpha

            # Special-case: tree fruits get a distinct warm color
            fruit_mat = None
            if (category or "").startswith("trees"):
                fruit_color = (0.85, 0.45, 0.18)  # warm orange-brown
                fruit_mat = bpy.data.materials.new("viz_fruit")
                fruit_mat.use_nodes = True
                fbsdf = fruit_mat.node_tree.nodes.get("Principled BSDF")
                if fbsdf:
                    fbsdf.inputs["Base Color"].default_value = (*fruit_color, 1)
                    fbsdf.inputs["Roughness"].default_value = 0.55
                    fbsdf.inputs["Metallic"].default_value = 0.0

            for obj in objs:
                obj.data.materials.clear()
                if fruit_mat is not None and obj.name.startswith("TreeFruits"):
                    obj.data.materials.append(fruit_mat)
                else:
                    obj.data.materials.append(mat)
            return objs

        if len(objs) < 4 and sum(len(o.data.vertices) for o in objs) < 200_000:
            for b in list(objs):
                for o in bpy.context.scene.objects: o.select_set(False)
                b.select_set(True)
                bpy.context.view_layer.objects.active = b
                bpy.ops.mesh.separate(type="LOOSE")
            bpy.context.view_layer.update()
            objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            # If too many loose parts (e.g. feathers, legs), re-join to avoid material overload
            if len(objs) > 50:
                for o in bpy.context.scene.objects: o.select_set(False)
                for o in objs: o.select_set(True)
                bpy.context.view_layer.objects.active = objs[0]
                bpy.ops.object.join()
                bpy.context.view_layer.update()
                objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        objs.sort(key=lambda o: len(o.data.vertices), reverse=True)

        # Depth-based alpha: parts closer to the camera (smaller X = more "front")
        # get lower alpha (more transparent), revealing interior structure behind them.
        # alpha = lerp(ALPHA_FRONT, ALPHA_BACK, t) where t = normalised depth 0→1.
        # This is fully general: no per-factory knowledge needed.
        ALPHA_FRONT = 0.18   # most-forward part (e.g. glass door, screen)
        ALPHA_BACK  = 0.92   # most-rearward part (back wall, body)

        all_corners = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
        if all_corners:
            gxs = [v.x for v in all_corners]
            global_x_min = min(gxs)
            global_x_range = max(max(gxs) - global_x_min, 1e-6)
        else:
            global_x_min, global_x_range = 0, 1

        def _depth_alpha(obj):
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            center_x = sum(v.x for v in corners) / len(corners)
            t = max(0.0, min(1.0, (center_x - global_x_min) / global_x_range))
            return ALPHA_FRONT + (ALPHA_BACK - ALPHA_FRONT) * t

        for i, obj in enumerate(objs):
            r, g, b = PALETTE[i % len(PALETTE)]
            alpha = _depth_alpha(obj)
            mat = _make_mat(f"part_{i}", (r, g, b), alpha)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        return objs

    def _bl_setup(objs, res, samples, category=None):
        bpy.context.view_layer.update()
        corners = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
        if not corners: return None, None, None
        xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in corners])
        cx, cy, cz = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2
        extent = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) or 1.0
        for o in objs:
            if o.parent is None:
                o.location = o.location - Vector((cx, cy, cz))
        center = Vector((0, 0, 0))
        cam_r, cam_h = extent * 2.2, extent * 0.6

        ovr = _CATEGORY_OVERRIDES.get(category, {})
        lm = ovr.get("light_mult", 1.0)
        fm = ovr.get("fill_mult", 1.0)
        world_strength = ovr.get("world_strength", 1.0)
        world_color = ovr.get("world_color", (0.012, 0.013, 0.021))

        # World — slightly brighter for spiky subjects that suffer heavy
        # self-shadowing (controlled via `world_strength` override).
        w = bpy.context.scene.world or bpy.data.worlds.new("World")
        bpy.context.scene.world = w
        w.use_nodes = True; w.node_tree.nodes.clear()
        bg = w.node_tree.nodes.new("ShaderNodeBackground")
        out = w.node_tree.nodes.new("ShaderNodeOutputWorld")
        bg.inputs[0].default_value = (*world_color, 1); bg.inputs[1].default_value = world_strength
        w.node_tree.links.new(bg.outputs[0], out.inputs[0])

        # 3-point area lights (size scales with extent → fixed energy)
        def _lt(name, loc, energy, sz=1.0):
            bpy.ops.object.light_add(type="AREA", location=loc)
            lt = bpy.context.object; lt.name = name
            lt.data.energy = energy; lt.data.size = extent * sz
            lt.rotation_euler = (center - Vector(loc)).to_track_quat("-Z","Y").to_euler()
        _lt("Key",  (cam_r*0.9, -cam_r*0.7, cam_h*1.7), 600 * lm, 0.7)
        _lt("Fill", (-cam_r*0.6, -cam_r*0.4, cam_h),      60 * lm * fm, 0.9)
        _lt("Rim",  (0, cam_r*0.9, cam_h*1.3),            100 * lm, 0.5)

        # Camera
        bpy.ops.object.camera_add(location=(cam_r, 0, cam_h))
        cam = bpy.context.object; cam.name = "Cam"
        bpy.context.scene.camera = cam
        cam.data.clip_end = extent * 25; cam.data.lens = 50

        # Render — transparent film so the PNG has alpha=0 background. The
        # world emits light onto the object but is invisible in the saved
        # image (compositors / paper figures see only the geometry).
        sc = bpy.context.scene
        sc.render.resolution_x = sc.render.resolution_y = res
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.film_transparent = True
        gpu_ok = False
        try:
            pr = bpy.context.preferences.addons["cycles"].preferences
            for dev_type in ("OPTIX", "CUDA"):
                pr.compute_device_type = dev_type
                pr.get_devices()
                gpus = [d for d in pr.devices if d.type == dev_type]
                if gpus:
                    for d in pr.devices: d.use = (d.type == dev_type)
                    sc.render.engine = "CYCLES"; sc.cycles.device = "GPU"
                    sc.cycles.samples = samples; sc.cycles.use_denoising = True
                    print(f"[viz] GPU render: {dev_type} x{len(gpus)}")
                    gpu_ok = True; break
        except Exception as _gpu_err:
            print(f"[viz] GPU init error: {_gpu_err}")
        if not gpu_ok:
            print("[viz] Falling back to CPU Cycles")
            sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
            sc.cycles.samples = samples; sc.cycles.use_denoising = True
        return cam, center, cam_r

    def _bl_render(cam, center, cam_r, out_dir):
        sc = bpy.context.scene; os.makedirs(out_dir, exist_ok=True)
        for az_d, el_d, fn in VIEWS:
            az, el = math.radians(az_d), math.radians(el_d)
            cam.location = (cam_r*math.cos(el)*math.cos(az),
                            cam_r*math.cos(el)*math.sin(az), cam_r*math.sin(el))
            cam.rotation_euler = (Vector(center)-cam.location).to_track_quat("-Z","Y").to_euler()
            sc.render.filepath = os.path.join(out_dir, fn)
            try: bpy.ops.render.render(write_still=True)
            except RuntimeError as e:
                if "Out of memory" in str(e) and sc.render.engine == "CYCLES":
                    sc.render.engine = "BLENDER_EEVEE"
                    for a in ("taa_render_samples","samples"):
                        try: setattr(sc.eevee, a, 64); break
                        except AttributeError: pass
                    bpy.ops.render.render(write_still=True)
                else: raise

    # ── Blender main ──
    args = _bl_parse()
    os.environ["INFINIGEN_SEED"] = str(args.seed)
    _bl_clear()
    category = _detect_category(args.script)
    objs = _bl_run_script(args.script)
    if not objs: print("ERROR: no mesh"); sys.exit(1)
    if not args.keep_materials:
        objs = _bl_color(objs, category=category)
    print(f"  {len(objs)} parts, {sum(len(o.data.vertices) for o in objs)} verts (cat={category})")
    cam, center, cam_r = _bl_setup(objs, args.resolution, args.samples, category=category)
    if not cam: print("ERROR: setup failed"); sys.exit(1)
    _bl_render(cam, center, cam_r, args.output)
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration mode — normal Python (not inside Blender)
# ═══════════════════════════════════════════════════════════════════════════════

import re, shutil, subprocess, tempfile, argparse, multiprocessing, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE         = Path(__file__).resolve().parent.parent   # repo root
SCRIPTS_ROOT = BASE / "objects_blender"                 # overridden by --scripts-root
REF_ROOT     = BASE / "output_factories"                # overridden by --ref-root
OUT_ROOT     = BASE / "renders" / "compare"             # overridden by --out-root
BLENDER      = os.environ.get("VIZ_BLENDER", "/path/to/blender-5.0/blender")
THIS_SCRIPT  = str(Path(__file__).resolve())

# 40-frame turntable: camera fixed at az=315°, asset rotates N*9°/frame
# Effective view az = 315° - N*9°.  az=0°→f35, az=90°→f25, az=180°→f15, az=270°→f5
REF_FRAMES = ["Image_035.png", "Image_025.png", "Image_015.png", "Image_005.png"]
CODE_VIEWS = ["view_001.png",  "view_002.png",  "view_003.png",  "view_004.png"]

# Grid layout
THUMB=768; LABEL_W=110; GAP=32; BORDER=16; HEADER_H=56; ROW_PAD=8
N_COLS=4; ROW_H=THUMB+ROW_PAD
PAGE_W = BORDER + LABEL_W + N_COLS*THUMB + GAP + N_COLS*THUMB + BORDER

# Colors (text only — grid background is fully transparent)
CODE_HDR=(120,196,255); REF_HDR=(180,230,140); SEED_COL=(255,214,120)
MISSING_TXT=(160,80,80)
ALIGN_REF_EPS = 0.03
ALIGN_SZ = 160

# ── Fonts ──
def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except: pass
    return ImageFont.load_default()
FONT_S, FONT_H, FONT_M = _font(28), _font(22), _font(14)

# ── Discovery ──
def discover_categories():
    return sorted(p.name for p in SCRIPTS_ROOT.iterdir()
                  if p.is_dir() and not p.name.startswith("."))

def discover_factories(cat):
    d = SCRIPTS_ROOT / cat
    if not d.exists(): return []
    return [p.name for p in sorted(d.iterdir())
            if p.is_dir() and not p.name.startswith("_")
            and any(p.glob(f"{p.name}_[0-9][0-9][0-9].py"))]

def discover_seeds(cat, fac):
    return sorted(int(m.group(1))
                  for py in (SCRIPTS_ROOT/cat/fac).glob(f"{fac}_[0-9][0-9][0-9].py")
                  for m in [re.search(r"_(\d{3})\.py$", py.name)] if m)

def find_category(fac):
    for c in discover_categories():
        if fac in discover_factories(c): return c
    return None

# ── Blender subprocess ──
def _blender(script_path, seed, out_dir, samples, res, gpu_id=None, keep_materials=False):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [BLENDER, "--background", "--python", THIS_SCRIPT, "--",
           "--blender-render", "--script", str(script_path), "--seed", str(seed),
           "--output", str(out_dir), "--samples", str(samples), "--resolution", str(res)]
    if keep_materials:
        cmd.append("--keep-materials")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT {Path(script_path).stem}"); return False
    if r.returncode != 0:
        print(f"    ERROR {Path(script_path).stem}: {(r.stderr or '')[-300:]}"); return False
    return True

def _num_gpus():
    try:
        import subprocess as _sp
        out = _sp.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"],
                               text=True, stderr=_sp.DEVNULL)
        return len([l for l in out.strip().splitlines() if l.strip()])
    except Exception:
        return 0

_N_GPUS = _num_gpus()

# Per-worker GPU ID, set by pool initializer so each worker owns one GPU permanently
_WORKER_GPU_ID = None
_KEEP_MATERIALS = False

def _pool_init(gpu_list):
    """Pool initializer: assign a fixed GPU to each worker process."""
    global _WORKER_GPU_ID
    idx = multiprocessing.current_process()._identity[0] - 1  # 0-based
    _WORKER_GPU_ID = gpu_list[idx % len(gpu_list)] if gpu_list else None

def _make_pool(workers):
    gpu_list = list(range(_N_GPUS)) if _N_GPUS > 0 else []
    return multiprocessing.Pool(workers, initializer=_pool_init, initargs=(gpu_list,))

def render_one(args_tuple):
    cat, fac, seed, samples, res = args_tuple
    sp = SCRIPTS_ROOT/cat/fac/f"{fac}_{seed:03d}.py"
    if not sp.exists(): return seed, False
    od = OUT_ROOT/cat/fac/f"{fac}_{seed:03d}"
    if (od/"view_001.png").exists(): return seed, True
    od.mkdir(parents=True, exist_ok=True)
    return seed, _blender(sp, seed, od, samples, res, gpu_id=_WORKER_GPU_ID, keep_materials=_KEEP_MATERIALS)

def render_to_memory(cat, fac, seed, samples, res):
    sp = SCRIPTS_ROOT/cat/fac/f"{fac}_{seed:03d}.py"
    if not sp.exists(): return {}
    tmp = Path(tempfile.mkdtemp(prefix="viz_"))
    try:
        if not _blender(sp, seed, tmp, samples, res, gpu_id=_WORKER_GPU_ID, keep_materials=_KEEP_MATERIALS): return {}
        return {vf: Image.open(tmp/vf).convert("RGBA").resize((THUMB,THUMB), Image.LANCZOS)
                for vf in CODE_VIEWS if (tmp/vf).exists()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def render_factory(cat, fac, seeds, workers=1, samples=64, res=512):
    print(f"\n  Rendering {cat}/{fac}  ({len(seeds)} seeds, workers={workers}, gpus={_N_GPUS})")
    args = [(cat, fac, s, samples, res) for s in seeds]
    if workers == 1:
        results = [render_one(a) for a in args]
    else:
        with _make_pool(workers) as pool:
            results = pool.map(render_one, args)
    n = sum(ok for _,ok in results)
    print(f"  Done: {n}/{len(seeds)}"); return n

# ── Grid helpers ──
def _thumb(path):
    if not Path(path).exists(): return None
    try: return Image.open(path).convert("RGBA").resize((THUMB,THUMB), Image.LANCZOS)
    except: return None

def _miss(text="missing"):
    t = Image.new("RGBA",(THUMB,THUMB),(0,0,0,0)); d = ImageDraw.Draw(t)
    tw = d.textlength(text, font=FONT_M)
    d.text(((THUMB-tw)/2, THUMB//2-8), text, font=FONT_M, fill=(*MISSING_TXT,220)); return t

def _header(draw, has_refs):
    lbl = "Code Renders  (v1  v2  v3  v4)"; cx = BORDER+LABEL_W
    draw.text((cx+(N_COLS*THUMB-draw.textlength(lbl,font=FONT_H))/2,(HEADER_H-22)//2),
              lbl, font=FONT_H, fill=CODE_HDR)
    rl = "Infinigen Reference  (yaw-aligned)" if has_refs else "Reference  (n/a)"
    rx = cx+N_COLS*THUMB+GAP
    draw.text((rx+(N_COLS*THUMB-draw.textlength(rl,font=FONT_H))/2,(HEADER_H-22)//2),
              rl, font=FONT_H, fill=REF_HDR)

def _paste_alpha(canvas, tile, pos):
    """Paste RGBA tile using its alpha channel as mask, preserving transparency."""
    if tile is not None and tile.mode == "RGBA":
        canvas.paste(tile, pos, tile)
    else:
        canvas.paste(tile or _miss(), pos)

def _avg_border_rgb(tile, border=10):
    if tile is None:
        return (0, 0, 0)
    rgba = tile.convert("RGBA")
    w, h = rgba.size
    pts = []
    for y in range(h):
        for x in list(range(border)) + list(range(max(border, w-border), w)):
            pts.append(rgba.getpixel((x, y))[:3])
    for x in range(border, max(border, w-border)):
        for y in list(range(border)) + list(range(max(border, h-border), h)):
            pts.append(rgba.getpixel((x, y))[:3])
    if not pts:
        return (0, 0, 0)
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))

def _mask_code(tile):
    if tile is None:
        return None
    rgba = tile.convert("RGBA").resize((ALIGN_SZ, ALIGN_SZ), Image.LANCZOS)
    bg = _avg_border_rgb(rgba)
    out = Image.new("L", rgba.size, 0)
    src = rgba.load()
    dst = out.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, _ = src[x, y]
            d = math.sqrt((r - bg[0]) ** 2 + (g - bg[1]) ** 2 + (b - bg[2]) ** 2)
            dst[x, y] = 255 if d > 22 else 0
    return out.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MaxFilter(3))

def _mask_ref(tile):
    if tile is None:
        return None
    rgba = tile.convert("RGBA").resize((ALIGN_SZ, ALIGN_SZ), Image.LANCZOS)
    return rgba.getchannel("A").point(lambda p: 255 if p > 8 else 0).filter(ImageFilter.MaxFilter(3))

def _dice(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None
    a = mask_a.load()
    b = mask_b.load()
    w, h = mask_a.size
    inter = cnt_a = cnt_b = 0
    for y in range(h):
        for x in range(w):
            av = a[x, y] > 0
            bv = b[x, y] > 0
            inter += av and bv
            cnt_a += av
            cnt_b += bv
    if cnt_a + cnt_b == 0:
        return 1.0
    return 2 * inter / (cnt_a + cnt_b)

def _ref_tiles(ref_base):
    return [_thumb(ref_base/rf) for rf in REF_FRAMES]

def _best_ref_shift(views, ref_tiles):
    code_masks = [_mask_code(views.get(vf)) for vf in CODE_VIEWS]
    ref_masks = [_mask_ref(tile) for tile in ref_tiles]
    scores = []
    for shift in range(4):
        vals = []
        for i in range(4):
            score = _dice(code_masks[i], ref_masks[(i + shift) % 4])
            if score is not None:
                vals.append(score)
        scores.append(sum(vals) / len(vals) if vals else float("-inf"))
    best_shift = max(range(4), key=lambda s: scores[s])
    if best_shift == 0 or scores[best_shift] < 0:
        return 0
    if scores[best_shift] - scores[0] < ALIGN_REF_EPS:
        return 0
    return best_shift

def _row(canvas, draw, ri, seed, views, ref_tiles):
    y = HEADER_H + ri*ROW_H
    sl = f"{seed:03d}"
    draw.text((BORDER+(LABEL_W-draw.textlength(sl,font=FONT_S))/2,y+(ROW_H-28)//2),
              sl, font=FONT_S, fill=SEED_COL)
    x = BORDER+LABEL_W
    for vf in CODE_VIEWS:
        _paste_alpha(canvas, views.get(vf) or _miss("no render"), (x,y)); x += THUMB
    x = BORDER+LABEL_W+N_COLS*THUMB+GAP
    shift = _best_ref_shift(views, ref_tiles)
    for i in range(4):
        _paste_alpha(canvas, ref_tiles[(i + shift) % 4] or _miss("no ref"), (x,y)); x += THUMB

def _save_page(page_seeds, views_list, cat, fac, page_idx, has_refs):
    n = len(page_seeds); ph = HEADER_H + n*ROW_H + BORDER
    # Fully transparent canvas — empty cell areas remain alpha=0 in the PNG.
    # Header / label / separator strips drawn below stay opaque for readability.
    c = Image.new("RGBA",(PAGE_W,ph),(0,0,0,0)); d = ImageDraw.Draw(c)
    _header(d, has_refs)
    for ri, seed in enumerate(page_seeds):
        ref_base = REF_ROOT/cat/fac/f"{fac}_{seed:03d}"
        _row(c, d, ri, seed, views_list[ri], _ref_tiles(ref_base))
    od = OUT_ROOT/cat; od.mkdir(parents=True, exist_ok=True)
    out = od / f"{fac}_grid_page{page_idx+1:02d}.png"
    c.save(out, optimize=False); print(f"    -> {out.relative_to(BASE)}")

# ── Grid modes ──
def make_grids(cat, fac, seeds, per_page=10):
    has_refs = any((REF_ROOT/cat/fac/f"{fac}_{s:03d}"/REF_FRAMES[0]).exists() for s in seeds)
    pages = [seeds[i:i+per_page] for i in range(0,len(seeds),per_page)]
    print(f"  Grid: {cat}/{fac}  {len(seeds)} seeds -> {len(pages)} pages")
    for pi, ps in enumerate(pages):
        vl = [{vf: _thumb(OUT_ROOT/cat/fac/f"{fac}_{s:03d}"/vf) for vf in CODE_VIEWS} for s in ps]
        _save_page(ps, vl, cat, fac, pi, has_refs)

def _stream_one(args_tuple):
    cat, fac, seed, samples, res = args_tuple
    v = render_to_memory(cat, fac, seed, samples, res)
    return seed, v

def make_grids_stream(cat, fac, seeds, per_page=10, workers=1, samples=64, res=512):
    has_refs = any((REF_ROOT/cat/fac/f"{fac}_{s:03d}"/REF_FRAMES[0]).exists() for s in seeds)
    pages = [seeds[i:i+per_page] for i in range(0,len(seeds),per_page)]
    print(f"  Stream: {cat}/{fac}  {len(seeds)} seeds -> {len(pages)} pages  (workers={workers}, gpus={_N_GPUS})")
    for pi, ps in enumerate(pages):
        args = [(cat, fac, s, samples, res) for s in ps]
        if workers == 1:
            results = [_stream_one(a) for a in args]
        else:
            with _make_pool(workers) as pool:
                results = pool.map(_stream_one, args)
        vl = []
        for s, v in results:
            print(f"    {fac}_{s:03d} {'ok' if v else 'FAIL'}", flush=True)
            vl.append(v)
        _save_page(ps, vl, cat, fac, pi, has_refs)

# ── CLI ──
def _parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-",1); return list(range(int(lo),int(hi)+1))
    return [int(x) for x in s.split(",")]

def main():
    # Default workers: 3 per GPU gives ~2× speedup with low GPU contention.
    # User can override with --workers 1 for deterministic debugging.
    default_workers = max(1, _N_GPUS * 3) if _N_GPUS > 0 else 1

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category",    default=None)
    p.add_argument("--factory",     default=None)
    p.add_argument("--seeds",       default=None, help="e.g. 0-49 or 0,1,5")
    p.add_argument("--workers",     type=int, default=default_workers,
                   help=f"Parallel render workers (default: {default_workers} on {_N_GPUS} GPUs)")
    p.add_argument("--samples",     type=int, default=64)
    p.add_argument("--resolution",  type=int, default=768)
    p.add_argument("--per-page",    type=int, default=10)
    p.add_argument("--stream",      action="store_true", help="Only save grid PNGs, no per-seed files")
    p.add_argument("--only-grid",   action="store_true", help="Rebuild grids from saved renders")
    p.add_argument("--only-render", action="store_true", help="Render only, skip grids")
    p.add_argument("--list",        action="store_true", help="List factories and exit")
    p.add_argument("--scripts-root", default=None, help="Path to scripts dir (default: {repo}/objects_blender)")
    p.add_argument("--ref-root",    default=None, help="Path to reference renders (default: {repo}/output_factories)")
    p.add_argument("--out-root",    default=None, help="Path to output dir (default: {repo}/renders/compare)")
    p.add_argument("--keep-materials", action="store_true", help="Preserve factory materials instead of overriding with viz colors")
    a = p.parse_args()

    global SCRIPTS_ROOT, REF_ROOT, OUT_ROOT, _KEEP_MATERIALS
    if a.scripts_root:
        SCRIPTS_ROOT = Path(a.scripts_root).resolve()
    if a.ref_root:
        REF_ROOT = Path(a.ref_root).resolve()
    if a.out_root:
        OUT_ROOT = Path(a.out_root).resolve()
    _KEEP_MATERIALS = a.keep_materials

    if a.factory and not a.category:
        a.category = find_category(a.factory)
        if not a.category: print(f"ERROR: '{a.factory}' not found"); sys.exit(1)

    cats = [a.category] if a.category else discover_categories()
    work = []
    for c in cats:
        for f in ([a.factory] if a.factory else discover_factories(c)):
            ss = _parse_seeds(a.seeds) if a.seeds else discover_seeds(c, f)
            if ss: work.append((c, f, ss))

    if not work: print("No factories found."); sys.exit(0)
    if a.list:
        for c,f,ss in work: print(f"  {c}/{f}  ({len(ss)} seeds)")
        print(f"\n{len(work)} total"); return

    for c, f, ss in work:
        if a.stream:
            make_grids_stream(c, f, ss, a.per_page, a.workers, a.samples, a.resolution)
        else:
            if not a.only_grid:
                render_factory(c, f, ss, a.workers, a.samples, a.resolution)
            if not a.only_render:
                make_grids(c, f, ss, a.per_page)

    print(f"\nDone. Output: {OUT_ROOT}/")

if __name__ == "__main__":
    main()
