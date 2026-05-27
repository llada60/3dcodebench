#!/usr/bin/env python3
"""
==============================================================================
Blender Parts-Colored Turntable GIF Renderer
==============================================================================

Renders standalone _bpy.py scripts as 360-degree turntable GIFs with per-part
coloring.  Each distinct mesh part gets a vivid research-style color from a
5-color palette (coral red, emerald teal, amber gold, deep violet, steel blue)
matching the ASTRA balloon visualization style.

Output per script:
    - 1 turntable .gif (no intermediate PNG files kept)
    - 8 static .png images from octant viewpoints (saved in <stem>_rendered_images/)

The 8 octant views cover both horizontal and vertical angles:
    - Lower ring (elevation 30deg): azimuth 0, 90, 180, 270
    - Upper ring (elevation 60deg): azimuth 45, 135, 225, 315

Usage:
    # Activate conda environment first (needs Pillow + numpy)
    conda activate infinigen

    # Render a single script
    python renderer.py /path/to/script.py -o ./renders

    # Render all scripts in a directory (recursive)
    python renderer.py /path/to/objects_blender/ -o ./renders

    # With custom settings
    python renderer.py /path/to/dir -o ./renders --frames 48 --fps 15

    # Multiple seeds
    python renderer.py /path/to/script.py -o ./renders -s 3

Options:
    target_path            .py file or directory to scan recursively
    -o, --output_dir       Output directory for GIF files (default: ./renders)
    -s, --seeds            Seeds per script (default: 1)
    -n, --frames           Turntable frames (default: 36)
    --fps                  GIF playback speed (default: 12)
    --resolution           Render resolution in pixels (default: 640)

Per-part coloring:
    - Scripts that produce multiple mesh objects get one color per object
    - Single-object scripts are auto-separated by LOOSE mesh islands
    - 5-color research palette: coral red, emerald teal, amber gold,
      deep violet, steel blue (cycles if more parts)

Requires: Blender 5.0+, Pillow, numpy
==============================================================================
"""

import sys
import os
import argparse
import subprocess
import tempfile
import shutil
import glob as glob_mod
from pathlib import Path
from datetime import datetime

# ── Detect if running inside Blender ──────────────────────────────────────────
try:
    import bpy
    IS_BLENDER = (
        hasattr(bpy, "app")
        and bpy.app.binary_path != ""
        and "python" not in Path(bpy.app.binary_path).name.lower()
    )
except ImportError:
    IS_BLENDER = False


# =============================================================================
#  BLENDER-SIDE: runs inside `blender --background --python THIS -- ...`
# =============================================================================
if IS_BLENDER:
    import math
    import runpy
    from mathutils import Vector

    # Research palette (ASTRA style) — 5 vivid, high-contrast hues
    PALETTE = [
        (0.92, 0.20, 0.12),  # coral red
        (0.04, 0.50, 0.30),  # emerald teal
        (0.95, 0.48, 0.00),  # amber gold
        (0.38, 0.08, 0.70),  # deep violet
        (0.08, 0.32, 0.80),  # steel blue
    ]

    def clean_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        for m in list(bpy.data.meshes):
            bpy.data.meshes.remove(m)
        for c in list(bpy.data.curves):
            bpy.data.curves.remove(c)
        for ng in list(bpy.data.node_groups):
            bpy.data.node_groups.remove(ng)
        bpy.context.scene.cursor.location = (0, 0, 0)

    def execute_target_script(script_path, seed):
        os.environ["INFINIGEN_SEED"] = str(seed)
        saved_argv = sys.argv[:]
        sys.argv = [script_path]
        try:
            runpy.run_path(script_path, run_name="__main__")
        finally:
            sys.argv = saved_argv

    def separate_and_color():
        """Separate mesh by LOOSE islands if needed, assign palette colors."""
        # Remove cameras/lights the target script may have added
        for obj in list(bpy.context.scene.objects):
            if obj.type in ("CAMERA", "LIGHT"):
                bpy.data.objects.remove(obj, do_unlink=True)

        bpy.context.view_layer.update()
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not mesh_objs:
            return []

        # Auto-separate by LOOSE if few objects and manageable vertex count
        LOOSE_VERT_LIMIT = 200_000
        total_verts = sum(len(o.data.vertices) for o in mesh_objs)
        if len(mesh_objs) < 4 and total_verts < LOOSE_VERT_LIMIT:
            for base_obj in list(mesh_objs):
                bpy.ops.object.select_all(action="DESELECT")
                base_obj.select_set(True)
                bpy.context.view_layer.objects.active = base_obj
                bpy.ops.mesh.separate(type="LOOSE")
            bpy.context.view_layer.update()
            mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        elif total_verts >= LOOSE_VERT_LIMIT:
            print(f"  Skipping LOOSE separation: {total_verts} verts exceeds limit")

        # Sort largest-first so dominant parts get the most distinct palette slots
        mesh_objs.sort(key=lambda o: len(o.data.vertices), reverse=True)

        # Assign palette colors
        for i, obj in enumerate(mesh_objs):
            r, g, b = PALETTE[i % len(PALETTE)]
            mat = bpy.data.materials.new(f"part_{i}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.40
                bsdf.inputs["Metallic"].default_value = 0.05
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        return mesh_objs

    def setup_scene(mesh_objs, resolution):
        """Center objects, create 3-point lighting, camera, and render settings."""
        bpy.context.view_layer.update()
        corners = [
            o.matrix_world @ Vector(c)
            for o in mesh_objs
            for c in o.bound_box
        ]
        if not corners:
            return None, None, None, None

        xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in corners])
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if extent == 0:
            extent = 1.0

        # Center at origin
        for obj in mesh_objs:
            if obj.parent is None:
                obj.location = obj.location - Vector((cx, cy, cz))

        center = Vector((0, 0, 0))
        cam_r = extent * 1.8
        cam_h = extent * 0.60

        # World — moderate IBL so vivid colors stay bright
        world = bpy.context.scene.world or bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs[1].default_value = 0.20

        # 3-point lighting (ASTRA style energy levels)
        def _light(name, loc, energy, sz=1.0):
            bpy.ops.object.light_add(type="AREA", location=loc)
            lt = bpy.context.object
            lt.name = name
            lt.data.energy = energy
            lt.data.size = extent * sz
            lt.rotation_euler = (
                center - Vector(loc)
            ).to_track_quat("-Z", "Y").to_euler()

        _light("Key",  (cam_r * 0.9, -cam_r * 0.7, cam_h * 1.7), 1200, 0.9)
        _light("Fill", (-cam_r * 0.6, -cam_r * 0.4, cam_h * 1.0), 400, 1.2)
        _light("Rim",  (0.0, cam_r * 0.9, cam_h * 1.3), 600, 0.7)

        # Camera
        bpy.ops.object.camera_add(location=(cam_r, 0, cam_h))
        cam = bpy.context.object
        cam.name = "TurntableCam"
        bpy.context.scene.camera = cam
        cam.data.clip_end = extent * 25
        cam.data.lens = 50

        # Render settings
        scene = bpy.context.scene
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = True

        # GPU Cycles if available, EEVEE fallback
        if not _setup_gpu_cycles(scene):
            scene.render.engine = "BLENDER_EEVEE"
            for attr in ("taa_render_samples", "samples"):
                try:
                    setattr(scene.eevee, attr, 64)
                    break
                except AttributeError:
                    pass

        return cam, center, cam_r, cam_h

    def _setup_gpu_cycles(sc):
        try:
            prefs = bpy.context.preferences
            cprefs = prefs.addons["cycles"].preferences
            cprefs.compute_device_type = "CUDA"
            cprefs.get_devices()
            gpu_names = []
            for d in cprefs.devices:
                if d.type == "CUDA":
                    d.use = True
                    gpu_names.append(d.name)
            if not gpu_names:
                return False
            sc.render.engine = "CYCLES"
            sc.cycles.device = "GPU"
            sc.cycles.samples = 128
            sc.cycles.use_denoising = True
            sc.cycles.use_adaptive_sampling = True
            sc.cycles.adaptive_threshold = 0.01
            print(f"  GPU Cycles enabled: {gpu_names}")
            return True
        except Exception as e:
            print(f"  GPU setup failed ({e}), using EEVEE")
            return False

    # 8 octant viewpoints: (azimuth_deg, elevation_deg, label)
    OCTANT_VIEWS = [
        (  0, 30, "front_low"),
        ( 90, 30, "right_low"),
        (180, 30, "back_low"),
        (270, 30, "left_low"),
        ( 45, 60, "front_right_high"),
        (135, 60, "back_right_high"),
        (225, 60, "back_left_high"),
        (315, 60, "front_left_high"),
    ]

    def render_turntable(cam, center, cam_r, cam_h, output_dir, n_frames):
        scene = bpy.context.scene
        os.makedirs(output_dir, exist_ok=True)
        for i in range(n_frames):
            angle = 2 * math.pi * i / n_frames
            cam.location = (
                cam_r * math.cos(angle),
                cam_r * math.sin(angle),
                cam_h,
            )
            cam.rotation_euler = (
                center - cam.location
            ).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(output_dir, f"frame_{i:04d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"  frame {i + 1:2d}/{n_frames}")

    def render_multiview(cam, center, cam_r, output_dir):
        """Render 8 octant views covering horizontal + vertical angles."""
        scene = bpy.context.scene
        os.makedirs(output_dir, exist_ok=True)
        for az_deg, el_deg, label in OCTANT_VIEWS:
            az = math.radians(az_deg)
            el = math.radians(el_deg)
            r = cam_r
            cam.location = (
                r * math.cos(el) * math.cos(az),
                r * math.cos(el) * math.sin(az),
                r * math.sin(el),
            )
            cam.rotation_euler = (
                center - cam.location
            ).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(output_dir, f"{label}.png")
            bpy.ops.render.render(write_still=True)
            print(f"  multiview: {label} (az={az_deg} el={el_deg})")

    # ── Blender entry point ───────────────────────────────────────────────────
    def blender_main():
        argv = sys.argv
        if "--" not in argv:
            sys.exit("No '--' separator found in argv")
        args = argv[argv.index("--") + 1:]
        if len(args) < 3:
            sys.exit(
                "Usage: blender --bg --python THIS "
                "-- <script> <frames_dir> <n_frames> [resolution] [seed] [multiview_dir]"
            )

        script_path = args[0]
        frames_dir = args[1]
        n_frames = int(args[2])
        resolution = int(args[3]) if len(args) > 3 else 640
        seed = int(args[4]) if len(args) > 4 else 0
        multiview_dir = args[5] if len(args) > 5 else None

        clean_scene()
        print(f"Executing: {script_path} (seed={seed})")
        execute_target_script(script_path, seed)

        mesh_objs = separate_and_color()
        if not mesh_objs:
            sys.exit("ERROR: no mesh objects after script execution")

        total_verts = sum(len(o.data.vertices) for o in mesh_objs)
        print(f"  {len(mesh_objs)} parts, {total_verts} total verts")

        cam, center, cam_r, cam_h = setup_scene(mesh_objs, resolution)
        if cam is None:
            sys.exit("ERROR: could not setup camera")

        render_turntable(cam, center, cam_r, cam_h, frames_dir, n_frames)

        if multiview_dir:
            render_multiview(cam, center, cam_r, multiview_dir)

        print("Blender-side done.")

    blender_main()


# =============================================================================
#  CLI-SIDE: runs outside Blender, orchestrates rendering + GIF assembly
# =============================================================================
else:
    import numpy as np
    from PIL import Image

    BLENDER = os.environ.get(
        "BLENDER", "/path/to/blender-5.0/blender"
    )

    def render_script_frames(script_path, frames_dir, n_frames, resolution, seed,
                             multiview_dir=None):
        """Launch Blender subprocess to render turntable frames + multiview."""
        cmd = [
            BLENDER, "--background",
            "--python", __file__,
            "--",
            str(script_path), frames_dir, str(n_frames),
            str(resolution), str(seed),
        ]
        if multiview_dir:
            cmd.append(multiview_dir)
        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        return result.returncode == 0

    def _to_gif_frame(img_rgba):
        """Convert RGBA PIL image to palette mode with index-0 transparency."""
        img_rgb = img_rgba.convert("RGB")
        img_p = img_rgb.quantize(
            colors=255, method=Image.Quantize.MEDIANCUT, dither=0
        )
        arr = np.array(img_p, dtype=np.uint8)
        arr += 1  # shift palette indices so index 0 is free for transparency
        alpha = np.array(img_rgba.split()[3], dtype=np.uint8)
        arr[alpha < 128] = 0
        pal = img_p.getpalette()
        result = Image.fromarray(arr, mode="P")
        result.putpalette([0, 0, 0] + pal[:255 * 3])
        return result

    def frames_to_gif(frames_dir, gif_path, fps=12):
        """Assemble PNG frames into a transparent animated GIF, then delete frames."""
        paths = sorted(glob_mod.glob(os.path.join(frames_dir, "frame_*.png")))
        if not paths:
            print(f"  No frames found in {frames_dir}")
            return False

        duration_ms = int(1000 / fps)
        frames = [_to_gif_frame(Image.open(p).convert("RGBA")) for p in paths]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            transparency=0,
            disposal=2,
            optimize=False,
        )
        size_kb = os.path.getsize(gif_path) // 1024
        print(f"  GIF: {gif_path} ({len(frames)} frames, {size_kb} KB)")

        # Clean up temporary frame PNGs
        shutil.rmtree(frames_dir, ignore_errors=True)
        return True

    def main():
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "target_path",
            help="Path to a .py script or a directory of scripts",
        )
        parser.add_argument("-o", "--output_dir", default="./renders")
        parser.add_argument("-s", "--seeds", type=int, default=1)
        parser.add_argument("-n", "--frames", type=int, default=36)
        parser.add_argument("--fps", type=int, default=12)
        parser.add_argument("--resolution", type=int, default=640)
        args = parser.parse_args()

        target = Path(args.target_path).resolve()
        output_dir = Path(args.output_dir).resolve()

        if not target.exists():
            sys.exit(f"Target not found: {target}")

        # Collect scripts
        if target.is_file():
            scripts = [target] if target.suffix == ".py" else []
        else:
            scripts = sorted(target.rglob("*.py"))
            scripts = [s for s in scripts if not s.name.startswith("__")]

        if not scripts:
            sys.exit("No matching .py files found.")

        total = len(scripts) * args.seeds
        print(f"Found {len(scripts)} script(s), {args.seeds} seed(s) each "
              f"-> {total} GIF(s)")
        print(f"Blender:    {BLENDER}")
        print(f"Output dir: {output_dir}")
        print(f"Frames: {args.frames}  FPS: {args.fps}  "
              f"Resolution: {args.resolution}x{args.resolution}\n")

        output_dir.mkdir(parents=True, exist_ok=True)
        passed = 0
        failed = []

        idx = 0
        for script in scripts:
            # Preserve directory structure in output
            try:
                rel = script.relative_to(
                    target if target.is_dir() else target.parent
                )
            except ValueError:
                rel = Path(script.name)

            for seed_offset in range(args.seeds):
                idx += 1
                seed = seed_offset

                gif_dir = output_dir / rel.parent
                gif_dir.mkdir(parents=True, exist_ok=True)
                gif_name = f"{script.stem}_seed{seed}.gif"
                gif_path = gif_dir / gif_name

                # 8-view images directory: <stem>_rendered_images/
                multiview_dir = gif_dir / f"{script.stem}_seed{seed}_rendered_images"

                # Temp directory for turntable frames (cleaned up after GIF)
                frames_dir = tempfile.mkdtemp(
                    prefix=f"frames_{script.stem}_s{seed}_"
                )

                print(f"\n[{idx}/{total}] {rel} (seed={seed})")
                ok = render_script_frames(
                    script, frames_dir, args.frames, args.resolution, seed,
                    multiview_dir=str(multiview_dir),
                )
                if ok:
                    if frames_to_gif(frames_dir, str(gif_path), fps=args.fps):
                        passed += 1
                    else:
                        failed.append((str(rel), seed, "GIF assembly failed"))
                        shutil.rmtree(frames_dir, ignore_errors=True)
                else:
                    failed.append((str(rel), seed, "Blender render failed"))
                    shutil.rmtree(frames_dir, ignore_errors=True)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  Results: {passed}/{total} succeeded")
        if failed:
            print(f"  Failed ({len(failed)}):")
            for name, seed, reason in failed:
                print(f"    {name} seed={seed}: {reason}")
        print(f"{'=' * 60}")

        # Write error log if any failures
        if failed:
            log_path = output_dir / "render_errors.log"
            with open(log_path, "w") as f:
                f.write(f"Render Error Log  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write("=" * 60 + "\n\n")
                for name, seed, reason in failed:
                    f.write(f"{name} seed={seed}: {reason}\n")
            print(f"\n  Error log: {log_path}")

    main()
