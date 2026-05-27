#!/usr/bin/env python3
"""
Blender Script Validator
========================

Recursively checks that all Python files in a directory can be executed by
Blender in background mode AND produce non-empty mesh output.

Two phases per script:
  1. Execute the script inside `blender --background --python`
  2. Check that at least one MESH object exists with > 0 vertices

Usage:
    # Validate all scripts in a directory
    python validate_blender_scripts.py /path/to/scripts/

    # Validate a single script
    python validate_blender_scripts.py /path/to/script.py

    # With custom Blender path and parallelism
    python validate_blender_scripts.py /path/to/scripts/ --blender /usr/bin/blender -j 4

    # Only show failures
    python validate_blender_scripts.py /path/to/scripts/ --failures-only

Options:
    target              .py file or directory to validate recursively
    --blender PATH      Path to Blender binary (default: env BLENDER or bundled 5.0)
    -j, --jobs N        Parallel workers (default: 1)
    -t, --timeout SECS  Per-script timeout in seconds (default: 120)
    --failures-only     Only print failed scripts
    -o, --output LOG    Write results to a log file
"""

import sys
import os
import argparse
import subprocess
import json
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── Detect if running inside Blender ────────────────────────────────────────
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
#  BLENDER-SIDE: runs inside `blender --background --python THIS -- <script>`
# =============================================================================
if IS_BLENDER:
    import runpy

    def blender_main():
        argv = sys.argv
        if "--" not in argv:
            sys.exit("No '--' separator found in argv")
        args = argv[argv.index("--") + 1:]
        if len(args) < 2:
            sys.exit("Usage: blender --bg --python THIS -- <script> <result_json>")

        script_path = args[0]
        result_path = args[1]

        result = {"script": script_path, "success": False, "error": None,
                  "mesh_count": 0, "total_verts": 0, "total_faces": 0,
                  "object_names": []}

        # Clean scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        for m in list(bpy.data.meshes):
            bpy.data.meshes.remove(m)
        for c in list(bpy.data.curves):
            bpy.data.curves.remove(c)
        for ng in list(bpy.data.node_groups):
            bpy.data.node_groups.remove(ng)
        bpy.context.scene.cursor.location = (0, 0, 0)

        # Execute target script
        try:
            saved_argv = sys.argv[:]
            sys.argv = [script_path]
            runpy.run_path(script_path, run_name="__main__")
            sys.argv = saved_argv
        except SystemExit as e:
            if e.code not in (None, 0):
                result["error"] = f"Script called sys.exit({e.code})"
                with open(result_path, "w") as f:
                    json.dump(result, f)
                return
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            with open(result_path, "w") as f:
                json.dump(result, f)
            return

        # Check output
        bpy.context.view_layer.update()
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

        result["mesh_count"] = len(mesh_objs)
        result["total_verts"] = sum(len(o.data.vertices) for o in mesh_objs)
        result["total_faces"] = sum(len(o.data.polygons) for o in mesh_objs)
        result["object_names"] = [o.name for o in mesh_objs]

        if not mesh_objs:
            result["error"] = "No mesh objects produced"
        elif result["total_verts"] == 0:
            result["error"] = "All meshes have 0 vertices"
        else:
            result["success"] = True

        with open(result_path, "w") as f:
            json.dump(result, f)

    blender_main()


# =============================================================================
#  CLI-SIDE: orchestrates validation
# =============================================================================
else:
    BLENDER_DEFAULT = os.environ.get(
        "BLENDER", "/path/to/blender-5.0/blender"
    )

    def validate_one(script_path, blender_bin, timeout):
        """Run a single script in Blender and return the result dict."""
        result_file = tempfile.NamedTemporaryFile(
            suffix=".json", prefix="validate_", delete=False
        )
        result_path = result_file.name
        result_file.close()

        cmd = [
            blender_bin, "--background",
            "--python", __file__,
            "--",
            str(script_path), result_path,
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0

            # Try to read the result JSON
            try:
                with open(result_path) as f:
                    result = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Blender crashed before writing result
                stderr_tail = (proc.stderr or "")[-500:]
                stdout_tail = (proc.stdout or "")[-500:]
                result = {
                    "script": str(script_path),
                    "success": False,
                    "error": f"Blender exited with code {proc.returncode}",
                    "mesh_count": 0,
                    "total_verts": 0,
                    "total_faces": 0,
                    "object_names": [],
                    "stderr_tail": stderr_tail,
                    "stdout_tail": stdout_tail,
                }

            result["elapsed_sec"] = round(elapsed, 1)
            result["returncode"] = proc.returncode

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            result = {
                "script": str(script_path),
                "success": False,
                "error": f"Timeout after {timeout}s",
                "mesh_count": 0,
                "total_verts": 0,
                "total_faces": 0,
                "object_names": [],
                "elapsed_sec": round(elapsed, 1),
                "returncode": -1,
            }
        finally:
            try:
                os.unlink(result_path)
            except OSError:
                pass

        return result

    def main():
        parser = argparse.ArgumentParser(
            description="Validate Blender scripts produce non-empty mesh output.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("target", help="Path to a .py file or directory")
        parser.add_argument("--blender", default=BLENDER_DEFAULT,
                            help=f"Blender binary (default: {BLENDER_DEFAULT})")
        parser.add_argument("-j", "--jobs", type=int, default=1,
                            help="Parallel workers (default: 1)")
        parser.add_argument("-t", "--timeout", type=int, default=120,
                            help="Per-script timeout in seconds (default: 120)")
        parser.add_argument("--failures-only", action="store_true",
                            help="Only print failed scripts")
        parser.add_argument("-o", "--output", type=str, default=None,
                            help="Write results to a log file")
        args = parser.parse_args()

        target = Path(args.target).resolve()
        if not target.exists():
            sys.exit(f"Target not found: {target}")

        # Collect scripts
        if target.is_file():
            scripts = [target] if target.suffix == ".py" else []
        else:
            scripts = sorted(target.rglob("*.py"))
            scripts = [s for s in scripts if not s.name.startswith("__")]

        if not scripts:
            sys.exit("No .py files found.")

        print(f"Blender Script Validator")
        print(f"========================")
        print(f"Blender:  {args.blender}")
        print(f"Target:   {target}")
        print(f"Scripts:  {len(scripts)}")
        print(f"Workers:  {args.jobs}")
        print(f"Timeout:  {args.timeout}s per script")
        print()

        results = []
        passed = 0
        failed = 0

        if args.jobs == 1:
            # Sequential — simpler output
            for i, script in enumerate(scripts, 1):
                rel = script.relative_to(target) if target.is_dir() else script.name
                print(f"[{i}/{len(scripts)}] {rel} ... ", end="", flush=True)
                r = validate_one(script, args.blender, args.timeout)
                results.append(r)
                if r["success"]:
                    passed += 1
                    status = (f"OK  ({r['mesh_count']} mesh, "
                              f"{r['total_verts']} verts, "
                              f"{r['elapsed_sec']}s)")
                    if not args.failures_only:
                        print(status)
                    else:
                        print()  # clear the line
                else:
                    failed += 1
                    print(f"FAIL  ({r['error']})")
        else:
            # Parallel
            future_map = {}
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for script in scripts:
                    fut = pool.submit(validate_one, script, args.blender, args.timeout)
                    future_map[fut] = script

                done_count = 0
                for fut in as_completed(future_map):
                    done_count += 1
                    script = future_map[fut]
                    rel = script.relative_to(target) if target.is_dir() else script.name
                    r = fut.result()
                    results.append(r)
                    if r["success"]:
                        passed += 1
                        if not args.failures_only:
                            print(f"[{done_count}/{len(scripts)}] OK    {rel}  "
                                  f"({r['mesh_count']} mesh, {r['total_verts']} verts, "
                                  f"{r['elapsed_sec']}s)", flush=True)
                    else:
                        failed += 1
                        print(f"[{done_count}/{len(scripts)}] FAIL  {rel}  "
                              f"({r['error']})", flush=True)

        # Summary
        print()
        print(f"{'=' * 60}")
        print(f"  PASSED: {passed}/{len(scripts)}")
        print(f"  FAILED: {failed}/{len(scripts)}")
        print(f"{'=' * 60}")

        if failed > 0:
            print(f"\nFailed scripts:")
            for r in results:
                if not r["success"]:
                    print(f"  {r['script']}")
                    print(f"    Error: {r['error']}")

        # Write log file
        if args.output:
            log_path = Path(args.output)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"Blender Script Validation Log\n")
                f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"Blender: {args.blender}\n")
                f.write(f"Target: {target}\n")
                f.write(f"Total: {len(scripts)}  Passed: {passed}  Failed: {failed}\n")
                f.write(f"{'=' * 60}\n\n")

                if failed > 0:
                    f.write("FAILURES:\n")
                    f.write("-" * 60 + "\n")
                    for r in results:
                        if not r["success"]:
                            f.write(f"\n{r['script']}\n")
                            f.write(f"  Error: {r['error']}\n")
                            if r.get("stderr_tail"):
                                f.write(f"  Stderr (tail):\n")
                                for line in r["stderr_tail"].splitlines()[-10:]:
                                    f.write(f"    {line}\n")
                    f.write("\n")

                f.write("\nALL RESULTS:\n")
                f.write("-" * 60 + "\n")
                for r in sorted(results, key=lambda x: x["script"]):
                    status = "OK" if r["success"] else "FAIL"
                    f.write(f"[{status:4s}] {r['script']}  "
                            f"mesh={r['mesh_count']} verts={r['total_verts']} "
                            f"faces={r['total_faces']} time={r.get('elapsed_sec', '?')}s")
                    if not r["success"]:
                        f.write(f"  error={r['error']}")
                    f.write("\n")

            print(f"\nLog written to: {log_path}")

        sys.exit(1 if failed > 0 else 0)

    main()
