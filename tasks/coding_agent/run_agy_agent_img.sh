#!/bin/bash
# Image-to-3D variant of run_agy_agent.sh. Antigravity CLI (agy) gets the
# four reference views via absolute paths in the prompt and reads them
# through its native multimodal Read tool.
# Usage: run_agy_agent_img.sh <model_label> <instance_name> [time_limit_sec]

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model label}"
INST="${2:?provide instance name}"
TIME_LIMIT="${3:-600}"

WORK_DIR="$EVAL_ROOT/results/image_to_3D_agent/${MODEL}/${INST}"
mkdir -p "$WORK_DIR"

if [ -f "$WORK_DIR/$INST.py" ] && [ -f "$WORK_DIR/.agent_done" ]; then
  echo "[SKIP] $INST already done"
  exit 0
fi

IMG_DIR="$EVAL_ROOT/data/$INST/images"
[ -d "$IMG_DIR" ] || { echo "[ERR] missing $IMG_DIR"; exit 1; }

# List the existing reference views as a bullet list of absolute paths.
REF_LIST=""
for v in Image_005.png Image_015.png Image_025.png Image_035.png; do
  [ -f "$IMG_DIR/$v" ] && REF_LIST="$REF_LIST
  - $IMG_DIR/$v"
done

BLENDER="/lab/yipeng/software/blender-5.0.0-linux-x64/blender"
SCRIPT_PATH="$WORK_DIR/$INST.py"
DONE_PATH="$WORK_DIR/.agent_done"

PROMPT=$(cat <<EOF
You are a Blender 5.0 Python expert. All file operations MUST use the absolute paths below.

# Task

First read these four reference views of a single 3D object (azimuths 45°/135°/225°/315° around the object, slight elevation):$REF_LIST

Then write a Blender 5.0 Python script at the absolute path
  $SCRIPT_PATH
that procedurally generates a 3D mesh matching the object shown in those reference views.

# CRITICAL RULES

1. Read the four reference images above first (use the Read tool with the absolute paths). Reason about the object's geometry from the views before coding.
2. Use the Write tool to save the script at EXACTLY $SCRIPT_PATH (absolute path).
3. Test by running: $BLENDER --background --python $SCRIPT_PATH 2>&1 | tail -15
4. Iterate up to 4 times if it fails. Blender 5.0 API notes:
   - Mesh.calc_normals removed → use mesh.update()
   - ShaderNodeTexMusgrave removed → use ShaderNodeTexNoise
   - ShaderNodeTexNoise output "Fac" → "Factor"
   - BLENDER_EEVEE_NEXT → BLENDER_EEVEE
   - NodeSocketVirtual removed → use NodeSocketFloat
   - bpy.ops.mesh.triangulate doesn't exist → bmesh.ops.triangulate(bm, faces=bm.faces[:])
   - Material.shadow_method removed → Material.surface_render_method
   - Mesh.use_auto_smooth removed → bpy.ops.object.shade_smooth()
   - primitive_*_add needs explicit location=(0,0,0)
   - numpy.bool_ rejected by v.select_set → wrap with bool(...)
   - mathutils.Vector += numpy.ndarray fails
   - GeoNodes CaptureAttribute starts empty; call .capture_items.new('FLOAT','Value')
   - GeoNodes Geometry input must be at interface index 0
5. Script constraints:
   - Pure Python; no markdown fences inside the .py file
   - Single 3D mesh at origin; no ground plane / backdrop
   - Allowed libs: bpy, bmesh, mathutils, math, random, itertools, collections, functools, dataclasses, enum, typing, numpy, scipy
   - Clear scene at start; untextured geometry
   - Final mesh must exist in bpy.data.objects as MESH type
   - Do NOT call sys.exit, bpy.ops.wm.quit_blender, or trigger renders
6. Aim for high geometric detail matching the reference views.
7. When the script runs cleanly, write a sentinel file at $DONE_PATH containing "ok".

Time budget: ${TIME_LIMIT}s. Be efficient.
EOF
)

LOG="$WORK_DIR/.agent_stdout.log"
START=$(date +%s)

timeout "${TIME_LIMIT}s" \
  /lab/yipeng/.local/bin/agy --print "$PROMPT" \
    --add-dir "$WORK_DIR" \
    --add-dir "$IMG_DIR" \
    --dangerously-skip-permissions \
    --print-timeout "$((TIME_LIMIT - 10))s" \
    > "$LOG" 2>&1
RC=$?
DURATION=$(( $(date +%s) - START ))

if [ -f "$SCRIPT_PATH" ]; then
  CHARS=$(wc -c < "$SCRIPT_PATH")
  if [ -f "$DONE_PATH" ]; then
    STATUS="OK_AGENT_DONE"
  else
    STATUS="OK_SCRIPT_NO_SENTINEL"
  fi
else
  STATUS="ERR_NO_SCRIPT"
  CHARS=0
fi

cat > "$WORK_DIR/.agent_meta.json" <<META
{
  "instance":     "$INST",
  "task":         "image_to_3d",
  "model":        "$MODEL",
  "agent_cli":    "agy",
  "status":       "$STATUS",
  "agent_exit":   $RC,
  "duration_s":   $DURATION,
  "code_chars":   $CHARS,
  "time_limit_s": $TIME_LIMIT
}
META

echo "[$STATUS] $INST  ${DURATION}s  ${CHARS}b"
