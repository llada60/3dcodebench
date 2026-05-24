#!/bin/bash
# Run antigravity-cli (agy) as an autonomous coding agent for ONE instance.
# Drop-in replacement for run_gemini_agent.sh. Notes:
#   - agy does NOT expose a -m flag; the Antigravity service picks its own
#     default model (currently Gemini 3 Pro / Claude 4.6 via Google OAuth).
#     MODEL is therefore just a label used in the output dir name.
#   - agy ignores the shell cwd; we use --add-dir + absolute paths inside
#     the prompt so files land where we expect.
# Usage: run_agy_agent.sh <model_label> <task> <instance_name> [time_limit_sec]

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model id (label only — agy uses its own service model)}"
TASK="${2:?provide task: text_to_3d|image_to_3d}"
INST="${3:?provide instance name}"
TIME_LIMIT="${4:-600}"

TASK_LABEL=$(echo "$TASK" | sed 's/text_to_3d/text_to_3D/; s/image_to_3d/image_to_3D/')
WORK_DIR="$EVAL_ROOT/results/${TASK_LABEL}_agent/${MODEL}/${INST}"
mkdir -p "$WORK_DIR"

if [ -f "$WORK_DIR/$INST.py" ] && [ -f "$WORK_DIR/.agent_done" ]; then
  echo "[SKIP] $INST already done"
  exit 0
fi

DESC_FILE="$EVAL_ROOT/data/$INST/prompt_description.txt"
[ -f "$DESC_FILE" ] || { echo "[ERR] missing $DESC_FILE"; exit 1; }
DESC=$(cat "$DESC_FILE")
BLENDER="/lab/yipeng/software/blender-5.0.0-linux-x64/blender"
SCRIPT_PATH="$WORK_DIR/$INST.py"
DONE_PATH="$WORK_DIR/.agent_done"

PROMPT=$(cat <<EOF
You are a Blender 5.0 Python expert. All file operations MUST use the absolute paths below.

# Task

Write a Blender 5.0 Python script at the absolute path
  $SCRIPT_PATH
that procedurally generates this 3D object as a single mesh:

> $DESC

# CRITICAL RULES

1. Use the Write tool to save the script at the EXACT path $SCRIPT_PATH (absolute). Code in your message text does NOT count as having created the file.
2. Test by running: $BLENDER --background --python $SCRIPT_PATH 2>&1 | tail -15
3. Iterate up to 4 times if it fails. Blender 5.0 API notes:
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
4. Script constraints:
   - Pure Python; no markdown fences inside the .py file
   - Single 3D mesh at origin; no ground plane / backdrop
   - Allowed libs: bpy, bmesh, mathutils, math, random, itertools, collections, functools, dataclasses, enum, typing, numpy, scipy
   - Clear scene at start; untextured geometry
   - Final mesh must exist in bpy.data.objects as MESH type
   - Do NOT call sys.exit, bpy.ops.wm.quit_blender, or trigger renders
5. Aim for high geometric detail matching the description.
6. When the script runs cleanly, write a sentinel file at $DONE_PATH containing "ok".

Time budget: ${TIME_LIMIT}s. Be efficient.
EOF
)

LOG="$WORK_DIR/.agent_stdout.log"
START=$(date +%s)

timeout "${TIME_LIMIT}s" \
  /lab/yipeng/.local/bin/agy --print "$PROMPT" \
    --add-dir "$WORK_DIR" \
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
  "task":         "$TASK",
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
