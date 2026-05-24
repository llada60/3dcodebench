#!/bin/bash
# Run gemini-cli as an autonomous coding agent for ONE instance.
# Usage: run_gemini_agent.sh <model> <task> <instance_name> [time_limit_sec]

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model id}"
TASK="${2:?provide task: text_to_3d|image_to_3d}"
INST="${3:?provide instance name}"
TIME_LIMIT="${4:-600}"

TASK_LABEL=$(echo "$TASK" | sed 's/text_to_3d/text_to_3D/; s/image_to_3d/image_to_3D/')
WORK_DIR="$EVAL_ROOT/results/${TASK_LABEL}_agent/${MODEL}/${INST}"
mkdir -p "$WORK_DIR"

# Skip if done
if [ -f "$WORK_DIR/$INST.py" ] && [ -f "$WORK_DIR/.agent_done" ]; then
  echo "[SKIP] $INST already done"
  exit 0
fi

DESC_FILE="$EVAL_ROOT/data/$INST/prompt_description.txt"
[ -f "$DESC_FILE" ] || { echo "[ERR] missing $DESC_FILE"; exit 1; }
DESC=$(cat "$DESC_FILE")
BLENDER="/lab/yipeng/software/blender-5.0.0-linux-x64/blender"

# Build prompt — CRITICAL: tell agent to USE WRITE TOOL, not just respond with code
PROMPT_FILE="$WORK_DIR/.agent_prompt.txt"
cat > "$PROMPT_FILE" <<EOF
You are a Blender 5.0 Python expert. Your current working directory is set up for you.

# Task

Write a Blender 5.0 Python script named **$INST.py** that procedurally generates this 3D object as a single mesh:

> $DESC

# CRITICAL RULES (failure modes if violated)

1. **USE THE WRITE TOOL** to save the script as $INST.py — do NOT just respond with code in markdown. Code in your message text does NOT count as having created the file.
2. **TEST your script** by running this Bash command and inspecting the output:
   $BLENDER --background --python $INST.py 2>&1 | tail -15
3. **ITERATE if it fails**: Blender 5.0 has API differences from 4.x. Common issues you may hit:
   - \`Mesh.calc_normals\` removed → use \`mesh.update()\`
   - \`ShaderNodeTexMusgrave\` removed → use \`ShaderNodeTexNoise\`
   - \`ShaderNodeTexNoise\` output renamed "Fac" → "Factor"
   - \`BLENDER_EEVEE_NEXT\` engine name → \`BLENDER_EEVEE\`
   - \`NodeSocketVirtual\` removed → use \`NodeSocketFloat\`
   - \`bpy.ops.mesh.triangulate\` doesn't exist → use \`bmesh.ops.triangulate(bm, faces=bm.faces[:])\`
   - \`Material.shadow_method\` removed → use \`Material.surface_render_method\`
   - \`Mesh.use_auto_smooth\` removed → use \`bpy.ops.object.shade_smooth()\`
   - \`primitive_*_add()\` puts object at 3D cursor unless you pass \`location=(0,0,0)\` explicitly
   - \`numpy.bool_\` rejected by \`v.select_set\` → wrap with \`bool(...)\`
   - \`mathutils.Vector += numpy.ndarray\` fails → use \`v + np.tolist()\`
   - GeoNodes \`CaptureAttribute\` starts with 0 capture_items — must call \`.capture_items.new('FLOAT','Value')\`
   - GeoNodes \`Geometry\` input must be at interface index 0 — call \`ng.interface.move(geom_input, 0)\`
4. **CONSTRAINTS** for the script:
   - Pure Python; no markdown fences inside the .py file
   - Single 3D object (or coherent assembly) at origin
   - NO ground plane / backdrop / extras
   - Allowed libs: bpy, bmesh, mathutils, math, random, itertools, collections, functools, dataclasses, enum, typing, numpy, scipy
   - Clear default scene at start (delete default cube, camera, light)
   - Untextured geometry; use bevel/subsurf/array/mirror modifiers + bmesh for detail
   - At end, final mesh must exist in \`bpy.data.objects\` as MESH type
   - Do NOT call \`sys.exit\`, \`bpy.ops.wm.quit_blender\`, or trigger renders
5. **Aim for high geometric detail** matching the description (parametric loops, modifiers, bmesh ops). Avoid trivial cube-stacking.
6. When the script runs cleanly (exit 0, no Python exceptions, mesh produced), use the Write tool to create a sentinel file named **.agent_done** with content "ok".

You have a hard time budget of ${TIME_LIMIT}s; be efficient. Iterate at most 4-5 times.
EOF

LOG="$WORK_DIR/.agent_stdout.log"
START=$(date +%s)

# CRITICAL: cd into WORK_DIR so write_file uses relative path correctly
cd "$WORK_DIR"

GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}" \
  timeout "${TIME_LIMIT}s" \
  gemini --skip-trust -m "$MODEL" -y \
    -p "$(cat $PROMPT_FILE)" \
    -o stream-json \
    > "$LOG" 2>&1
RC=$?

DURATION=$(( $(date +%s) - START ))

# Status determination
if [ -f "$WORK_DIR/$INST.py" ]; then
  CHARS=$(wc -c < "$WORK_DIR/$INST.py")
  if [ -f "$WORK_DIR/.agent_done" ]; then
    STATUS="OK_AGENT_DONE"
  else
    STATUS="OK_SCRIPT_NO_SENTINEL"
  fi
else
  STATUS="ERR_NO_SCRIPT"
  CHARS=0
fi

# Extract usage stats from JSON output (last "result" line)
TOKENS=$(grep '"type":"result"' "$LOG" | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); s=d.get('stats',{}); print(f\"{s.get('total_tokens',0)},{s.get('input_tokens',0)},{s.get('output_tokens',0)},{s.get('tool_calls',0)}\")" 2>/dev/null || echo "0,0,0,0")
TOTAL=$(echo "$TOKENS" | cut -d, -f1)
INP=$(echo "$TOKENS" | cut -d, -f2)
OUT=$(echo "$TOKENS" | cut -d, -f3)
TC=$(echo "$TOKENS" | cut -d, -f4)

cat > "$WORK_DIR/.agent_meta.json" <<EOF
{
  "instance":      "$INST",
  "task":          "$TASK",
  "model":         "$MODEL",
  "status":        "$STATUS",
  "agent_exit":    $RC,
  "duration_s":    $DURATION,
  "code_chars":    $CHARS,
  "total_tokens":  $TOTAL,
  "input_tokens":  $INP,
  "output_tokens": $OUT,
  "tool_calls":    $TC,
  "time_limit_s":  $TIME_LIMIT
}
EOF

echo "[$STATUS] $INST  ${DURATION}s  ${CHARS}b  tools=$TC  tok=$TOTAL"
