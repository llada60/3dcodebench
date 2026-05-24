#!/bin/bash
# image_to_3d variant of run_gemini_agent.sh — gives the agent the four
# reference views from data/<inst>/images/ via @<path> @-mentions, so it
# can Read them with its built-in image-capable tools.
# Usage: run_gemini_agent_img.sh <model> <instance_name> [time_limit_sec]

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model id}"
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
BLENDER="/lab/yipeng/software/blender-5.0.0-linux-x64/blender"

# Build prompt with @-mentions of the 4 reference views.
# gemini-cli auto-reads files referenced via @<path>.
REF_REFS=""
for v in Image_005.png Image_015.png Image_025.png Image_035.png; do
  [ -f "$IMG_DIR/$v" ] && REF_REFS="$REF_REFS @$IMG_DIR/$v"
done

PROMPT_FILE="$WORK_DIR/.agent_prompt.txt"
cat > "$PROMPT_FILE" <<EOF
You are a Blender 5.0 Python expert. Your current working directory is set up for you.

# Task

Look at these four reference views of a single 3D object (azimuths 45°/135°/225°/315° around the object, slight elevation): $REF_REFS

Write a Blender 5.0 Python script named **$INST.py** that procedurally generates a 3D mesh matching that object.

# CRITICAL RULES (failure modes if violated)

1. **USE THE WRITE TOOL** to save the script as $INST.py — do NOT just respond with code in markdown. Code in your message text does NOT count as having created the file.
2. **TEST your script** by running this Bash command and inspecting the output:
   $BLENDER --background --python $INST.py 2>&1 | tail -15
3. **ITERATE if it fails**: Blender 5.0 has API differences from 4.x. Common issues:
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
5. **Aim for high geometric detail** matching the reference views (parametric loops, modifiers, bmesh ops). Avoid trivial cube-stacking.
6. When the script runs cleanly (exit 0, no Python exceptions, mesh produced), use the Write tool to create a sentinel file named **.agent_done** with content "ok".

You have a hard time budget of ${TIME_LIMIT}s; be efficient. Iterate at most 4-5 times.
EOF

LOG="$WORK_DIR/.agent_stdout.log"
START=$(date +%s)
cd "$WORK_DIR"

GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}" \
  timeout "${TIME_LIMIT}s" \
  gemini --skip-trust -m "$MODEL" -y \
    -p "$(cat $PROMPT_FILE)" \
    -o stream-json \
    > "$LOG" 2>&1
RC=$?
DURATION=$(( $(date +%s) - START ))

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

TOKENS=$(grep '"type":"result"' "$LOG" | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); s=d.get('stats',{}); print(f\"{s.get('total_tokens',0)},{s.get('input_tokens',0)},{s.get('output_tokens',0)},{s.get('tool_calls',0)}\")" 2>/dev/null || echo "0,0,0,0")
TOTAL=$(echo "$TOKENS" | cut -d, -f1)
INP=$(echo "$TOKENS" | cut -d, -f2)
OUT=$(echo "$TOKENS" | cut -d, -f3)
TC=$(echo "$TOKENS" | cut -d, -f4)

cat > "$WORK_DIR/.agent_meta.json" <<META
{
  "instance":      "$INST",
  "task":          "image_to_3d",
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
META

echo "[$STATUS] $INST  ${DURATION}s  ${CHARS}b  tools=$TC  tok=$TOTAL"
