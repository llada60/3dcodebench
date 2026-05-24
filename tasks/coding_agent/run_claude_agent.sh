#!/bin/bash
# Run claude (Claude Code CLI) as an autonomous coding agent for ONE instance.
# Usage: run_claude_agent.sh <model> <task> <instance_name> [time_limit_sec] [max_budget_usd]
# Examples:
#   run_claude_agent.sh sonnet text_to_3d ArmChair_seed0 600 1.0
#   run_claude_agent.sh opus   text_to_3d ArmChair_seed0 900 5.0

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model id (sonnet|opus|claude-sonnet-4-6|claude-opus-4-7)}"
TASK="${2:?provide task: text_to_3d|image_to_3d}"
INST="${3:?provide instance name}"
TIME_LIMIT="${4:-600}"
MAX_BUDGET="${5:-2.0}"

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

PROMPT_FILE="$WORK_DIR/.agent_prompt.txt"
cat > "$PROMPT_FILE" <<EOF
You are a Blender 5.0 Python expert. Your current working directory is $WORK_DIR.

# Task

Use the Write tool to create a Blender 5.0 Python script named **$INST.py** that procedurally generates this 3D object as a single mesh:

> $DESC

# Workflow

1. Use the **Write** tool to create $INST.py.
2. Use the **Bash** tool to test it: \`$BLENDER --background --python $INST.py 2>&1 | tail -15\`
3. If it fails, use the **Edit** tool to fix the script and re-test.
4. When the script runs cleanly (exit 0, no exceptions, mesh produced), use the **Write** tool to create a sentinel file named **.agent_done** with content "ok".

# Blender 5.0 API gotchas (common breaks)

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
- \`mathutils.Vector += numpy.ndarray\` fails — convert first
- GeoNodes \`CaptureAttribute\` starts empty — must call \`.capture_items.new('FLOAT','Value')\`
- GeoNodes \`Geometry\` input must be at interface index 0 — call \`ng.interface.move(geom_in, 0)\`
- GeoNodes \`FilletCurve.mode\` is now an input socket; valid values are \`'Poly'\` and \`'Bézier'\` (with é)

# Constraints

- Pure Python; no markdown fences inside the .py file.
- Single 3D object (or coherent assembly) at origin. No ground plane / backdrop / extras.
- Allowed libs: bpy, bmesh, mathutils, math, random, itertools, collections, functools, dataclasses, enum, typing, numpy, scipy.
- Clear default scene at start.
- Aim for high geometric detail (parametric loops, modifiers, bmesh ops) — do not just stack cubes.
- Final mesh must exist in \`bpy.data.objects\` as MESH at end.
- Do NOT call sys.exit, bpy.ops.wm.quit_blender, or trigger renders.

Hard time budget: ${TIME_LIMIT}s. Iterate at most 4-5 times. Be efficient.
EOF

LOG="$WORK_DIR/.agent_stdout.log"
START=$(date +%s)

cd "$WORK_DIR"

# claude flags:
#   -p:                            non-interactive print mode
#   --model:                       sonnet | opus | claude-sonnet-4-6 etc
#   --effort medium:               thinking budget medium
#   --dangerously-skip-permissions: headless tool auto-approval
#   --max-budget-usd:              hard cost cap
#   --bare:                        minimal mode (no hooks/auto-memory/plugin sync)
#   --output-format json:          structured single-result output
#   --add-dir:                     extra dirs the agent can touch
timeout "${TIME_LIMIT}s" \
  claude -p "$(cat $PROMPT_FILE)" \
    --model "$MODEL" \
    --effort medium \
    --dangerously-skip-permissions \
    --max-budget-usd "$MAX_BUDGET" \
    --bare \
    --output-format json \
    --add-dir "$WORK_DIR" \
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

# Parse JSON output for cost / token / turn count
META=$(python3 -c "
import json,sys
try:
  d = json.load(open('$LOG'))
  print(f\"{d.get('total_cost_usd',0)},{d.get('usage',{}).get('input_tokens',0)},{d.get('usage',{}).get('output_tokens',0)},{d.get('num_turns',0)},{d.get('duration_api_ms',0)}\")
except Exception as e:
  print('0,0,0,0,0')
" 2>/dev/null)
COST=$(echo "$META" | cut -d, -f1)
INP=$(echo "$META" | cut -d, -f2)
OUT=$(echo "$META" | cut -d, -f3)
TURNS=$(echo "$META" | cut -d, -f4)

cat > "$WORK_DIR/.agent_meta.json" <<EOF
{
  "instance":      "$INST",
  "task":          "$TASK",
  "model":         "$MODEL",
  "status":        "$STATUS",
  "agent_exit":    $RC,
  "duration_s":    $DURATION,
  "code_chars":    $CHARS,
  "cost_usd":      $COST,
  "input_tokens":  $INP,
  "output_tokens": $OUT,
  "num_turns":     $TURNS,
  "time_limit_s":  $TIME_LIMIT,
  "max_budget":    $MAX_BUDGET
}
EOF

echo "[$STATUS] $INST  ${DURATION}s  ${CHARS}b  turns=$TURNS  \$$COST"
