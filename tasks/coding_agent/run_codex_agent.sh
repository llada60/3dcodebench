#!/bin/bash
# Run codex CLI as autonomous agent for ONE instance.
# Usage: run_codex_agent.sh <model> <task> <instance> [time_limit]

set -e
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?provide model id (gpt-5.5|gpt-5.4|gpt-5.4-mini)}"
TASK="${2:?provide task}"
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

PROMPT_FILE="$WORK_DIR/.agent_prompt.txt"
cat > "$PROMPT_FILE" <<EOF
You are a Blender 5.0 Python expert. Working directory: $WORK_DIR.

# Task

Create a Blender 5.0 Python script named **$INST.py** that procedurally generates this 3D object as a single mesh:

> $DESC

# Workflow

1. Write the script to $INST.py in the current working directory.
2. Test it: \`$BLENDER --background --python $INST.py 2>&1 | tail -15\`
3. If it fails, fix the script and retry.
4. When the script runs cleanly (exit 0, no exceptions, mesh produced), create a sentinel file **.agent_done** with content "ok".

# Blender 5.0 API gotchas

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
LAST_MSG="$WORK_DIR/.agent_last_msg.txt"
START=$(date +%s)

cd "$WORK_DIR"

# codex exec flags:
#   -m: model
#   --dangerously-bypass-approvals-and-sandbox: headless mode (auto-approve everything)
#   -C: working dir
#   --skip-git-repo-check: don't require git
#   --json: print events as JSONL (for parsing)
#   --ephemeral: no session persistence
#   --ignore-user-config: don't load user config
#   -o: write last assistant message to file
OPENAI_API_KEY="${OPENAI_API_KEY:?missing OPENAI_API_KEY env}" \
  timeout "${TIME_LIMIT}s" \
  codex exec \
    -m "$MODEL" \
    --dangerously-bypass-approvals-and-sandbox \
    -C "$WORK_DIR" \
    --skip-git-repo-check \
    --json \
    --ephemeral \
    --ignore-user-config \
    -o "$LAST_MSG" \
    "$(cat $PROMPT_FILE)" \
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

# Parse JSONL log for token usage. Codex emits one `turn.completed` event
# per agent turn with cumulative-style usage in that turn's segment.
META=$(python3 -c "
import json
total_in = total_out = total_cached = turns = 0
try:
  with open('$LOG') as f:
    for line in f:
      line = line.strip()
      if not line.startswith('{'): continue
      try: ev = json.loads(line)
      except: continue
      if ev.get('type') == 'turn.completed':
        u = ev.get('usage', {}) or {}
        total_in += u.get('input_tokens', 0)
        total_out += u.get('output_tokens', 0)
        total_cached += u.get('cached_input_tokens', 0)
        turns += 1
  print(f'{total_in},{total_out},{total_cached},{turns}')
except Exception:
  print('0,0,0,0')
" 2>/dev/null)
INP=$(echo "$META" | cut -d, -f1)
OUT=$(echo "$META" | cut -d, -f2)
CACHED=$(echo "$META" | cut -d, -f3)
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
  "input_tokens":  $INP,
  "output_tokens": $OUT,
  "cached_tokens": $CACHED,
  "num_turns":     $TURNS,
  "time_limit_s":  $TIME_LIMIT
}
EOF

echo "[$STATUS] $INST  ${DURATION}s  ${CHARS}b  turns=$TURNS  in=$INP out=$OUT cached=$CACHED"
