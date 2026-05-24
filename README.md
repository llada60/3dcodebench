# 3DCodeBench

**Benchmarking Agentic Procedural 3D Modeling Via Code.**
*Yipeng Gao, Lei Shu, Genzhi Ye, Xi Xiong, Ameesh Makadia, Meiqi Guo, Laurent Itti, Jindong Chen*

| Project page | Paper | Online arena |
|---|---|---|
| [3dcodebench.com](https://www.3dcodebench.com) | (preprint coming soon) | [3dcodebench.com/arena](https://www.3dcodebench.com/arena) |

3DCodeBench measures how well frontier models can **write Blender 5.0 Python
that procedurally builds a specific 3D object**. The benchmark covers 212
categories — chairs, plants, sea creatures, coral, kitchen hardware, … — each
with a ground-truth factory script, a text description, and a structured
instruction. We evaluate single-shot, multi-turn, and full coding-agent
settings, and score outputs on executability, image similarity (SigLIP-2 /
DINOv3), 3D-shape distance (Chamfer / Uni3D), and LLM-as-judge.

## Repository layout

```
3dcodebench/
├── benchmark/categories/   212 categories, each = factory.py + 2 prompt txts
├── tasks/                  one entry per eval setting
│   ├── text_to_3d/         description → Blender Python
│   ├── image_to_3d/        rendered image → Blender Python
│   ├── multi_turn/         T=3 retry loop with traceback feedback
│   └── coding_agent/       Claude Code / Codex / Gemini CLI / agy wrappers
├── metrics/                executability, SigLIP/DINOv3, Chamfer, Uni3D, LLM judge
├── core/                   shared runner, provider abstraction, render/export
├── configs/                one YAML per model (API key from env)
├── prompts/                system + template prompts
├── CONTRIBUTING.md         how to add new categories
└── LICENSE
```

Each subdirectory has its own README. Start with [`tasks/README.md`](tasks/README.md)
to see what to run, then [`metrics/README.md`](metrics/README.md) for scoring.

## Quickstart

### Install

```bash
git clone https://github.com/gaoypeng/3dcodebench.git
cd 3dcodebench
pip install -r requirements.txt

# Blender 5.0 must be installed separately (https://www.blender.org/download/):
export BLENDER=/path/to/blender-5.0/blender

# API keys -- set the ones for the providers you'll call:
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

Outputs land at `results/<model>/<Category>_seed0/`:
* `<Category>_seed0.py` -- generated Blender Python
* `<Category>_seed0.glb` -- exported mesh
* `<Category>_seed0.json` -- usage + cost + traceback (on failure)

### Run inference

```bash
# Task 1: single-shot text-to-3D
python tasks/text_to_3d/run.py    --config configs/gemini_3_1_pro.yaml

# Task 2: single-shot image-to-3D
#   The runner renders the reference image on the fly from the factory.
python tasks/image_to_3d/run.py   --config configs/gemini_3_1_pro.yaml --task image_to_3d

# Task 3: multi-turn error-feedback loop (T=3 retries on failed instances)
python tasks/multi_turn/run.py    --config configs/gemini_3_1_pro.yaml \
                                  --max-feedback-rounds 3

# Task 4: coding-agent harness (Claude Code / Codex / Gemini CLI / agy)
#   See tasks/coding_agent/README.md for per-CLI setup.
bash tasks/coding_agent/run_claude_agent.sh ArmChair_seed0
```

### Score the outputs

```bash
RESULTS=results/gemini-3.1-pro-preview

# Geometry-free scorers (no GPU required):
python metrics/executability.py    --results-dir $RESULTS
python metrics/shape_chamfer.py    --results-dir $RESULTS --reference-dir benchmark/categories
python metrics/failure_taxonomy.py --results-dir $RESULTS

# Image-grounded scorers (need GPU + SigLIP-2 / DINOv3 weights):
python metrics/image_similarity.py --results-dir $RESULTS --reference-dir benchmark/categories \
                                   --model siglip2-base
python metrics/image_similarity.py --results-dir $RESULTS --reference-dir benchmark/categories \
                                   --model dinov3

# 3D-3D scorer (needs Uni3D weights):
python metrics/shape_uni3d.py      --results-dir $RESULTS --reference-dir benchmark/categories

# LLM-as-judge (pairwise or absolute):
python metrics/llm_judge/judge.py  --mode image --results-dir $RESULTS
```

See [`metrics/README.md`](metrics/README.md) for the full setup of SigLIP-2,
DINOv3, and Uni3D (model weights, conda env, GPU notes).

## Contributing

The benchmark grows by **adding new categories**. If you have a procedural
Blender script for something we don't cover yet (a new vehicle, a building, a
musical instrument, …), please open a PR. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the format and review checklist.

PRs that add new **eval tasks** (e.g. a sketch-to-3D variant) or new
**metrics** (e.g. material-fidelity scorer) are also welcome — please open
an issue first to align on scope.

## Citation

```bibtex
@misc{gao2026threedcodebench,
  title  = {3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code},
  author = {Gao, Yipeng and Shu, Lei and Ye, Genzhi and Xiong, Xi and
            Makadia, Ameesh and Guo, Meiqi and Itti, Laurent and Chen, Jindong},
  year   = {2026},
  howpublished = {\url{https://www.3dcodebench.com}}
}
```

## Acknowledgements

Categories are distilled from the [Infinigen](https://github.com/princeton-vl/infinigen)
procedural asset library (Princeton Vision & Learning Lab).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The factory
scripts under `benchmark/categories/` retain Infinigen's BSD-3-Clause license.
