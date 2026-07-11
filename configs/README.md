# Model configs

One YAML per model. Pick the right one and point any `tasks/*/run.py` at it.

| Family | Files |
|---|---|
| Gemini | `gemini_3_1_pro.yaml`, `gemini_3_1_flash_lite.yaml`, `gemini_3_5_flash.yaml`, `gemini_3_flash.yaml`, `gemini_2_5_pro.yaml`, `gemini_cli_3_pro.yaml` |
| Anthropic | `claude_opus_4_7.yaml`, `claude_sonnet_4_6.yaml`, `claude_haiku_4_5.yaml` |
| OpenAI | `gpt_5_5.yaml`, `gpt_5_5_pro.yaml`, `gpt_5_4.yaml`, `gpt_5_4_mini.yaml`, `gpt_5_4_nano.yaml` |
| Open-source | `gemma_4_26b.yaml`, `gemma_4_31b.yaml` |
| CLI-backed (no API key) | `agy_gemini_3_pro.yaml` — Gemini 3 Pro through the local Antigravity `agy` CLI login, parity with 3D-CoT `inference_geometry_oneshot.py --generator_type agy-gemini-3-pro`; needs the 3D-CoT pixi env (pyagy). `codex.yaml` — the local Codex CLI's default model (`codex login`, or `OPENAI_API_KEY` if set; pinning `codex-<model>` needs API-key auth), parity with `--generator_type codex`; needs the 3D-CoT pixi env (pycodex) |

## API keys

Every yaml has `api_key: ${ENV_VAR}` as a placeholder. The runner expands
`${GEMINI_API_KEY}`, `${ANTHROPIC_API_KEY}`, or `${OPENAI_API_KEY}` from your
environment at load time. **Never commit a real key.**

```bash
export GEMINI_API_KEY=AIza...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## Editable fields

```yaml
provider: gemini | anthropic | openai | claude_code | gemini_cli
model: gemini-3.1-pro-preview     # provider-specific model ID
temperature: 0.7
thinking: off | low | medium | high | dynamic | <int>
max_output_tokens: 65536
task: text_to_3d | image_to_3d
prompt_type: description | instruction
max_workers: 50                   # request concurrency
rpm: 25                           # provider RPM cap
tpm: 1000000                      # provider TPM cap
max_sweeps: 3                     # extra full passes for missed instances
parse_retries: 3                  # in-call resamples on ast.parse failure
```
