"""Per-model API pricing for the 3DProBench eval set.

All prices are USD per **1 M tokens** at the *standard* / short-context
(<=200K-context-input) tier. Cache and long-context premiums are noted
inline. Sources verified 2026-04-29:

  Gemini   - https://ai.google.dev/gemini-api/docs/pricing
  Anthropic- https://platform.claude.com/docs/en/about-claude/models/overview
  OpenAI   - https://platform.openai.com/docs/pricing  (short-context tier)

Used by:
  - eval/run_inference.py  (writes per-instance + per-run cost into logs)
  - any post-hoc cost analysis over the JSONL logs in eval/logs/

The pricing table is the only authoritative source — update *here* when
a model is renamed, repriced, or replaced.
"""

# ──────────────────────────────────────────────────────────────────────
# Pricing table.  Per 1 M tokens, USD.
# Keys per entry:
#   input          — base input rate
#   output         — output rate (also billed for thinking/reasoning tokens)
#   cached_input   — cache-read rate; ~0.1x input for Anthropic, automatic
#                    for OpenAI (>=1024 tok prefix), explicit for Gemini.
#                    None if the provider doesn't expose a cached rate.
#   cache_write_5m — Anthropic only: rate to *write* a 5-minute cache entry
#                    (1.25x input). 1-hour TTL would be 2x but we don't use
#                    it. Absent for non-Anthropic models.
# ──────────────────────────────────────────────────────────────────────

PRICING = {
    # ============ Gemini (Google AI Studio "Paid", <=200K context) ===========
    # Above 200K context input prices roughly double; not modeled here.
    "gemini-2.5-pro":                {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    "gemini-3-flash-preview":        {"input": 0.50, "output":  3.00, "cached_input": 0.05},
    "gemini-3-pro-preview":          {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gemini-3.1-pro-preview":        {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output":  1.50, "cached_input": 0.025},
    # Gemini 3.5 Flash — list price not officially published at time of run; using
    # gemini-3-flash-preview pricing as the conservative placeholder. Update once
    # Google announces the paid tier.
    "gemini-3.5-flash":              {"input": 0.50, "output":  3.00, "cached_input": 0.05},

    # ============ Gemma (open-weight; Gemini API hosting — pricing TBD) ======
    # Listed as 0 since Google AI Studio currently serves Gemma free in preview;
    # adjust if/when paid pricing publishes.
    "gemma-4-26b-a4b-it":            {"input": 0.0,  "output":  0.0,  "cached_input": 0.0},
    "gemma-4-31b-it":                {"input": 0.0,  "output":  0.0,  "cached_input": 0.0},

    # ============ Anthropic (Claude API list price) ==========================
    # cached_input = 0.1 * input;  cache_write_5m = 1.25 * input
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write_5m": 6.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write_5m": 3.75},
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00, "cached_input": 0.10, "cache_write_5m": 1.25},

    # ============ OpenAI (Standard / short-context tier) =====================
    # Long-context (>=256K input) tier roughly 2x; not modeled here.
    "gpt-5.5":      {"input":  5.00, "output":  30.00, "cached_input":  0.50},
    "gpt-5.5-pro":  {"input": 30.00, "output": 180.00, "cached_input":  None},
    "gpt-5.4":      {"input":  2.50, "output":  15.00, "cached_input":  0.25},
    "gpt-5.4-mini": {"input":  0.75, "output":   4.50, "cached_input":  0.075},
    "gpt-5.4-nano": {"input":  0.20, "output":   1.25, "cached_input":  0.02},
}


def cost_usd(model,
             *,
             input_tokens=0,
             output_tokens=0,
             thoughts_tokens=0,
             cache_read_tokens=0,
             cache_creation_tokens=0):
    """Compute USD cost for one inference call.

    All token counts are raw counts. Returns ``None`` if ``model`` is not
    in :data:`PRICING`.

    Notes
    -----
    * Thoughts/reasoning tokens are billed at the **output** rate by all
      three providers (Gemini, Anthropic, OpenAI), so they are folded
      into ``output_tokens`` before pricing.
    * ``input_tokens`` should be the *uncached* portion only;
      ``cache_read_tokens`` is priced separately at ``cached_input``.
    * For providers without a documented cached rate, cache reads
      silently fall back to the full input rate (no error).
    * ``cache_creation_tokens`` is Anthropic-specific; ignored if the
      model has no ``cache_write_5m`` entry.
    """
    p = PRICING.get(model)
    if p is None:
        return None
    M = 1_000_000
    out_total = (output_tokens or 0) + (thoughts_tokens or 0)
    cost = 0.0
    cost += (input_tokens or 0) * p["input"] / M
    cost += out_total           * p["output"] / M
    cached_rate = p.get("cached_input") or p["input"]
    cost += (cache_read_tokens or 0) * cached_rate / M
    if "cache_write_5m" in p:
        cost += (cache_creation_tokens or 0) * p["cache_write_5m"] / M
    return cost
