# Gemini Prompt Templates

This document describes the prompt engineering strategy used by the AI Capacity Forecaster Agent.

## 1. Intent Detection Prompt

**Purpose:** Convert natural language queries into structured JSON parameters.

**Inputs:** User query string

**Outputs:**
- `intent` — One of: threshold_crossing, forecast_period, compare_resources, risk_analysis, what_if, general
- `metrics` — List of metric column names
- `threshold` — Percentage threshold (default 80)
- `forecast_months` — Horizon in months (default 6)
- `growth_multiplier` — What-if growth factor (1.2 = 20% faster)
- `confidence` — Parser confidence 0–1

**Design choices:**
- Strict JSON-only response to enable reliable parsing
- Explicit defaults documented in prompt to reduce ambiguity
- All three metrics returned for compare/risk intents

## 2. Answer Generation Prompt

**Purpose:** Produce business-friendly narrative answers from structured analysis.

**Inputs:** Original query + analysis JSON (forecasts, risk, confidence, recommendations)

**Constraints:**
- Under 200 words
- No invented data
- Include dates, percentages, risk level, and one recommendation

## 3. Validation Prompt

**Purpose:** Second-pass consistency check on agent output.

**Inputs:** Interim analysis JSON

**Outputs:**
- `is_valid` — Boolean
- `issues` — List of inconsistencies
- `adjusted_confidence` — Confidence multiplier 0–1

## Fallback Behavior

When Gemini is unavailable (missing API key or API error), the system uses:

1. **Rule-based NL parser** — Regex and keyword matching
2. **Template answer generator** — Structured text from forecast data

This ensures the application remains functional without external API access.

## Example Queries → Expected Intent

| Query | Intent | Metrics | Threshold |
|-------|--------|---------|-----------|
| When will disk usage hit 80%? | threshold_crossing | disk_usage | 80 |
| Forecast CPU for next 6 months | forecast_period | cpu_usage | 80 |
| Which resource exceeds first? | compare_resources | all | 80 |
| Show risk analysis | risk_analysis | all | 80 |
| What if CPU grows 20% faster? | what_if | cpu_usage | 80 |
