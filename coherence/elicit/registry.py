"""The model set, as a factorial design rather than a leaderboard.

Every entry exists to answer one of the questions in the proposal's Part 4:

  reasoning vs not   qwen3-32b-think   vs qwen3-32b-nothink   (identical weights)
                     qwen3-8b-think    vs qwen3-8b-nothink
  medical vs general medgemma-27b      vs qwen3-32b           (comparable size)
                     medgemma-1.5-4b   vs qwen3-4b
  scale              medgemma 4b vs 27b; qwen3 4b / 8b / 32b
  recipe             r1-distill-32b (distilled reasoning) vs qwen3-32b (native)

Hardware note. This machine has ONE RTX PRO 5000 Blackwell with 48 GB. The
27B-32B arms therefore run 4-bit (AWQ/GPTQ), which is the deployment regime
the paper is actually about. `gpt-oss-120b` does not fit: its MXFP4 weights
are ~63 GB, so it is declared with CPU offload and marked `fits_in_vram=False`
-- it runs only on the reduced A1 core subset, and every table reports it as
such rather than silently omitting it.

Two corrections to the proposal's model list, made after checking the Hub:
  * "MedGemma 1.5 27B" does not exist. Google released medgemma-1.5-4b-it
    only; the 27B line stops at medgemma-27b-text-it. Both are included.
  * Qwen3-32B in bf16 is 66 GB and does not fit; the official Qwen3-32B-AWQ
    checkpoint is used, and the thinking contrast runs on those same weights.
"""
from __future__ import annotations

from coherence.elicit.engine import ModelSpec

REGISTRY: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


# --- Qwen3: the thinking / non-thinking contrast, at two scales -----------
for _key, _hf, _q, _p in [
    ("qwen3-4b", "Qwen/Qwen3-4B", None, 4.0),
    ("qwen3-8b", "Qwen/Qwen3-8B", None, 8.2),
    ("qwen3-32b", "Qwen/Qwen3-32B-AWQ", "awq_marlin", 32.8),
]:
    for _think in (True, False):
        _add(ModelSpec(
            key=f"{_key}-{'think' if _think else 'nothink'}",
            hf_id=_hf, quantization=_q, thinking=_think, family="qwen3",
            params_b=_p, max_model_len=16384 if _think else 8192,
            notes="identical weights; only the chat template's reasoning mode differs",
        ))

# --- MedGemma: domain tuning, at two scales ------------------------------
_add(ModelSpec(key="medgemma-1.5-4b", hf_id="google/medgemma-1.5-4b-it",
               family="medgemma", params_b=4.3, max_model_len=8192,
               notes="latest MedGemma line; no 27B counterpart exists"))
_add(ModelSpec(key="medgemma-27b", hf_id="bbarn4/medgemma-27b-text-it-GPTQ",
               quantization="gptq_marlin", family="medgemma", params_b=27.0,
               max_model_len=8192,
               notes="community GPTQ of google/medgemma-27b-text-it; 4-bit for 48 GB"))

# --- GPT-OSS ------------------------------------------------------------
_add(ModelSpec(key="gpt-oss-20b", hf_id="openai/gpt-oss-20b", family="gpt-oss",
               params_b=21.0, max_model_len=8192,
               notes="native MXFP4; fits comfortably"))
_add(ModelSpec(key="gpt-oss-120b", hf_id="openai/gpt-oss-120b", family="gpt-oss",
               params_b=117.0, max_model_len=8192, gpu_memory_utilization=0.95,
               extra_engine_kwargs={"cpu_offload_gb": 24},
               notes="MXFP4 weights ~63 GB exceed 48 GB VRAM; CPU-offloaded, "
                     "core A1 subset only"))

# --- Distilled reasoning -------------------------------------------------
_add(ModelSpec(key="r1-distill-32b", hf_id="casperhansen/deepseek-r1-distill-qwen-32b-awq",
               quantization="awq_marlin", family="deepseek", params_b=32.8,
               max_model_len=16384,
               notes="distilled reasoning recipe, contrasted with Qwen3-32B native"))

# --- General-purpose medical baselines -----------------------------------
_add(ModelSpec(key="med42-8b", hf_id="m42-health/Llama3-Med42-8B", family="llama3-med",
               params_b=8.0, max_model_len=8192,
               notes="medically tuned Llama-3 8B, general-vs-medical control at 8B"))

VRAM_GB = 48.0
FITS_IN_VRAM = {k: (s.key != "gpt-oss-120b") for k, s in REGISTRY.items()}

# Ordering used for every results table and figure.
DEFAULT_ORDER = [
    "qwen3-4b-nothink", "qwen3-4b-think",
    "qwen3-8b-nothink", "qwen3-8b-think",
    "qwen3-32b-nothink", "qwen3-32b-think",
    "medgemma-1.5-4b", "medgemma-27b", "med42-8b",
    "gpt-oss-20b", "r1-distill-32b", "gpt-oss-120b",
]

# The subset that the full battery runs on. `gpt-oss-120b` is excluded here
# and run separately on the A1 core.
FULL_BATTERY_MODELS = [k for k in DEFAULT_ORDER if k != "gpt-oss-120b"]


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown model {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def summary() -> str:
    rows = ["| key | HF id | params (B) | quant | reasoning | fits 48 GB |",
            "|---|---|---|---|---|---|"]
    for k in DEFAULT_ORDER:
        s = REGISTRY[k]
        rows.append(f"| `{k}` | `{s.hf_id}` | {s.params_b:.1f} | "
                    f"{s.quantization or 'bf16/native'} | {s.reasoning_tag} | "
                    f"{'yes' if FITS_IN_VRAM[k] else 'NO (CPU offload)'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(summary())
