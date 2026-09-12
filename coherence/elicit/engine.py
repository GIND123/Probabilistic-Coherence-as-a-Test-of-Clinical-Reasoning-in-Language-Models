"""vLLM serving wrapper with grammar-constrained decoding.

One engine per model, reused across every axiom and baseline so that the
sampling configuration is identical across conditions. Anything that varies
between conditions has to be the thing under study.

Prefix caching is on: every prompt shares the system message and the fixed
49-line candidate list, so that block is prefilled once per run rather than
once per item.
"""
from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Blackwell (sm_120): FlashInfer JIT-compiles against the system nvcc, which
# on this box is CUDA 12.4 and cannot target sm_120. Fall back to the PyTorch
# sampler rather than failing engine start-up.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


@dataclass
class ModelSpec:
    """One model configuration. `key` names it in every results file."""

    key: str
    hf_id: str
    quantization: str | None = None
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    thinking: bool | None = None     # Qwen3 enable_thinking, None = n/a
    tensor_parallel_size: int = 1
    family: str = ""
    params_b: float = 0.0
    notes: str = ""
    extra_engine_kwargs: dict = field(default_factory=dict)

    @property
    def reasoning_tag(self) -> str:
        if self.thinking is None:
            return "n/a"
        return "thinking" if self.thinking else "non-thinking"


@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1400
    seed: int | None = None
    n: int = 1


class Engine:
    """Thin wrapper around vLLM's offline LLM."""

    def __init__(self, spec: ModelSpec, enforce_eager: bool = False,
                 max_num_seqs: int = 128):
        from vllm import LLM

        self.spec = spec
        kwargs: dict[str, Any] = dict(
            model=spec.hf_id,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            dtype=spec.dtype,
            tensor_parallel_size=spec.tensor_parallel_size,
            enable_prefix_caching=True,
            # xgrammar honours disable_any_whitespace; the "auto" backend may
            # pick one that does not, and models then pad the JSON array with
            # tabs until max_tokens and truncate it mid-way.
            structured_outputs_config={"backend": "xgrammar",
                                       "disable_any_whitespace": True},
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        if spec.quantization:
            kwargs["quantization"] = spec.quantization
        kwargs.update(spec.extra_engine_kwargs)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    # -- chat templating --------------------------------------------------
    def apply_template(self, messages: list[dict]) -> str:
        kw: dict[str, Any] = {}
        if self.spec.thinking is not None:
            kw["enable_thinking"] = self.spec.thinking
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            # Model's template does not accept enable_thinking.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    # -- generation -------------------------------------------------------
    def generate(self, prompts: list[list[dict]], gen: GenConfig,
                 json_schema: dict | None = None) -> list[str]:
        from vllm import SamplingParams

        sp_kwargs: dict[str, Any] = dict(
            temperature=gen.temperature, top_p=gen.top_p,
            max_tokens=gen.max_tokens, n=gen.n,
        )
        if gen.seed is not None:
            sp_kwargs["seed"] = gen.seed
        if json_schema is not None:
            sp_kwargs["structured_outputs"] = self._structured(json_schema)
        sp = SamplingParams(**sp_kwargs)
        texts = [self.apply_template(m) for m in prompts]
        outs = self.llm.generate(texts, sp)
        return [o.outputs[0].text for o in outs]

    @staticmethod
    def _structured(schema: dict):
        """Grammar-constrained decoding.

        `disable_any_whitespace` matters more than it looks: without it the
        grammar permits unbounded whitespace between array elements, and
        models padded with tabs and newlines until they hit max_tokens,
        truncating the array mid-way. That showed up as a ~3% schema-failure
        rate that had nothing to do with the model's beliefs.
        """
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return StructuredOutputsParams(json=schema, disable_any_whitespace=True)
        except ImportError:  # older vLLM
            from vllm.sampling_params import GuidedDecodingParams

            return GuidedDecodingParams(json=schema, disable_any_whitespace=True)

    def close(self) -> None:
        try:
            del self.llm
        except AttributeError:
            pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------
_DECODER = json.JSONDecoder()


def _load_leading_json(text: str) -> dict | None:
    """Parse the first JSON object in `text`, ignoring anything after it.

    Grammar-constrained decoding completes the object but does not always
    stop, so a well-formed response is often followed by whitespace padding
    up to max_tokens. Treating that as a parse failure would have discarded
    genuine posteriors -- and, worse, at a rate that could differ between
    evidence orderings, which is exactly the effect under study.
    """
    t = text.strip()
    if not t:
        return None
    start = t.find("{")
    if start < 0:
        return None
    try:
        obj, _ = _DECODER.raw_decode(t[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _salvage_array(text: str, n: int) -> np.ndarray | None:
    """Recover the first `n` numbers when the JSON array never closed.

    Constrained decoding occasionally pads inside the array until max_tokens.
    When the model has already emitted at least `n` values the response is
    informative and discarding it would bias the sample -- so it is salvaged,
    and every salvaged row is flagged in the results so the rate can be
    reported rather than hidden.
    """
    start = text.find("[")
    if start < 0:
        return None
    nums = _NUM.findall(text[start:])
    if len(nums) < n:
        return None
    try:
        return np.asarray([float(x) for x in nums[:n]], dtype=np.float64)
    except ValueError:
        return None


def parse_probabilities(text: str, n: int, salvage: bool = True) -> np.ndarray | None:
    """Parse and normalise a posterior. Returns None if unusable.

    A returned None is recorded as a schema failure, never silently dropped:
    differential parse rates across evidence orderings would be an order
    effect in the instrument rather than in the model.
    """
    obj = _load_leading_json(text)
    if obj is None or "probabilities" not in obj:
        v = _salvage_array(text, n) if salvage else None
        if v is None:
            return None
    else:
        try:
            v = np.asarray(obj["probabilities"], dtype=np.float64)
        except Exception:
            return None
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        return None
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s <= 0:
        # An all-zero plausibility vector is well-formed and expresses no
        # preference among the candidates, which after normalisation IS the
        # uniform posterior. Treating it as a parse failure dropped the item
        # entirely and unbalanced the per-arm sample; it is instead returned
        # as uniform and picked up by `is_informative` as a non-informative
        # response, which is the accounting it belongs in.
        return np.full(n, 1.0 / n)
    return v / s


def parse_weights(text: str, n: int, fmt: str = "numeric") -> np.ndarray | None:
    """Parse elicited evidence weights, returned as natural-log LRs."""
    from coherence.elicit.schema import ORDINAL_BANDS

    obj = _load_leading_json(text)
    if obj is None or "weights" not in obj:
        return None
    raw = obj["weights"]
    if len(raw) != n:
        return None
    try:
        if fmt == "ordinal":
            v = np.asarray([ORDINAL_BANDS[str(x)] for x in raw], dtype=np.float64)
        elif fmt == "numeric":
            a = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                return None
            v = np.log(np.clip(a, 1e-3, 1e3))
        else:  # logodds
            v = np.asarray(raw, dtype=np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(v)):
        return None
    return np.clip(v, -7.0, 7.0)


def parse_index(text: str, n: int) -> int | None:
    obj = _load_leading_json(text)
    if obj is None or "index" not in obj:
        return None
    try:
        i = int(obj["index"])
    except Exception:
        return None
    return i if 0 <= i < n else None
