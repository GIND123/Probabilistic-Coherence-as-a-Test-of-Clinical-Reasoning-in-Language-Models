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
    lora_path: str | None = None     # serve a LoRA adapter over `hf_id`
    kv_cache_dtype: str | None = None  # "fp8" halves KV bytes/token
    attention_backend: str | None = None  # force VLLM_ATTENTION_BACKEND
    # gpt-oss only: harmony needs a free-running decode, see _so_config below
    free_form: bool = False               # never constrain decoding with a grammar
    final_channel_marker: str | None = None  # keep only what follows this marker
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


def _so_config(spec: "ModelSpec") -> dict[str, Any]:
    """Structured-output settings for the engine."""
    return {"backend": "xgrammar", "disable_any_whitespace": True}


class _Harmony:
    """Why gpt-oss is decoded without a grammar.

    gpt-oss is trained on the harmony format. Its rendered prompt ends at
    `<|start|>assistant` and the template states that a channel must be
    included for every message, so the first generated token has to open a
    channel. Applying a JSON grammar from token zero makes that impossible:
    the model emitted a schema-valid `{"index": 0}` for essentially every case
    and scored 0.039 on pick-one-diagnosis against a 0.020 chance rate, while
    Qwen3-32B scored 0.537 on the same items.

    vLLM's `openai_gptoss` reasoning parser does not help here -- its
    `is_reasoning_end` returns True unconditionally, because it exists for the
    serving path where harmony is decoded by a separate layer, not for offline
    `generate`. So the grammar still applies from token zero.

    Left to decode freely the same model reasons correctly and closes with
    `assistantfinal{"index": 28}` -- the true label. So gpt-oss is decoded
    unconstrained and the text after the final-channel marker is parsed, which
    the existing salvage parsers already handle. The prompt, the sampling
    parameters and the parsed quantity are unchanged; only the decoding
    constraint differs, and that is reported as a per-family deviation.
    """


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
            structured_outputs_config=_so_config(spec),
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        if spec.quantization:
            kwargs["quantization"] = spec.quantization
        # Decode on this single card is KV-bound, not compute-bound: at fp16 a
        # 32B model reserves 256 KB of KV per token and MedGemma-27B (16 KV
        # heads, no GQA compression) 496 KB, which caps real concurrency near
        # 69 and 36 sequences respectively -- far below max_num_seqs. Halving
        # KV width is the only lever left, since gpu_memory_utilization is
        # already at the card's limit. CODX_KV_DTYPE overrides per run so the
        # setting can be A/B'd against the fp16 baseline without code edits.
        kv = os.environ.get("CODX_KV_DTYPE") or spec.kv_cache_dtype
        if kv:
            kwargs["kv_cache_dtype"] = kv
        # vLLM prefers FlashInfer for models with attention sinks (gpt-oss),
        # but FlashInfer JIT-compiles against the system nvcc, which is 12.4
        # here while targeting sm_120 needs >= 12.9. It raises inside
        # _normalize_cuda_arch, the exception is swallowed into a warning that
        # leaves TARGET_CUDA_ARCHS empty, and check_cuda_arch then reports the
        # misleading "FlashInfer requires GPUs with sm75 or higher". TRITON_ATTN
        # is the other backend vLLM itself lists as valid for has_sink=True.
        # vLLM 0.29 dropped the VLLM_ATTENTION_BACKEND env var; the backend is
        # now a field on AttentionConfig, passed through as an engine kwarg.
        if spec.attention_backend:
            kwargs["attention_config"] = {"backend": spec.attention_backend}
        if spec.lora_path:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = 64
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
        if self.spec.free_form:
            sp_kwargs.pop("structured_outputs", None)
        sp = SamplingParams(**sp_kwargs)
        texts = [self.apply_template(m) for m in prompts]
        if self.spec.lora_path:
            from vllm.lora.request import LoRARequest

            outs = self.llm.generate(
                texts, sp,
                lora_request=LoRARequest("elr", 1, self.spec.lora_path))
        else:
            outs = self.llm.generate(texts, sp)
        return [self._final_channel(o.outputs[0].text) for o in outs]

    def _final_channel(self, text: str) -> str:
        """Keep only the answer channel for models that emit a reasoning one.

        gpt-oss closes its analysis channel and opens the answer channel with
        a literal marker, so the JSON is everything after the LAST occurrence
        of it. Sliced here rather than in the parsers so that every task and
        every salvage path sees the same text. A response truncated before the
        marker is left untouched and fails to parse, which is the honest
        outcome -- it is a response with no answer in it.
        """
        marker = self.spec.final_channel_marker
        if not marker or marker not in text:
            return text
        return text.rsplit(marker, 1)[-1]

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
