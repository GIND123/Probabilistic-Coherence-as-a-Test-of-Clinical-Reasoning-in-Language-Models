"""Pin every external checkpoint to the revision this study actually ran.

A model id like `Qwen/Qwen3-8B` is a moving target: the Hub serves whatever is
on `main` today, and `main` moves. Re-running the sweep a year from now against
an id alone therefore does not reproduce these numbers, and nothing in the
output would say so.

What this module does is read the revision each checkpoint was *pinned to at
download time* out of the local Hub cache (`<HF_HOME>/hub/models--*/refs/main`)
and record it. That file is written when the weights were fetched, so it is
evidence of what ran, not a guess made afterwards.

Two products:

  reports/artifact_manifest.json   machine-readable: checkpoint revisions, Hub
                                   artifact paths, sizes and blob hashes
  ARTIFACTS.md                     the same thing as a table, for the README

`drift` re-resolves each id against the Hub and reports where `main` has moved
since. Drift is not an error -- it is the normal state of an open checkpoint --
but it is the difference between "reproducible" and "reproducible if you are
lucky", so it is reported rather than silently tolerated.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from coherence.config import HF_TOKEN, REPORTS, ROOT
from coherence.elicit.registry import DEFAULT_ORDER, FITS_IN_VRAM, REGISTRY

MANIFEST = REPORTS / "artifact_manifest.json"
ARTIFACTS_MD = ROOT / "ARTIFACTS.md"


def _cache_root() -> Path:
    """The Hub cache this project downloads into.

    scripts/env.sh redirects HF_HOME onto the data partition, so the default
    ~/.cache/huggingface is almost never the right place to look.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path(os.environ.get("HF_HUB_CACHE",
                               Path.home() / ".cache" / "huggingface" / "hub"))


def cached_revision(hf_id: str) -> str | None:
    """The commit this id was pinned to locally, or None if never fetched."""
    ref = _cache_root() / f"models--{hf_id.replace('/', '--')}" / "refs" / "main"
    if not ref.exists():
        return None
    rev = ref.read_text().strip()
    return rev or None


# `ModelSpec.quantization` is the kernel *override* passed to vLLM, not a
# description of the weights. Three checkpoints ship pre-quantised and are
# loaded with the override left at None so vLLM reads their own
# `quantization_config` and picks the kernel -- rendering those as
# "bf16/native" would misstate the deployment regime the paper is about.
WEIGHT_FORMAT = {
    "bbarn4/medgemma-27b-text-it-GPTQ": "4-bit compressed-tensors",
    "openai/gpt-oss-20b": "MXFP4",
    "openai/gpt-oss-120b": "MXFP4",
}


def pinned_models() -> list[dict]:
    """One row per evaluated arm, in the paper's table order.

    Deduplication matters here: six of the arms are the *same* weights under a
    different chat-template setting (Qwen3 think/nothink), and
    qwen3-8b-elr-lora is Qwen3-8B plus our adapter. Rows are per arm, but the
    revision is per underlying checkpoint, so a reader can see that the
    thinking contrast really was run on identical weights.
    """
    rows = []
    for key in [*DEFAULT_ORDER, "qwen3-8b-elr-lora"]:
        spec = REGISTRY[key]
        rows.append({
            "arm": key,
            "hf_id": spec.hf_id,
            "revision": cached_revision(spec.hf_id),
            "params_b": spec.params_b,
            "weights": WEIGHT_FORMAT.get(spec.hf_id,
                                         spec.quantization or "bf16"),
            "vllm_quantization_override": spec.quantization,
            "reasoning": spec.reasoning_tag,
            "max_model_len": spec.max_model_len,
            "lora_path": spec.lora_path,
            "fits_48gb": FITS_IN_VRAM[key],
            "ran_full_battery": key != "gpt-oss-120b",
        })
    return rows


def hub_artifacts(kind: str = "dataset") -> dict:
    """Every file present in the release repo, with size and blob hash."""
    from huggingface_hub import HfApi

    from coherence.hub.hf_sync import repo_id

    api = HfApi(token=HF_TOKEN)
    rid = repo_id(kind)
    info = api.repo_info(rid, repo_type=kind, files_metadata=True)
    files = {}
    for f in info.siblings:
        files[f.rfilename] = {
            "size": f.size,
            # lfs blobs carry a sha256; small files carry a git blob sha1.
            "sha": (f.lfs.get("sha256") if isinstance(f.lfs, dict) else None)
                   or getattr(f, "blob_id", None),
        }
    return {"repo_id": rid, "repo_type": kind, "revision": info.sha,
            "private": info.private, "n_files": len(files), "files": files}


def drift() -> list[dict]:
    """Where the Hub's `main` has moved away from the pinned revision."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    seen, out = set(), []
    for row in pinned_models():
        hf_id = row["hf_id"]
        if hf_id in seen:
            continue
        seen.add(hf_id)
        try:
            head = api.model_info(hf_id).sha
        except Exception as exc:                       # gated, renamed, deleted
            head = f"unavailable: {type(exc).__name__}"
        out.append({"hf_id": hf_id, "pinned": row["revision"], "hub_main": head,
                    "moved": bool(row["revision"]) and row["revision"] != head})
    return out


def build(include_hub: bool = True) -> dict:
    import subprocess

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "unknown"
    man = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": rev,
        "seed": 20260912,
        "models": pinned_models(),
    }
    if include_hub:
        for kind in ("dataset", "model"):
            try:
                man[f"hub_{kind}"] = hub_artifacts(kind)
            except Exception as exc:
                man[f"hub_{kind}"] = {"error": f"{type(exc).__name__}: {exc}"}
    return man


def _fmt_size(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return str(n)


def to_markdown(man: dict) -> str:
    L = [
        "# Artifacts and pinned revisions",
        "",
        "Generated by `python -m coherence.hub.manifest`. Do not edit by hand.",
        "",
        f"Source tree at `{man['git_rev']}`, generated {man['generated_utc']}, "
        f"global seed `{man['seed']}`.",
        "",
        "## Evaluated checkpoints",
        "",
        "`revision` is the commit each checkpoint was pinned to in this "
        "machine's Hub cache when the sweep ran. Pass it as `revision=` to "
        "reproduce against the same weights rather than against whatever "
        "`main` serves today.",
        "",
        "| arm | HF id | revision | params (B) | weights | reasoning | full battery |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in man["models"]:
        rev = f"`{m['revision'][:12]}`" if m["revision"] else "*not cached*"
        lora = " + LoRA" if m["lora_path"] else ""
        L.append(
            f"| `{m['arm']}` | [`{m['hf_id']}`](https://huggingface.co/{m['hf_id']}) "
            f"| {rev} | {m['params_b']:.1f} | {m['weights']}{lora} | "
            f"{m['reasoning']} | {'yes' if m['ran_full_battery'] else 'no'} |")
    L += ["",
          "`weights` is the format on the Hub, not a vLLM flag: the three "
          "pre-quantised checkpoints are loaded with no `quantization` override "
          "so vLLM reads their own `quantization_config`. The 27B–32B arms run "
          "4-bit because one 48 GB card is the deployment regime this paper is "
          "about.",
          "",
          "`qwen3-8b-elr-lora` is `Qwen/Qwen3-8B` at the revision above plus the "
          "LoRA adapter in the model repo below. The six Qwen3 arms are three "
          "checkpoints under two chat-template settings, so the "
          "thinking/non-thinking contrast is on identical weights by "
          "construction.", ""]

    for kind, title in (("dataset", "Dataset repo"), ("model", "Model repo")):
        h = man.get(f"hub_{kind}")
        if not h or "error" in h:
            L += [f"## {title}", "",
                  f"*unavailable at generation time: {h.get('error') if h else 'not queried'}*",
                  ""]
            continue
        vis = "private" if h["private"] else "public"
        L += [f"## {title}", "",
              f"[`{h['repo_id']}`](https://huggingface.co/"
              f"{'datasets/' if kind == 'dataset' else ''}{h['repo_id']}) "
              f"— {vis}, {h['n_files']} files, at revision `{h['revision'][:12]}`.",
              ""]
        groups: dict[str, list[tuple[str, dict]]] = {}
        for path, meta in sorted(h["files"].items()):
            if path.startswith("."):
                continue
            groups.setdefault(path.split("/")[0] if "/" in path else "(root)",
                              []).append((path, meta))
        L += ["| directory | files | bytes |", "|---|---|---|"]
        for g, items in sorted(groups.items()):
            total = sum(m.get("size") or 0 for _, m in items)
            L.append(f"| `{g}` | {len(items)} | {_fmt_size(total)} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hub", action="store_true",
                    help="skip the Hub query; pin checkpoints only")
    ap.add_argument("--drift", action="store_true",
                    help="report where Hub main has moved off the pinned revision")
    a = ap.parse_args()

    if a.drift:
        moved = 0
        for d in drift():
            flag = "MOVED" if d["moved"] else "ok   "
            print(f"  {flag}  {d['hf_id']:<48s} pinned={str(d['pinned'])[:12]} "
                  f"main={str(d['hub_main'])[:12]}")
            moved += bool(d["moved"])
        print(f"\n{moved} checkpoint(s) have moved since the sweep. The pinned "
              f"revision in {MANIFEST.name} is what reproduces the paper.")
        return

    man = build(include_hub=not a.no_hub)
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    ARTIFACTS_MD.write_text(to_markdown(man))
    n_pinned = sum(1 for m in man["models"] if m["revision"])
    print(f"  {MANIFEST.relative_to(ROOT)}   {n_pinned}/{len(man['models'])} arms pinned")
    print(f"  {ARTIFACTS_MD.relative_to(ROOT)}")
    for kind in ("dataset", "model"):
        h = man.get(f"hub_{kind}", {})
        if "error" in h:
            print(f"  hub_{kind}: {h['error']}")
        elif h:
            print(f"  hub_{kind}: {h['repo_id']} {h['n_files']} files @ {h['revision'][:12]}")


if __name__ == "__main__":
    main()
