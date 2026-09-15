#!/usr/bin/env bash
# Wait out the in-flight r1-distill-32b run, decide whether an fp8 KV cache is
# measurably different from fp16 (C11), apply the verdict to the models that
# have not run yet, then resume the pipeline.
#
# KV dtype is treated as a per-model property held constant across every arm
# for that model, so each model's order-effect is still measured against its
# own test-retest floor and between-patient ceiling under one engine config.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)
PY=.venv/bin/python

while pgrep -f "coherence.run.runner" >/dev/null; do
  echo "  r1-distill-32b still generating  $(date -Is)"; sleep 120
done
echo "### GPU free  $(date -Is)"

echo "### C11  fp8 KV cache equivalence test  $(date -Is)"
CODX_KV_DTYPE=fp8 timeout 3600 $PY -m coherence.run.calib_kv_cache \
    --model qwen3-32b-nothink --cases 200 \
    2>&1 | grep -vE "^\(|it/s\]|Adding requests|Processed prompts"

VERDICT=$($PY -c "
import json,pathlib
p=pathlib.Path('reports/calib_kv_cache.json')
print(json.loads(p.read_text()).get('accept') if p.exists() else False)" 2>/dev/null)

if [ "$VERDICT" = "True" ]; then
  echo "### C11 ACCEPTED -- enabling fp8 KV for the models not yet run"
  $PY - <<'PYEOF'
import re, pathlib
p = pathlib.Path("coherence/elicit/registry.py")
s = p.read_text()
# Only models with no rows on disk yet. Anything already measured keeps fp16,
# so no model is ever half fp16 and half fp8.
for key in ("gpt-oss-20b", "medgemma-27b", "gpt-oss-120b"):
    pat = re.compile(r'(_add\(ModelSpec\(key="%s".*?)\)\)' % re.escape(key), re.S)
    m = pat.search(s)
    if m and "kv_cache_dtype" not in m.group(1):
        s = s[:m.end(1)] + ', kv_cache_dtype="fp8"' + s[m.end(1):]
        print(f"  fp8 KV -> {key}")
p.write_text(s)
PYEOF
  $PY -c "
from coherence.elicit.registry import get
for k in ['gpt-oss-20b','medgemma-27b','gpt-oss-120b','qwen3-32b-nothink']:
    print(f'  {k:16s} kv={get(k).kv_cache_dtype}')"
else
  echo "### C11 REJECTED (or failed) -- staying on fp16 KV everywhere"
fi

echo "### resuming full pipeline  $(date -Is)"
exec bash scripts/run_all.sh
