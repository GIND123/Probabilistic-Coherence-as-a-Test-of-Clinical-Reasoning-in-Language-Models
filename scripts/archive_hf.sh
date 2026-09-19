#!/usr/bin/env bash
# Periodic archival of every shareable artifact to the Hugging Face Hub.
#
# Runs on a loop so that a long GPU sweep is checkpointed to the Hub rather
# than living only on this machine -- the thesis depends on these artifacts
# surviving a disk failure.
#
# What goes up: code, tables, figures, reports, the battery, the corpus
# parquet, the LoRA adapter and its checkpoints.
# What never goes up: anything matching the credentialed-data guard in
# coherence/hub/hf_sync.py (mimic, physionet, discharge.csv, radiology.csv).
# results/raw_mimic/ is excluded by that guard AND by .gitignore.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)
PY=.venv/bin/python

INTERVAL="${ARCHIVE_INTERVAL:-3600}"      # seconds between snapshots
ONCE="${ARCHIVE_ONCE:-0}"

snapshot() {
  local stamp; stamp=$(date -Is)
  echo "=== archive $stamp ==="
  $PY -m coherence.hub.hf_sync 2>&1 | sed 's/^/  /' || echo "  dataset sync failed (non-fatal)"
  $PY -m coherence.hub.hf_sync --model-only 2>&1 | sed 's/^/  /' || echo "  model sync failed (non-fatal)"
  # A manifest so a future reader can tell which snapshot produced which numbers.
  $PY - <<'PYEOF' 2>&1 | sed 's/^/  /'
import hashlib, json, subprocess, time
from pathlib import Path
from coherence.config import REPORTS, TABLES, FIGURES

def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]

try:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
except Exception:
    rev = "unknown"
man = {
    "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_rev": rev,
    "tables": {p.name: sha(p) for p in sorted(TABLES.glob("*.csv"))},
    "figures": sorted(p.name for p in FIGURES.glob("*.png")),
}
out = REPORTS / "archive_manifest.json"
out.write_text(json.dumps(man, indent=2))
print(f"manifest: {len(man['tables'])} tables, {len(man['figures'])} figures @ {rev}")
PYEOF
  echo "=== done $stamp ==="
}

if [ "$ONCE" = "1" ]; then snapshot; exit 0; fi
while true; do
  snapshot
  sleep "$INTERVAL"
done
