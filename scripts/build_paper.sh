#!/usr/bin/env bash
# Build reports/writeup.tex into the submission PDF.
#
# Why this script exists: writeup.tex references its figures by bare filename
# (\includegraphics{fig1_anchors.pdf}), and three of those are NOT the figures
# in reports/figures/. They are produced by reports/mkfigs.py, a separate
# lightweight generator that reads the committed CSVs in reports/tables/ and
# writes a Times-serif set sized for the two-column layout. Those three PDFs
# are derived, so they are not committed; without this script the tex does not
# compile on a fresh clone and the failure looks like a missing file rather
# than a missing build step.
#
# Everything here is CPU-only and reads committed CSVs. No GPU, no Hub, no
# re-elicitation.
#
# Requires pdflatex. If TeX Live is not installed system-wide, TinyTeX in
# $HOME is picked up automatically (install: https://yihui.org/tinytex/).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY=${PY:-.venv/bin/python}
OUT=${1:-build/paper}
PDF_NAME=Probabilistic_Coherence_Clinical_LLM_Reasoning.pdf

# TinyTeX is a user-local install, so it is not on PATH by default.
for d in "$HOME/.TinyTeX/bin/x86_64-linux" "$HOME/.TinyTeX/bin/universal-darwin"; do
  [ -d "$d" ] && export PATH="$d:$PATH"
done
command -v pdflatex >/dev/null || {
  echo "error: pdflatex not found." >&2
  echo "  install TeX Live, or TinyTeX without root:" >&2
  echo "  curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh" >&2
  echo "  \$HOME/.TinyTeX/bin/*/tlmgr install geometry psnfss booktabs caption titlesec hyphenat grfext" >&2
  exit 1
}

rm -rf "$OUT"; mkdir -p "$OUT"

echo "### figures  (reports/mkfigs.py <- reports/tables/*.csv)"
$PY reports/mkfigs.py "$OUT" reports/tables

echo "### staging"
cp reports/writeup.tex "$OUT"/
cp reports/figures/fig15_pipeline.pdf "$OUT"/

echo "### pdflatex  (two passes)"
cd "$OUT"
for pass in 1 2; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error writeup.tex > "pass$pass.log" 2>&1; then
    echo "  pass $pass FAILED:" >&2
    grep -A4 '^!' "pass$pass.log" | head -40 >&2
    exit 1
  fi
done
cd "$ROOT"

# Only overwrite the committed PDF once the build has actually succeeded, so a
# failed build never leaves a truncated submission artifact behind.
cp "$OUT/writeup.pdf" "reports/$PDF_NAME"

PAGES=$($PY - <<PYEOF
from pypdf import PdfReader
print(len(PdfReader("reports/$PDF_NAME").pages))
PYEOF
) || PAGES="?"
BAD=$(grep -c 'Overfull\|Underfull' "$OUT/pass2.log" || true)

echo "### done"
echo "  reports/$PDF_NAME   ${PAGES} pages, ${BAD} over/underfull box(es)"
echo "  build tree kept at $OUT"
