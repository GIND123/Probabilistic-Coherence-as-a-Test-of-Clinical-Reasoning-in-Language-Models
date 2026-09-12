"""Central paths and constants. Everything else imports from here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
DDXPLUS_RAW = DATA / "DDXPlus" / "extracted"
MIMIC_NOTE = DATA / "MIMIC-IV-Note" / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
MEDQA = DATA / "MedQA-USMLE-4-options-hf"
MEDRBENCH = DATA / "MedRBench" / "data" / "MedRBench"

# Derived artifacts (git-ignored, mirrored to the Hub).
BUILD = ROOT / "build"
PARQUET = BUILD / "parquet"
KB = BUILD / "kb"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
LOGS = ROOT / "logs"
MODELS = BUILD / "models"

for _p in (BUILD, PARQUET, KB, RESULTS, REPORTS, FIGURES, TABLES, LOGS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

DDXPLUS_SPLITS = {
    "train": DDXPLUS_RAW / "release_train_patients",
    "validate": DDXPLUS_RAW / "release_validate_patients",
    "test": DDXPLUS_RAW / "release_test_patients",
}
CONDITIONS_JSON = DDXPLUS_RAW / "release_conditions.json"
EVIDENCES_JSON = DDXPLUS_RAW / "release_evidences.json"

HF_TOKEN = os.environ.get("hf") or os.environ.get("HF_TOKEN")
HF_USER = os.environ.get("HF_USER", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "codx-clinical-coherence")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "codx-elr-fusion")

SEED = 20260912
