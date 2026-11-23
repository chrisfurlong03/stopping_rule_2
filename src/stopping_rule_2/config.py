"""Paths and constants for the onset stopping rule."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHIRPS_DIR = DATA_DIR / "CHIRPS"
SHAPEFILE = DATA_DIR / "woreda_shapefile" / "manual_zones_woredas.shp"
PROCESSED_DIR = DATA_DIR / "processed"
CHIRPS_WOREDA = PROCESSED_DIR / "chirps_woreda.parquet"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"
OUTPUT_DIR = ROOT / "outputs"

AIFS_DIR = Path("/net/monsoon/cfurlong/stopping_rule2_variants/v1_aifs_ens_direct")
AIFS_PATTERN = "ens_woreda_{year}.npz"

PCODE_FIELD = "adm3_pcode"

# --- Onset definition -------------------------------------------------------
WET_SPELL_DAYS = 3
WET_SPELL_MM = 20.0
DRY_DAY_MM = 1.0
DRY_SPELL_DAYS = 7
CHECK_DAYS = 21  # days after the wet spell that must contain no dry spell

# --- Season / years ---------------------------------------------------------
# Candidate onset days t run May 1 .. Aug 31. Per-year arrays are indexed by
# offset from May 1; ARRAY_START_OFFSET is the offset of the first stored day.
SEASON_START = (5, 1)
SEASON_LEN = 123  # May 1 .. Aug 31 inclusive
ARRAY_START = (3, 1)  # Mar 1, fixed 61 days before May 1 (leap-safe)
ARRAY_START_OFFSET = -61
ARRAY_END = (10, 31)

CHIRPS_YEARS = list(range(1998, 2026))  # climatology years (2025 ends Oct 31)
MODEL_YEARS = list(range(2000, 2025))  # years with AIFS reforecasts

# Decision is made on D = t + DECISION_LAG (the day after the wet spell ends)
DECISION_LAG = WET_SPELL_DAYS

# AIFS lead L is the forecast for calendar day init + L + AIFS_LEAD_OFFSET.
# Verified against CHIRPS by `stopping-rule-2 check-leads`.
AIFS_LEAD_OFFSET = 0

# Candidate AIFS dry-day thresholds (mm); per-woreda threshold is chosen per CV
# fold so AIFS dry-day frequency matches CHIRPS (<1 mm) on training years.
AIFS_DRY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

# --- Climatology ------------------------------------------------------------
CLIM_HALF_WINDOW = 15  # +/- days for smoothing climatology along the season

# --- Model ------------------------------------------------------------------
INNER_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
