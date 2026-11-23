# stopping-rule-2

Woreda-level rainy-season **onset stopping rule** for Amhara and Oromia, built from CHIRPS v3.0 and AIFS ENS reforecasts.

**Onset** on day *t* means two things hold:

- rainfall on t..t+2 totals at least 20 mm (the wet spell), and
- the following 21 days (t+3..t+23) contain no 7-day dry spell, where a dry day has less than 1 mm.

The model is used only after a wet spell has been observed, so it works as a **dry-spell detector**. On decision day D = t+3 it gives a calibrated probability that the 21-day window D..D+20 contains no 7-day dry spell. The stopping rule declares onset on the first wet-spell day whose probability is at or above a threshold. That threshold maximizes the Peirce skill score (PSS) on the training data.

The **baseline** rule declares onset at every wet spell (p = 1).

## Pipeline

```bash
uv run stopping-rule-2 aggregate     # CHIRPS grid -> woreda (area-weighted), data/processed/chirps_woreda.parquet
uv run stopping-rule-2 check-leads   # verify AIFS lead L == calendar day init+L
uv run stopping-rule-2 features      # base arrays, data/processed/base.npz (~2 min)
sbatch scripts/run_cv.sbatch         # train (LOYO CV + final model) and evaluate on slurm
uv run stopping-rule-2 report        # self-contained HTML report, outputs/report.html
uv run pytest
```

`train` and `evaluate` can also be run directly. They use `SLURM_CPUS_PER_TASK` workers, or all cores if that is not set. Use `train --years 2005,2015` for a quick smoke test.

## Data

| Source | Location | Notes |
|---|---|---|
| CHIRPS v3.0 daily, 0.25° | `data/CHIRPS/{1998..2025}.nc` | Aggregated to the woredas in `data/woreda_shapefile/`, weighted by the area of each cell that falls inside the woreda |
| AIFS ENS reforecasts | see `dsi_info.txt` (`config.AIFS_DIR`) | Already woreda-aggregated: 30 inits/yr (May 1–Aug 25), 25 members, leads 0–30, 2000–2024 |

The AIFS file was aggregated with a different boundary version than the shapefile. They share **326 woredas**, and only those are modelled.

## Design

- **Candidates:** wet-spell days t between May 1 and Aug 31. The model is trained and scored on candidates **on or before each woreda-year's true onset**: false starts plus the true onset, about 50% positive. These are the cases the stopping rule actually faces. Scores on all wet-spell days (about 96% positive) are reported for reference.
- **AIFS pairing:** each decision day D uses the latest init on or before D. Its forecast age k (0–9 days) is a feature, and the window D..D+20 maps to leads k..k+20.
- **Features:**

  | Group | Features | Source |
  |---|---|---|
  | Climatology | `clim_logit_success`: P(criteria met \| wet spell) | CHIRPS ±15-day DOY window |
  | | `clim_logit_dry7`: P(7-day dry spell in 21 days) | |
  | | `clim_log_rain21`: mean 21-day rainfall | |
  | | `doy_minus_clim_onset`: days from the median onset date | |
  | Realtime observations | `rain14_z`, `rain30_z`: rainfall anomalies over t−14..t−1 and t−30..t−1 | CHIRPS before the wet spell |
  | | `std_z`: season-to-date (from Apr 1) anomaly | |
  | | `dry14`: dry-day count in t−14..t−1 | |
  | AIFS forecast | `ens_logit_dry7`: fraction of members with a 7-day dry spell | Window D..D+20 |
  | | ensemble-mean longest dry spell and number of dry days | |
  | | ensemble-mean 21-day and week-1/2/3 rainfall | |
  | | `fcst_age` | |

  The AIFS dry-day threshold is set per woreda so that AIFS dry-day frequency matches CHIRPS (< 1 mm) on the training years.
- **No leakage:** climatology for a row from year y excludes both y and the held-out test year. AIFS thresholds are fitted on training years only.
- **Model:** StandardScaler → L2 logistic regression, pooled across woredas.
  - It is wrapped in `CalibratedClassifierCV` with sigmoid or isotonic calibration, whichever gives the lower Brier score on year-grouped inner CV.
  - It is evaluated with **leave-one-year-out** CV over 2000–2024.
  - Three feature-set ablations are run: climo / climo+obs / climo+obs+AIFS.

## Outputs (`outputs/`)

- `metrics.json`:
  - Brier score; BSS against the wet-spell baseline and against climatology (the training base rate); AUC; POD, POFD, FAR and PSS at the threshold.
  - Stopping-rule scores: false-onset rate, onset-date MAE and bias, share within ±7 days, early/late rates.
  - 95% confidence intervals from a year-block bootstrap.
- `oof_predictions.parquet`: out-of-fold probability for every wet-spell day.
- `cv_fold_info.csv`: calibration method and threshold chosen in each fold.
- `coefficients_*.csv`: logistic-regression coefficients per fold.
- `final_model.pkl`: operational model trained on all years.
- `report.html`: self-contained model and verification report.
- `figures/`: reliability, ROC, stopping-rule outcomes, onset-error histogram, per-woreda BSS maps and coefficients, each with a CSV twin.
