"""Self-contained HTML report on the model and its verification.

Every number is read from the pipeline outputs (metrics.json, CV fold info,
out-of-fold predictions, figure CSVs), so the report regenerates with the run.
"""

from __future__ import annotations

import base64
import html
import json
import pickle
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import plots

OUT = C.OUTPUT_DIR
FIG = OUT / "figures"
G_FULL, G_OBS, G_CLIMO = "climo+obs+aifs", "climo+obs", "climo"
GROUP_LABEL = {G_CLIMO: "Climatology", G_OBS: "+ realtime obs", G_FULL: "+ AIFS (full model)",
               "wet_spell": "Wet-spell baseline"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def img(name: str, alt: str) -> str:
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def figure(name, alt, caption):
    return f"<figure>{img(name, alt)}<figcaption>{caption}</figcaption></figure>"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def num(x, d=3):
    return f"{x:,.{d}f}"


def table(df: pd.DataFrame, fmt: dict | None = None, best: dict | None = None,
          index=True, caption=None) -> str:
    """HTML table; fmt maps column/row label -> formatter; best marks cells bold."""
    fmt = fmt or {}
    best = best or {}
    h = ['<table>']
    if caption:
        h.append(f"<caption>{caption}</caption>")
    h.append("<thead><tr>")
    if index:
        h.append(f'<th class="l">{html.escape(str(df.index.name or ""))}</th>')
    for c in df.columns:
        h.append(f"<th>{html.escape(str(c))}</th>")
    h.append("</tr></thead><tbody>")
    for i, row in df.iterrows():
        h.append("<tr>")
        if index:
            h.append(f'<th class="l">{html.escape(str(i))}</th>')
        for c, v in row.items():
            f = fmt.get(i, fmt.get(c, lambda x: html.escape(x) if isinstance(x, str) else num(x)))
            s = f(v) if not (isinstance(v, float) and np.isnan(v)) else "–"
            cls = ' class="best"' if best.get(i) == c or best.get(c) == i else ""
            h.append(f"<td{cls}>{s}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def lead_check() -> pd.DataFrame:
    path = OUT / "lead_check.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    from . import aifs, chirps

    corr = aifs.check_lead_alignment(chirps.load_woreda_series())
    corr.to_csv(path)
    return pd.read_csv(path, index_col=0)


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------
def gather():
    m = json.loads((OUT / "metrics.json").read_text())
    info = pd.read_csv(OUT / "cv_fold_info.csv")
    oof = pd.read_parquet(OUT / "oof_predictions.parquet")
    with open(OUT / "final_model.pkl", "rb") as f:
        final = pickle.load(f)
    base = F.load_base()
    ac, rows = base["ac"], base["rows"]
    kthr = F.choose_aifs_thresholds(ac, base["obs_cnt"], base["obs_tot"],
                                    np.ones(len(ac["years"]), bool))
    aifs_thr = pd.Series(ac["thresholds"][kthr])
    k1 = int(np.argmin(np.abs(ac["thresholds"] - C.DRY_DAY_MM)))
    n_p = len(base["pcodes"])
    drizzle = {"aifs": ac["dry_count"][..., k1].sum() / (ac["dry_total"].sum() * n_p),
               "chirps": base["obs_cnt"].sum() / (base["obs_tot"].sum() * n_p)}
    bss_w = pd.read_csv(FIG / "bss_by_woreda.csv")
    after = pd.read_csv(FIG / "forecast_vs_days_from_onset.csv", index_col=0)
    outcomes = pd.read_csv(FIG / "stopping_rule_outcomes.csv", index_col=0)
    coefs = pd.read_csv(OUT / "coefficients_climo_obs_aifs.csv", index_col=0)
    return dict(m=m, info=info, oof=oof, final=final, base=base, rows=rows,
                aifs_thr=aifs_thr, bss_w=bss_w, after=after, outcomes=outcomes,
                coefs=coefs, lead=lead_check(), drizzle=drizzle)


def build(d) -> str:
    m, info, rows, oof = d["m"], d["info"], d["rows"], d["oof"]
    pre = m["probabilistic_pre_onset"]
    allw = m["probabilistic_all_wet_days"]
    sr = m["stopping_rule"]
    ci = m.get("bootstrap_95ci", {})
    n_pc = len(d["base"]["pcodes"])
    n_wy = sr[G_FULL]["n_woreda_years"]
    after = d["after"]
    near = after.loc[(after.index >= 0) & (after.index < 15)]
    near_obs = (near["observed"] * near["n"]).sum() / near["n"].sum()
    near_below = (near["share_below_threshold"] * near["n"]).sum() / near["n"].sum()
    near_fc = (near["forecast"] * near["n"]).sum() / near["n"].sum()
    at0 = after.loc[0]
    sr_full = plots.E.simulate_stopping_rule(oof, G_FULL)
    late = sr_full.loc[sr_full["error"] > 0, "error"]
    late_q = late.quantile([0.25, 0.5, 0.75])
    sr_base = plots.E.simulate_stopping_rule(oof, None)
    early_b = sr_base.loc[sr_base["error"] < 0, "error"]
    thr_by = info.groupby("group")["threshold"].agg(["median", "min", "max"])
    meth = info.groupby(["group", "method"]).size().unstack(fill_value=0)
    bw = d["bss_w"]
    lead = d["lead"]
    lead_best = (lead.idxmax(axis=1) == "0").all()
    final = d["final"]
    aifs_thr = d["aifs_thr"]

    def ci_txt(key):
        c = ci.get(key)
        return f"{num(c['lo'], 2)}–{num(c['hi'], 2)}" if c else "–"

    def ci_pct(key):
        c = ci.get(key)
        return f"{pct(c['lo'], 1)}–{pct(c['hi'], 1)}" if c else "–"

    # ---------------- headline tiles
    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="label">False-onset rate</div>
    <div class="value">{pct(sr[G_FULL]['false_onset_rate'], 1)}</div>
    <div class="sub">vs {pct(sr['wet_spell']['false_onset_rate'], 1)} for the wet-spell baseline</div></div>
  <div class="tile"><div class="label">BSS vs wet-spell baseline</div>
    <div class="value">{num(pre[G_FULL]['BSS_vs_wet_spell'], 2)}</div>
    <div class="sub">95% CI {ci_txt(f'BSS_vs_wet_spell[{G_FULL}]')}, up-to-onset cases</div></div>
  <div class="tile"><div class="label">BSS vs climatology</div>
    <div class="value">{num(pre[G_FULL]['BSS_vs_climatology'], 2)}</div>
    <div class="sub">{num(pre[G_OBS]['BSS_vs_climatology'], 2)} without AIFS</div></div>
  <div class="tile"><div class="label">Onset-date error (mean absolute)</div>
    <div class="value">{num(sr[G_FULL]['MAE_days'], 1)} d</div>
    <div class="sub">vs {num(sr['wet_spell']['MAE_days'], 1)} d for the baseline: not yet better</div></div>
</div>"""

    # ---------------- tables
    def prob_table(src, caption):
        rows_ = {
            "Cases": ("n", lambda x: f"{int(x):,}"),
            "Share meeting criteria": ("base_rate", lambda x: pct(x, 1)),
            "Brier score": ("BS", lambda x: num(x, 3)),
            "BSS vs wet-spell baseline": ("BSS_vs_wet_spell", lambda x: num(x, 3)),
            "BSS vs climatology": ("BSS_vs_climatology", lambda x: num(x, 3)),
            "AUC": ("AUC", lambda x: num(x, 3)),
            "POD (hit rate)": ("POD", lambda x: num(x, 3)),
            "POFD (false-alarm rate)": ("POFD", lambda x: num(x, 3)),
            "FAR (false-alarm ratio)": ("FAR", lambda x: num(x, 3)),
            "PSS": ("PSS", lambda x: num(x, 3)),
        }
        df = pd.DataFrame({GROUP_LABEL[g]: {k: src[g][v[0]] for k, v in rows_.items()}
                           for g in (G_CLIMO, G_OBS, G_FULL)})
        df.index.name = "Score"
        fmt = {k: v[1] for k, v in rows_.items()}
        hi = ["BSS vs wet-spell baseline", "BSS vs climatology", "AUC", "POD (hit rate)", "PSS"]
        lo = ["Brier score", "POFD (false-alarm rate)", "FAR (false-alarm ratio)"]
        best = {r: df.loc[r].idxmax() for r in hi} | {r: df.loc[r].idxmin() for r in lo}
        return table(df, fmt, best, caption=caption)

    t_pre = prob_table(pre, "Wet-spell days on or before the true onset (primary).")
    t_all = prob_table(allw, "All wet-spell days, May 1–Aug 31 (reference only).")

    sr_rows = {
        "False-onset rate": ("false_onset_rate", lambda x: pct(x, 1)),
        "Exact day": ("exact_rate", lambda x: pct(x, 1)),
        "Within ±7 days": ("within_7d", lambda x: pct(x, 1)),
        "Early (= false onset)": ("early_rate", lambda x: pct(x, 1)),
        "Late": ("late_rate", lambda x: pct(x, 1)),
        "Never declared": ("declared_rate", lambda x: pct(1 - x, 1)),
        "Mean absolute error (days)": ("MAE_days", lambda x: num(x, 1)),
        "Mean error / bias (days)": ("bias_days", lambda x: f"{x:+.1f}"),
    }
    order = ["wet_spell", G_CLIMO, G_OBS, G_FULL]
    df_sr = pd.DataFrame({GROUP_LABEL[g]: {k: sr[g][v[0]] for k, v in sr_rows.items()} for g in order})
    df_sr.index.name = "Outcome"
    best_sr = {"False-onset rate": df_sr.loc["False-onset rate"].idxmin(),
               "Exact day": df_sr.loc["Exact day"].idxmax(),
               "Within ±7 days": df_sr.loc["Within ±7 days"].idxmax(),
               "Mean absolute error (days)": df_sr.loc["Mean absolute error (days)"].idxmin()}
    t_sr = table(df_sr, {k: v[1] for k, v in sr_rows.items()}, best_sr,
                 caption=f"{n_wy:,} woreda-years, 2000–2024 (leave-one-year-out predictions). The false-onset rate is the share of "
                         "declared woreda-years. Exact, within ±7, early, late and the errors are over woreda-years "
                         "with both a declaration and a true onset.")

    ci_rows = []
    for g in order:
        ci_rows.append({"Rule": GROUP_LABEL[g],
                        "BSS vs wet-spell (95% CI)": ci_txt(f"BSS_vs_wet_spell[{g}]") if g != "wet_spell" else "0 (reference)",
                        "False-onset rate (95% CI)": ci_pct(f"false_onset_rate[{g}]"),
                        "MAE days (95% CI)": ci_txt(f"MAE_days[{g}]")})
    t_ci = table(pd.DataFrame(ci_rows).set_index("Rule"),
                 caption=f"Year-block bootstrap ({C.N_BOOTSTRAP} resamples of the 25 years).")

    feats = pd.DataFrame([
        ("Climatology", "clim_logit_success", "P(criteria met | wet spell) for this woreda and date (±15 d), logit"),
        ("Climatology", "clim_logit_dry7", "P(7-day dry spell in the 21-day window), logit"),
        ("Climatology", "clim_log_rain21", "Mean 21-day rainfall after the wet spell, log"),
        ("Climatology", "doy_minus_clim_onset", "Days from the woreda's median true-onset date"),
        ("Realtime obs", "rain14_z, rain30_z", "Rainfall anomaly over t−14..t−1 and t−30..t−1 (before the wet spell)"),
        ("Realtime obs", "std_z", "Season-to-date rainfall anomaly (Apr 1 to t−1)"),
        ("Realtime obs", "dry14", "Dry days (< 1 mm) in t−14..t−1"),
        ("AIFS", "ens_logit_dry7", "Share of the 25 members with a 7-day dry spell in D..D+20, logit"),
        ("AIFS", "ens_mean_maxdry, ens_mean_drydays", "Ensemble-mean longest dry run and number of dry days"),
        ("AIFS", "ens_log_rain21, ens_log_wk1..3", "Ensemble-mean 21-day and week-1/2/3 rainfall, log"),
        ("AIFS", "fcst_age", "Days between the AIFS init and decision day D (0–9)"),
    ], columns=["Group", "Feature", "Definition"]).set_index("Group")

    t_feats = table(feats, caption="All three groups enter one pooled model; ablations add them in turn.")

    lead_tab = lead.copy()
    lead_tab.columns = [f"day init+L{int(c):+d}" for c in lead_tab.columns]
    lead_tab.index.name = "Lead L"
    t_lead = table(lead_tab, {c: (lambda x: num(x, 3)) for c in lead_tab.columns},
                   {i: lead_tab.loc[i].idxmax() for i in lead_tab.index},
                   caption="Correlation of AIFS ensemble-mean rainfall at lead L with CHIRPS on "
                           "different calendar days (all woredas, inits, years).")

    after_show = after.loc[[0, 1, 3, 6, 9, 12, 15, 21, 30, 45]].copy()
    after_show.index = ["0 (true onset)", "1–2", "3–5", "6–8", "9–11", "12–14", "15–17",
                        "21–23", "30–32", "45–47"]
    after_show.index.name = "Days after true onset"
    after_show = after_show.rename(columns={"n": "Cases", "observed": "Observed share meeting criteria",
                                            "forecast": "Mean forecast", "share_below_threshold": "Share rejected"})
    t_after = table(after_show, {"Cases": lambda x: f"{int(x):,}",
                                 "Observed share meeting criteria": lambda x: pct(x, 1),
                                 "Mean forecast": lambda x: num(x, 2),
                                 "Share rejected": lambda x: pct(x, 1)})

    coefs = d["coefs"].agg(["mean", "min", "max"]).T
    flips = coefs[(coefs["min"] < 0) & (coefs["max"] > 0)].index.tolist()

    meth_txt = "; ".join(
        f"{GROUP_LABEL[g]}: isotonic in {int(meth.loc[g].get('isotonic', 0))}, "
        f"sigmoid in {int(meth.loc[g].get('sigmoid', 0))}" for g in (G_CLIMO, G_OBS, G_FULL))

    n_pos_bss = (bw["BSS_vs_wet_spell"] > 0).mean()
    n_pos_gain = (bw["AIFS_gain"] > 0).mean()
    neg = bw["BSS_vs_wet_spell"] < 0
    chb = d["base"]["ch"]
    yi = np.searchsorted(chb["years"], C.MODEL_YEARS)
    april = (chb["label"][yi][:, : F.H] == 1).any(axis=1)  # criteria met Apr 16-30
    on_may1 = chb["onset"][yi] == 0
    april_given_may1 = april[on_may1].mean()
    br_neg, br_pos = bw.loc[neg, "base_rate"].median(), bw.loc[~neg, "base_rate"].median()

    # ---------------- assemble
    body = f"""
<header>
  <p class="eyebrow">Rainy-season onset · Amhara &amp; Oromia woredas</p>
  <h1>A dry-spell detector for declaring onset after a wet spell</h1>
  <p class="lede">A calibrated logistic regression that uses CHIRPS climatology, realtime CHIRPS and AIFS ENS forecasts.
  It cuts false onsets from {pct(sr['wet_spell']['false_onset_rate'])} to {pct(sr[G_FULL]['false_onset_rate'])} of woreda-years
  compared with declaring at the first wet spell. It is not yet better at timing: when it turns down the true onset, it
  tends to wait several weeks.</p>
  <p class="meta">Leave-one-year-out cross-validation, 2000–2024 · {n_pc} woredas · generated {date.today().isoformat()}</p>
</header>

{tiles}

<section>
<h2>1. Summary</h2>
<ul>
  <li><b>The probabilities are skilful and well calibrated.</b> On the cases the stopping rule actually faces (wet-spell days on or before the true
  onset; {pct(pre[G_FULL]['base_rate'])} of them meet the criteria), the full model scores BSS = {num(pre[G_FULL]['BSS_vs_wet_spell'], 2)} against the
  wet-spell baseline and {num(pre[G_FULL]['BSS_vs_climatology'], 2)} against climatology, with AUC {num(pre[G_FULL]['AUC'], 2)}.</li>
  <li><b>AIFS adds skill beyond climatology and observations:</b> BSS against climatology rises from {num(pre[G_OBS]['BSS_vs_climatology'], 2)} to
  {num(pre[G_FULL]['BSS_vs_climatology'], 2)}, AUC from {num(pre[G_OBS]['AUC'], 2)} to {num(pre[G_FULL]['AUC'], 2)}, and PSS from
  {num(pre[G_OBS]['PSS'], 2)} to {num(pre[G_FULL]['PSS'], 2)}.</li>
  <li><b>As a stopping rule it mostly prevents false onsets:</b> {pct(sr[G_FULL]['false_onset_rate'], 1)} of woreda-years
  (95% CI {ci_pct(f'false_onset_rate[{G_FULL}]')}), against {pct(sr['wet_spell']['false_onset_rate'], 1)}
  ({ci_pct('false_onset_rate[wet_spell]')}) for the wet-spell baseline.</li>
  <li><b>Timing is not yet better.</b> The mean absolute onset-date error is {num(sr[G_FULL]['MAE_days'], 1)} days against
  {num(sr['wet_spell']['MAE_days'], 1)} for the baseline, and the confidence intervals overlap. The baseline is early in {pct(sr['wet_spell']['early_rate'])} of woreda-years;
  the model is late in {pct(sr[G_FULL]['late_rate'])}, by a median of {late_q[0.5]:.0f} days. The cause and some fixes are in Section 6.</li>
</ul>
</section>

<section>
<h2>2. Problem and onset definition</h2>
<p><b>Onset</b> on day <i>t</i> means two things hold:</p>
<ul>
  <li><b>Wet spell:</b> rainfall on days <i>t</i>..<i>t</i>+2 totals at least {C.WET_SPELL_MM:.0f} mm.</li>
  <li><b>No dry spell after it:</b> the next {C.CHECK_DAYS} days (<i>t</i>+3..<i>t</i>+23) contain no run of
  {C.DRY_SPELL_DAYS} consecutive dry days, where a dry day has less than {C.DRY_DAY_MM:.0f} mm.</li>
</ul>
<p>The <b>true onset</b> of a woreda-year is the first day between May 1 and Aug 31 that meets both conditions.</p>
<p>In operation the model is used only <i>after</i> a wet spell has been observed, so it is a <b>dry-spell detector</b>:</p>
<ul>
  <li>On decision day <i>D</i> = <i>t</i>+3, it gives the probability that the window <i>D</i>..<i>D</i>+20 has no 7-day dry spell.</li>
  <li>The <b>stopping rule</b> declares onset at the first wet spell whose probability reaches a threshold, and then stops.</li>
  <li>The <b>baseline</b> declares onset at every wet spell (probability 1), so in practice at the first wet spell of the season.</li>
</ul>
</section>

<section>
<h2>3. Data</h2>
<table>
<thead><tr><th class="l">Source</th><th class="l">Use</th><th class="l">Details</th></tr></thead>
<tbody>
<tr><th class="l">CHIRPS v3.0 daily</th><td class="l">Labels, climatology, realtime observations</td>
<td class="l">0.25° grid (CDO remapcon), 1998–2025 (2025 to Oct 31). Aggregated to woredas using the area of each
grid cell inside the woreda. On a check of large woredas, this agrees with a cell-centre mask at r ≈ 0.997–0.998. A woreda averages about 4 cells.</td></tr>
<tr><th class="l">AIFS ENS reforecasts</th><td class="l">Forecast features</td>
<td class="l">Already aggregated to woredas. 30 inits per year (May 1–Aug 25; at most 6 days apart), 25 members, leads 0–30 days, 2000–2024.</td></tr>
<tr><th class="l">Woreda shapefile</th><td class="l">Spatial units</td>
<td class="l">346 Amhara and Oromia woredas (<code>adm3_pcode</code>). The AIFS file uses a different boundary version
(mostly West Gojam and the zones around Addis Ababa). <b>{n_pc} woredas</b> appear in both, and only those are modelled.</td></tr>
</tbody></table>
<p><b>AIFS lead alignment.</b> {"Lead <i>L</i> correlates best with CHIRPS on calendar day init+<i>L</i> at every lead, so lead 0 is the init day itself." if lead_best else "The best-matching offset varies by lead; see the table."}</p>
{t_lead}
<p><b>AIFS drizzle.</b> Pooled over all woredas, years and leads 0–20, {pct(d['drizzle']['aifs'])} of AIFS member-days fall below 1 mm,
against {pct(d['drizzle']['chirps'])} of CHIRPS days.
For each woreda, the AIFS dry-day threshold is set so that AIFS dry-day frequency matches CHIRPS on the training years.
Across woredas the chosen threshold has median {aifs_thr.median():.1f} mm, ranging from {aifs_thr.min():.1f} to {aifs_thr.max():.1f} mm.</p>
</section>

<section>
<h2>4. Method</h2>
<h3>Candidate cases</h3>
<ul>
  <li>There are {len(rows):,} wet-spell days between May 1 and Aug 31 (2000–2024). {pct(rows['y'].mean(), 1)} of them meet the onset criteria, mostly easy mid-season days.</li>
  <li>The model is trained and scored on the <b>{int(rows['pre_onset'].sum()):,} wet-spell days on or before each woreda-year's true onset</b>,
  that is, the false starts plus the true onset. {pct(rows.loc[rows.pre_onset, 'y'].mean(), 1)} of these meet the criteria. These are the cases the stopping rule faces.</li>
  <li>Predictions are made for every wet-spell day so that the stopping rule can be simulated over the whole season.</li>
</ul>
<h3>Timing and AIFS pairing</h3>
<p>For each candidate <i>t</i>:</p>
<ul>
  <li>The decision day is <i>D</i> = <i>t</i>+3, the day after the wet spell, which is fully observed in CHIRPS. Realtime CHIRPS is assumed to have no delay.</li>
  <li>The model uses the latest AIFS init on or before <i>D</i>. The window <i>D</i>..<i>D</i>+20 then corresponds to leads <i>k</i>..<i>k</i>+20, where the forecast age <i>k</i> is 0–9 days and is also a feature.</li>
</ul>
<h3>Features</h3>
{t_feats}
<h3>Model, calibration and validation</h3>
<ul>
  <li><b>Model:</b> standardized features, then an L2 logistic regression pooled across woredas. Climatology features carry the local information.</li>
  <li><b>Calibration:</b> <code>CalibratedClassifierCV</code>, sigmoid or isotonic, whichever gives the lower Brier score in year-grouped inner CV
  ({meth_txt} of the 25 folds).</li>
  <li><b>Validation:</b> leave-one-year-out cross-validation over 2000–2024.</li>
  <li><b>No leakage:</b> climatology for a row from year <i>y</i> leaves out both <i>y</i> and the test year. The AIFS dry-day thresholds are fitted on training years only.</li>
  <li><b>Declaration threshold:</b> the one that maximizes PSS on inner-CV predictions for the training years.
  For the full model it had median {thr_by.loc[G_FULL, 'median']:.2f} across folds (range {thr_by.loc[G_FULL, 'min']:.2f}–{thr_by.loc[G_FULL, 'max']:.2f}).</li>
  <li><b>Operational model:</b> fitted on all years, with {final['calibration']} calibration and threshold {final['threshold']:.2f}.</li>
</ul>
</section>

<section>
<h2>5. Results</h2>
<h3>5.1 Probabilistic skill</h3>
{t_pre}
<p>The baseline always says yes, so its Brier score equals the share of cases that fail
({num(pre[G_FULL]['BS_wet_spell_baseline'], 3)}). The climatology reference forecasts the training-years share of cases
meeting the criteria. Bold marks the best value in each row.</p>
<div class="grid2">
{figure("reliability.png", "Reliability diagram", "Reliability on the up-to-onset cases. All three models sit close to the diagonal. The full model puts more cases at the extremes (lower panel), which is where its extra skill comes from.")}
{figure("roc.png", "ROC curves", "ROC curves. The wet-spell baseline is the single point (1, 1): it detects every onset and also accepts every false start.")}
</div>
<details><summary>Scores on all wet-spell days (reference)</summary>
{t_all}
<p>On all wet-spell days, {pct(allw[G_FULL]['base_rate'], 1)} meet the criteria, so the baseline's Brier score is only
{num(allw[G_FULL]['BS_wet_spell_baseline'], 3)}. Models trained on the up-to-onset cases under-forecast the easy
post-onset days, so the BSS against the baseline is negative here for models with observation features. This is the same effect described in Section 6.</p>
</details>

<h3>5.2 Stopping rule</h3>
{t_sr}
{figure("stopping_rule_outcomes.png", "Stopping-rule outcomes", "Outcome of each rule against the true onset. Because the true onset is the first day that meets the criteria, any earlier declaration is a false onset.")}
{t_ci}
{figure("onset_error_hist.png", "Onset-date error histogram",
        f"Error of the non-exact declarations (exact day: baseline {pct(sr['wet_spell']['exact_rate'], 1)}, full model {pct(sr[G_FULL]['exact_rate'], 1)}). "
        f"The baseline's misses are early (median {early_b.median():.0f} d). The model's misses are mostly late (median +{late_q[0.5]:.0f} d, interquartile range {late_q[0.25]:.0f}–{late_q[0.75]:.0f} d).")}

<h3>5.3 Spatial skill</h3>
{figure("bss_maps.png", "Maps of per-woreda BSS",
        f"Per-woreda BSS of the full model against the wet-spell baseline (left; positive in {pct(n_pos_bss)} of woredas) and the change from adding AIFS (right; positive in {pct(n_pos_gain)}). "
        f"Colour scales are clipped (arrows). In the {int(neg.sum())} woredas with negative BSS, a median {pct(br_neg)} of cases meet the criteria (against {pct(br_pos)} elsewhere). "
        "Where almost every case meets them, the baseline's Brier score is tiny and BSS becomes unstable.")}

<h3>5.4 Feature effects</h3>
{figure("coefficients.png", "Coefficient plot", "Standardized coefficients of the full model across CV folds.")}
<p class="note"><b>Read the coefficients with care.</b> Several features are nearly collinear: 21-day rainfall versus the week-1/2/3 totals,
the three AIFS dry-spell measures, and the two climatology probabilities. Some coefficients have counter-intuitive signs, for example
<code>ens_logit_dry7</code> &gt; 0, and {"none" if not flips else ("one" if len(flips) == 1 else len(flips))} {"changes" if len(flips) == 1 else "change"} sign between folds{(" (" + ", ".join(f"<code>{f}</code>" for f in flips) + ")") if flips else ""}.
The calibrated probabilities are unaffected, but a coefficient should not be read as the effect of one predictor on its own.</p>
</section>

<section>
<h2>6. Why the stopping rule is late</h2>
{figure("forecast_vs_days_from_onset.png", "Forecast vs days from onset",
        "Mean forecast probability and observed share meeting the criteria, by days relative to the true onset. Days before the true onset fail by definition.")}
<p>From the true onset to two weeks after it, {pct(near_obs, 1)} of wet-spell days meet the criteria, but the model's mean forecast is only
{num(near_fc, 2)} and {pct(near_below)} of those days score below the declaration threshold. On the true-onset day itself, {pct(at0['share_below_threshold'])} are rejected.
These days look like the early-season cases the model was trained on, so if it rejects the true onset it usually goes on rejecting the following days.
It catches up only as the season settles (about 3 weeks later).</p>
{t_after}
<p><b>Possible fixes</b> (not yet tried; each needs re-validation):</p>
<ol>
  <li><b>Tune the threshold for the stopping rule itself.</b> Choose it on training years to minimize a cost that combines false onsets and days late, rather than per-case PSS.</li>
  <li><b>Widen the training set.</b> Also include wet-spell days in the first 2–3 weeks after the true onset, so the model sees the cases it gets wrong in operation.</li>
  <li><b>Lower the threshold as the season goes on.</b> Base this on days since the first wet spell, or on climatological onset progress.</li>
  <li><b>Drop overlapping features.</b> Use either 21-day or weekly AIFS rainfall, and one AIFS dry-spell measure. This makes the coefficients easier to interpret.</li>
</ol>
</section>

<section>
<h2>7. Limitations</h2>
<ul>
  <li><b>Woreda coverage:</b> only {n_pc} of 346 woredas have AIFS data, and the AIFS woreda averages were not produced with this shapefile.</li>
  <li><b>Coarse rainfall:</b> CHIRPS at 0.25° gives about 4 cells per woreda, which smooths out local dry spells.</li>
  <li><b>Data delay:</b> realtime CHIRPS is assumed to be available without delay; in practice preliminary CHIRPS is released with a lag of a few days.</li>
  <li><b>Reforecasts vs operations:</b> reforecasts exist only on 30 fixed dates per year, so forecast ages of 0–9 days appear in validation. Operational AIFS runs daily, so forecasts would be fresher.</li>
  <li><b>Season window:</b> searching only from May 1 means {pct(sr_full['true_onset'].eq(0).mean())} of woreda-years have their true onset on May 1. In {pct(april_given_may1)} of these, the criteria were already met on some day between Apr 16 and 30, so the season had begun before the search window.</li>
  <li><b>Correlated woredas:</b> woredas within a year are spatially correlated, so the year-block bootstrap is the honest measure of uncertainty (25 independent seasons).</li>
</ul>
</section>

<section>
<h2>8. Reproduce</h2>
<pre>uv run stopping-rule-2 aggregate      # CHIRPS → woreda series
uv run stopping-rule-2 features       # base arrays (~2 min)
sbatch scripts/run_cv.sbatch          # CV + final model + metrics + figures (~1 min on 25 CPUs)
uv run stopping-rule-2 report         # this report → outputs/report.html
uv run pytest                         # onset definition and leakage-control tests</pre>
<p>The CSV files behind every figure are in <code>outputs/figures/</code>, and all scores are in <code>outputs/metrics.json</code>.</p>
</section>
"""
    return TEMPLATE.replace("{{BODY}}", body)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Onset stopping rule: model and verification report</title>
<style>
.viz-root, :root { color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --accent:#2a78d6; --border:rgba(11,11,11,0.10); }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 40px 28px 80px; background:var(--surface-1); }
header { border-bottom:1px solid var(--grid); padding-bottom: 20px; margin-bottom: 24px; }
.eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:12px; margin:0 0 6px; }
h1 { font-size: 28px; line-height:1.2; margin: 0 0 12px; font-weight: 650; }
.lede { font-size: 17px; color: var(--ink2); margin: 0 0 10px; max-width: 72ch; }
.meta { color: var(--muted); font-size: 13px; margin:0; }
h2 { font-size: 20px; margin: 40px 0 12px; padding-top: 12px; border-top:1px solid var(--grid); font-weight:650; }
h3 { font-size: 16px; margin: 26px 0 8px; font-weight:650; }
p, li { max-width: 78ch; }
ul, ol { padding-left: 22px; }
li { margin: 4px 0; }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; background:#f0efec; padding:1px 4px; border-radius:4px; }
pre { font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; background:#f0efec; padding:14px 16px; border-radius:8px; overflow-x:auto; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 8px 0 8px; }
.tile { border:1px solid var(--border); border-radius:10px; padding:14px 16px; background:var(--surface-1); }
.tile .label { color:var(--ink2); font-size:13px; }
.tile .value { font-size: 32px; font-weight: 600; margin: 2px 0; }
.tile .sub { color:var(--muted); font-size: 12.5px; }
table { border-collapse: collapse; margin: 12px 0 16px; font-size: 13.5px; width:100%; }
caption { caption-side: top; text-align:left; color:var(--ink2); font-size:13px; padding-bottom:6px; }
th, td { padding: 6px 10px; border-bottom:1px solid var(--grid); text-align:right; font-variant-numeric: tabular-nums; vertical-align: top; }
thead th { color:var(--ink2); font-weight:600; border-bottom:1px solid var(--axis); }
th.l, td.l { text-align:left; }
tbody th { font-weight: 500; color: var(--ink); }
td.best { font-weight: 700; }
figure { margin: 18px 0; }
figure img { width:100%; height:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface-1); }
figcaption { color:var(--ink2); font-size:13px; margin-top:6px; max-width: 90ch; }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 820px) { .grid2 { grid-template-columns: 1fr; } }
details { margin: 10px 0 18px; border:1px solid var(--border); border-radius:8px; padding: 8px 14px; }
summary { cursor:pointer; color:var(--ink2); font-weight:600; }
.note { background:#f0efec; border-radius:8px; padding:10px 14px; }
@media print { body { background:#fff; } main { padding: 0; } details { display:block; } }
</style></head>
<body><main class="viz-root">{{BODY}}</main></body></html>
"""


def make_report(path=OUT / "report.html") -> str:
    oof = pd.read_parquet(OUT / "oof_predictions.parquet")
    plots.make_all(oof)
    path.write_text(build(gather()), encoding="utf-8")
    return str(path)
