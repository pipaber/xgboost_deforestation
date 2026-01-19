# Peru Deforestation Area Prediction (District-Year)

## Rolling Time-Series Cross-Validation (recommended tuning)
Your first hyperparameter tuning used a **single validation window (2017–2018)**. That can overfit to that specific period and may not yield the best generalization to later years.

To reduce this risk, tune with a **rolling/expanding window time CV**:

- You evaluate multiple “future-like” validation slices (several folds).
- You select hyperparameters that perform well across folds (average RMSE/MAE), not just one window.
- After selecting best params, you train a final model on all years up to a cutoff (e.g., <=2018) and test on 2019–2020.

### How to run rolling time CV training (script)
Train/tune with rolling time CV and log all trials:

```/dev/null/commands.txt#L1-2
uv run python src/deforestation/train_xgb_timecv.py --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --trials 200 --seed 42 --outdir models --run-name xgb_timecv_v1
```

Artifacts are saved under:
- `models/xgb_timecv_v1/`
  - `model.json`
  - `bundle.joblib`
  - `metrics_report.json`
  - `feature_columns.json`
  - `test_predictions.csv`
  - `trials_log.csv`

### How to run SHAP on the rolling-CV-trained model
```/dev/null/commands.txt#L1-2
uv run python src/deforestation/explain_shap.py --bundle models/xgb_timecv_v1/bundle.joblib --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --split test --out reports/shap/xgb_timecv_v1
```

### Notes
- This project uses XGBoost on `log1p(Def_ha)` and inverts predictions with `expm1`.
- Numeric `NaN` values are allowed; XGBoost handles them natively.
- Categorical features are one-hot encoded.
- Extremely sparse (~90–95% missing) columns are excluded from the v1 feature set.

---

This repository trains and serves a machine learning model to **predict annual deforestation area (hectares)** at the **district (UBIGEO) × year** level for Peru.

The dataset file(s) currently in the repo:

- `deforestation_dataset_PERU.csv` (tab-separated)
- `deforestation_dataset_PERU_imputed_coca.csv` (semicolon-separated; coca was partially imputed upstream)
- `deforestation_dataset_PERU_imputed_coca_idh.csv` (semicolon-separated; contains `IDH_source`/`IDH_filled_from_xlsx` but IDH years are not fully annual)

Target variable:

- `Def_ha` — deforested area in hectares (continuous regression target)

---

## 0) Execution environment

Dependencies are managed via `uv` (`pyproject.toml`). When a package is missing, install it with:

- `uv add <package>`

Run scripts with:

- `uv run python <script.py>`

### 0.1 SHAP/Numba/Numpy compatibility (important)
SHAP requires `numba`, and `numba` requires **NumPy <= 2.3.x**.

If you see:
- `ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.`

Fix by pinning NumPy:
- `uv add "numpy<2.4"`

Then verify:
- `uv run python -c "import numpy as np, numba, shap; print(np.__version__, numba.__version__, shap.__version__)"`

---

## 1) Dataset quick evaluation (what you have)

### Shape & grain
From inspection of `deforestation_dataset_PERU.csv`:

- Rows: **7,880**
- Columns: **42**
- Time span: **2001–2020** (20 years)
- Panel key uniqueness: `(UBIGEO, YEAR)` is unique
- Spatial units:
  - `UBIGEO`: **~394** unique codes
- This is a panel dataset: **district-year observations**.

### Columns (high level)
- Identifiers / geo labels: `UBIGEO`, `NOMBDEP`, `NOMBPROB`, `NOMBDIST`, `Región`, `Cluster`, `YEAR`
- Target: `Def_ha`
- Potential drivers / correlates:
  - Land use / crops: `Coca_ha`, `Yuca_ha`, `area_agropec`, `form_boscosa`
  - Socioeconomic: `Pobreza`, `IDH`, `Pbi_dist`, `Población`, `dens_pob`, migration (`tot_salieron`, `tot_ingresaron`)
  - Climate / environment: `tmean`, `pp`, `hum_suelo`
  - Extractives / infra: `Minería`, `Infraestructura`
  - Accessibility / distance: `Dist_*` variables, `Dist_vías`, `Dist_ríos`
  - Government efficiency: `Efic_gasto`
  - Employment structure: `Emp_*`

---

## 2) New milestone: XGBoost model (with tuning, metrics, SHAP, scenarios)

This section defines the plan to build a production-quality XGBoost regression model, evaluate it correctly for panel/time/spatial data, explain it with SHAP, and run scenario simulations.

### 2.1 Why XGBoost (and how it handles NA)
- XGBoost can handle **numeric `NaN` values natively** by learning a “default direction” in tree splits.
- This is desirable here because many columns have large missingness; heavy imputation can inject artificial structure.
- We still need preprocessing for:
  - categorical features (one-hot/ordinal encoding)
  - leakage-safe splitting (time-aware, optionally group-aware)
  - ensuring consistent feature availability at prediction time

### 2.2 Which columns should NOT be used (v1 “safe core”)
To avoid leakage/memorization and reduce risk from extreme sparsity:

**Do not use as features (identifiers / text labels):**
- `UBIGEO` (pure ID; encourages memorization)
- `NOMBPROB`, `NOMBDIST` (names; can be unstable/duplicated)

**Use with caution (categorical context only, one-hot):**
- `NOMBDEP`, `Región`, `Cluster`

**Drop for v1 due to ~90–95% missingness (model will learn mostly “missingness patterns”):**
- Distances with ~95% missing:
  - `Dist_ríos`, `Dist_vías`, `Dist_comunid`, `Dist_conc_mad`
- Employment block (very sparse):
  - `Emp_estado`, `Emp_cientificos`, `Emp_tecnicos`, `Emp_oficinistas`, `Emp_servicios`, `Emp_agricultura`,
    `Emp_obreros_minasyenerg`, `Emp_obreros`, `Emp_nocalif_agric_indust`, `Emp_otros`
- Migration totals:
  - `tot_salieron`, `tot_ingresaron`

**High-missing features to consider later (v2+) after confirming quality/coverage:**
- `Efic_gasto`, `hum_suelo`, `Yuca_ha` (and any others with very high missingness)
- `IDH` is not annual; we will keep it as-is (NaN allowed) and optionally add missingness flags.

**Recommended v1 feature set (high coverage, plausible drivers):**
- Climate: `pp`, `tmean`
- Land/cover: `form_boscosa`, `area_agropec`
- Extractives/infra: `Minería`, `Infraestructura`
- Distances with full coverage: `Dist_ANP`, `Dist_cat_Min`, `Río_lago_u_océano`
- Socioeconomics with moderate missingness: `Población`, `dens_pob`, `Pbi_dist`
- Optional (if you want): `Coca_ha` and `Pobreza` after you finalize their merges and validate coverage
- Time: `YEAR` (and optionally year one-hot if needed)

### 2.3 Splitting strategy: do we need stratification?
You asked if stratified splitting is necessary given spatial correlation.

**For this problem, the most important thing is a time-aware split (to avoid leakage), not classic stratification.**
- Standard “stratified split” is designed for classification / preserving label distribution and does not directly solve spatial correlation.
- XGBoost can learn patterns correlated with geography via features (region, distances, land-use), but it can also “memorize” if you leak information through random splits.

**Recommended evaluation splits (in increasing strictness):**
1) **Time split (minimum requirement)**:
   - Train: 2001–2016
   - Val: 2017–2018
   - Test: 2019–2020

2) **Time split + spatial holdout (stress test)**:
   - Hold out some `UBIGEO` entirely (or hold out 1–2 regions/clusters) as an additional test set.
   - This tests generalization to unseen districts/areas.

3) **Group-aware time CV (for tuning)**:
   - Rolling/expanding windows by year,
   - With grouping constraints by `UBIGEO` if using lag features.

**Conclusion:** do *time-based splitting* for core metrics. Add spatial holdout as a robustness check. “Stratified splitting” is not the right tool here.

### 2.4 Target engineering
`Def_ha` is non-negative and skewed. Use:
- train on `log1p(Def_ha)`
- predict with `expm1(pred)`

### 2.5 Hyperparameter tuning plan (optimum selection)
We’ll tune XGBoost with a leakage-safe validation strategy:

- Primary objective: `reg:squarederror`
- Evaluation metrics:
  - MAE (ha)
  - RMSE (ha)
  - R²
  - Additionally: metrics on `Def_ha > 0` subset, because low/zero values can dominate some metrics.

Tuning approach:
1) Establish baselines:
   - “last year” baseline if lag feature exists
   - XGBoost with reasonable defaults
2) Use a time-based validation scheme (rolling-origin or fixed val years).
3) Use early stopping with a validation fold.
4) Search strategy:
   - start with random search (broad ranges) or Bayesian optimization
   - then narrow to a small grid around best region

Typical parameter space:
- `n_estimators` (with early stopping, can be large)
- `learning_rate`
- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `reg_alpha`, `reg_lambda`
- `gamma`

### 2.6 SHAP explainability plan (required)
We will add SHAP plots after training:
- Global:
  - SHAP summary (beeswarm) for feature impact ranking
  - SHAP bar plot (mean |SHAP|)
- Local:
  - Force plot or waterfall for a selected district-year
- Diagnostics:
  - Dependence plots for top features (e.g., `area_agropec`, `pp`, `tmean`, `Minería`)

Implementation notes:
- Requires `shap` + `numba` (and NumPy <= 2.3.x)
- For tree models, use `TreeExplainer`
- Script: `src/deforestation/explain_shap.py` (writes outputs to `reports/shap/<run>/`)

### 2.7 Scenario modeling plan (good / bad / incremental)
Scenario analysis should be explicit, reproducible, and tied to variables we can realistically “perturb”.

We will implement a scenario runner that:
1) Starts from a baseline feature row set for a chosen year (e.g., 2019 or 2020).
2) Applies deterministic transformations to selected drivers.
3) Re-scores the rows with the trained model.
4) Reports:
   - delta in predicted deforestation (ha)
   - top districts impacted
   - aggregate change by region/cluster

**Important constraint:**
We should only create scenarios on features that represent policy/pressure levers or plausible changes, such as:
- `Infraestructura` (increase/decrease)
- `Minería` (increase/decrease)
- `area_agropec` (expansion/contraction)
- climate anomalies: `pp`, `tmean` shifts (drought/heat conditions)

**Scenario set (v1 templates):**
- Good scenario (“strengthened protection / reduced pressure”):
  - decrease `Infraestructura` (or proxy) by X%
  - decrease `Minería` by X%
  - reduce `area_agropec` expansion by X%
- Bad scenario (“commodity boom + weak enforcement + road expansion”):
  - increase `Minería` by X%
  - increase `Infraestructura` by X%
  - increase `area_agropec` by X%
  - optionally add drought: reduce `pp`, increase `tmean`
- Incremental scenario (“gradual change consistent with reported trends”):
  - small increases in infrastructure/agriculture
  - mild climate anomaly

**About “based on news from Peru”:**
We will not hardcode claims. Instead:
- create a `scenarios/scenarios.yaml` (or CSV) with parameters and citations/links in comments/metadata,
- keep it versioned so the scenario assumptions are auditable.

---

## 3) Action plan (what we build next)

### Step 1 — Add dependencies
- `uv add xgboost joblib pyyaml`
- For SHAP: ensure `numpy<2.4`, then add `numba` + `shap`

### Step 2 — Implement training pipeline
- Data loader that can read `;` or `\t` correctly (based on input file)
- Column selection (drop high-missing blocks, drop IDs)
- Categorical encoding (one-hot)
- Train/val/test time split
- Train XGBoost on `log1p(Def_ha)`
- Report metrics table + save artifacts

### Step 3 — Hyperparameter tuning
- Fixed validation (2017–2018) + early stopping
- Record all trials to `trials_log.csv`
- Save best params + metrics report

### Step 4 — SHAP report
- Run `src/deforestation/explain_shap.py` against a trained model bundle:
  - writes SHAP summary + dependence + waterfall plots under `reports/shap/<run>/`

### Step 5 — Scenario runner
- Load trained model + baseline dataset slice
- Apply scenario config
- Output scenario report CSV + plots

---

## 4) Notes on next engineering steps (repo structure suggestion)
- `src/deforestation/`
  - `data.py` (load/clean/splits)
  - `train_xgb.py` (train + tune + metrics)
  - `explain_shap.py` (SHAP plots)
  - `scenarios.py` (scenario transformations + reporting)
- `models/` (saved artifacts)
- `reports/` (metrics, figures, shap plots)
- `scenarios/` (scenario configs with citations)

---

## 5) Current status (as of latest run)

### 5.1 Implemented training script
Training + tuning entry point:
- `src/deforestation/train_xgb.py`

Key behaviors:
- Time split:
  - Train: 2001–2016
  - Val: 2017–2018
  - Test: 2019–2020
- Target transform: train on `log1p(Def_ha)`, predict with `expm1`
- One-hot encoding for: `Región`, `NOMBDEP`, `Cluster`
- Drops extreme-missingness blocks (`Emp_*`, several `Dist_*`, migration totals) and identifier columns (`UBIGEO`, name fields)
- Feature-name sanitization to satisfy XGBoost restrictions (no `[`, `]`, `<`)
- Trial logging: `trials_log.csv` (append-only when using `--run-name`)
- Bundle portability: `bundle.joblib` stores configs as plain dicts (avoids pickle errors when loading from other scripts)

### 5.2 Hyperparameter tuning run (200 trials)
A random-search tuning run was executed with 200 trials on:
- data: `deforestation_dataset_PERU_imputed_coca.csv`
- separator: `;`
- run name / artifacts dir: `models/xgb_tune_v1/`

Best validation result:
- Best trial (global): 185
- Val RMSE ≈ 343.85 ha

Test set (2019–2020) metrics for the best validation model:
- MAE ≈ 218.75 ha
- RMSE ≈ 543.75 ha
- R² ≈ 0.676

All trials are recorded at:
- `models/xgb_tune_v1/trials_log.csv`

### 5.3 SHAP explainability (generated)
SHAP plots were generated from the trained model bundle:
- Bundle: `models/xgb_tune_v1/bundle.joblib`
- Data: `deforestation_dataset_PERU_imputed_coca.csv` (sep `;`)
- Split explained: `test`
- Output directory:
  - `reports/shap/xgb_tune_v1/`

Run command:
- `uv run python src/deforestation/explain_shap.py --bundle models/xgb_tune_v1/bundle.joblib --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --split test --out reports/shap/xgb_tune_v1`

Expected outputs:
- `shap_summary_beeswarm.png`
- `shap_summary_bar.png`
- `shap_mean_abs.csv`
- `top_features.txt`
- `shap_dependence_*.png`
- `shap_waterfall_example.png`
- `shap_run_metadata.json`

### 5.4 Spatial holdout evaluation (20% UBIGEO)
A spatial robustness evaluation was run by holding out 20% of districts (`UBIGEO`) entirely (seed=42) and testing only on held-out districts in 2019–2020.

Run command:
- `uv run python src/deforestation/eval_spatial_holdout.py --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --holdout-frac 0.2 --seed 42 --out reports/spatial_holdout_ubigeo20_seed42.json`

Holdout split:
- Train UBIGEO: 315
- Holdout UBIGEO: 79
- Rows:
  - Train (<=2016): 5040
  - Val (2017–2018): 630
  - Test (2019–2020, holdout UBIGEO): 158

Holdout test metrics (2019–2020, unseen UBIGEO):
- MAE ≈ 216.11 ha
- RMSE ≈ 612.04 ha
- R² ≈ 0.553

Interpretation:
- R² drops from the standard time-based test evaluation, which is expected when requiring generalization to unseen districts.
- This indicates the model is learning both global relationships and some district-specific patterns; to improve spatial generalization, consider:
  - adding stronger geography proxies (reliable distance/accessibility layers with full coverage),
  - simplifying the model (regularization, shallower trees),
  - tuning with a spatially aware validation scheme (group holdout) if deployment requires predictions for unseen districts.

### 5.5 Next required work
1) Scenario runner:
   - Implement `good`, `bad`, and `incremental` scenarios via deterministic feature perturbations, with parameters and citations stored in a versioned scenario config.