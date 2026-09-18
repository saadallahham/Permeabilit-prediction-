# Synthetic carbonate permeability from five wells

Python scripts generate synthetic well logs and laboratory measurements for **five wells**, train fuzzy logic (FL), an artificial neural network (ANN), and a genetic algorithm (GA) blend on **ENC1, ENC2 and ENC3**, and evaluate the frozen models on **ENC4 and ENC5**. The pipeline creates all figure types in the supplied *manuscript permeability v5.docx*, extending the well-specific figures to all five wells.

**This is a synthetic methodological adaptation, not a numerical reproduction of the paper.** No real well data, original figures, or confidential manuscript files are distributed. Every figure is labeled synthetic or schematic. Model rankings and reported statistics come from the generated data; they are not adjusted to match the paper.

## Run

**Using Spyder?** Open `spyder_permeability.py` and click Run file. This standalone version has editable settings at the top, displays figures, and leaves results in the Variable Explorer. See [Spyder instructions](SPYDER_README.md).

Use Python 3.11 or newer (tested locally on Python 3.12).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
```

Open `outputs/figures/GALLERY.md` for the complete gallery. Each figure is saved as a PNG and a vector PDF. The default run produces **19 figures and 6 result tables**, each in both formats. The manuscript is not needed to run any script.

```bash
python run_pipeline.py --config config.json --output outputs
python run_pipeline.py --output outputs_quick --skip-figures
python predict_new_logs.py outputs/data/well_logs.csv --output new_predictions.csv
```

The `--output` directory is overwritten for matching filenames on reruns. Use a separate output directory for each experiment. `requirements-lock.txt` records the exact packages used for the delivered run; `requirements.txt` allows compatible installations across platforms. For the same environment, seed and configuration, numerical results are deterministic. Minor numerical variation across numerical libraries or platforms is possible.

## Data

| Well | Role | Log rows | Lab samples |
|---|---|---:|---:|
| ENC1 | Train | 1335 | 100 |
| ENC2 | Train | 1150 | 90 |
| ENC3 | Train | 929 | 80 |
| ENC4 | Blind test | 1100 | 85 |
| ENC5 | Blind test | 1000 | 75 |

Depth spans 0–250 m relative to the top of the synthetic interval. Beds contain grainstone, grainstone/packstone, mudstone, and packstone/wackestone. Porosity, clay fraction, fluid saturation and mineral density drive depth-correlated log responses. Permeability depends on porosity, fabric and unresolved heterogeneity. Sparse laboratory samples include independent measurement noise. `config.json` controls counts, seed, model settings and the whole-well split.

| CSV field | Meaning / unit | Model input? |
|---|---|---|
| well, sample_id | Unique joint sampling key | No |
| depth_m | Relative interval depth, m | No |
| split | train / test, assigned by well | No |
| facies, facies_code | Synthetic core description / code 0–3 | No |
| GR_API | Gamma ray, API | Yes |
| RHOB_g_cm3 | Bulk density, g/cm³ | Yes |
| NPHI_vv | Neutron porosity, fraction | Yes |
| DT_us_ft | Sonic slowness, µs/ft | Yes |
| RT_ohm_m | Resistivity, Ω m | No; figure only |
| SWT_vv | Total water saturation, fraction | No; figure only |
| SWIRR_vv | Synthetic irreducible water saturation, fraction | No; Timur only |
| PHI_vv | Synthetic porosity, fraction | No; Timur only |
| k_timur_mD | Empirical comparison, mD | No |
| porosity_lab_vv | Noisy laboratory porosity, fraction | No |
| k_lab_mD | Laboratory permeability, mD | Target |

The Timur comparison is `0.136 * (100*PHI_vv)**4.4 / (100*SWIRR_vv)**2`. Irreducible saturation is separate from total saturation. This empirical relation can be poorly calibrated for these synthetic carbonates; it is not used to generate the target. The percent-unit convention is discussed in [Huet's Texas A&M thesis, Appendix C](https://blasingame.engr.tamu.edu/0_TAB_Grad/TAB_Grad_Thesis_Archive/MS_036_HUET_Caroline_Thesis_TAMU_%28Dec_2005%29.pdf).

## Modeling and leakage prevention

1. Select laboratory samples from the three training wells only. Laboratory depths match log sample IDs exactly, avoiding target interpolation.
2. Generate FL and ANN out-of-fold predictions with one entire training well withheld in each of three folds. Each fold fits its own median imputer and standard scaler.
3. Fit a real-coded GA to these training-only out-of-fold predictions, minimizing log10 RMSE. The GA uses tournament selection, arithmetic crossover, Gaussian mutation and elitism. It estimates two nonnegative weights in [0, 2], without a sum-to-one constraint: `k_GA = w_FL*k_FL + w_ANN*k_ANN` in mD.
4. Refit FL and ANN using all three training wells. Freeze both models, their preprocessing, and the GA weights.
5. Predict all logs. Reveal test laboratory targets only for evaluation and plotting. Descriptive lines in test crossplots are post-evaluation summaries, not recalibration or fitted prediction models.

FL is a first-order Takagi–Sugeno fuzzy system with Gaussian product memberships, nine data-derived rule centers, and ridge-fitted linear consequents. ANN is a (16, 8) tanh MLP with L-BFGS optimization and fixed regularization. Both learn `log10(k/mD)`; their predictions are converted back to mD. Broad fixed numerical bounds of 10^-4 to 10^5 mD prevent numerical overflow. Architecture and rule choices are explicit adaptations because the manuscript does not supply enough detail to reconstruct its original fitted models. See the official [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) and [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html) documentation.

No test labels, facies, depth, well ID, laboratory porosity, generated porosity or permeability-derived columns enter the predictors. Test data are not used for scaling, rule construction, early stopping, weight optimization or hyperparameter selection. GA out-of-fold calibration scores are not an independent performance estimate of the blend; only the two untouched wells provide its blind evaluation. Training figures show in-sample fits and are labeled accordingly. The test suite perturbs both test features and test labels and verifies identical fitted weights and predictions on an unchanged input.

## Metrics

`outputs/results/metrics.csv` reports each well and pooled train/test results for FL, ANN, GA and Timur:

- `SSE_mD2 = sum((prediction - laboratory)**2)`.
- `RMSE_mD = sqrt(SSE / n)`.
- `R2 = 1 - SSE / sum((laboratory - mean(laboratory))**2)`; negative values are valid.
- `Pearson_r2` is the squared Pearson correlation, distinct from predictive R².
- `RMSE_log10` measures error in log10 permeability.

All linear metrics use the same paired samples in mD, so `SSE = n * RMSE**2` always holds. The manuscript mixes terminology for R² and contains values that do not obey that identity; these scripts recompute metrics consistently. The table's “best model” is selected by linear RMSE for descriptive reporting only. GA minimizes log-space error, so it need not win on linear RMSE or on either test well.

## Figure correspondence

| Manuscript | Generated files | Adaptation |
|---|---|---|
| Figure 1 | fig01_location_schematic | Conceptual regional setting and invented five-well structure map; not surveyed geography |
| Figure 2 | fig02_stratigraphy_schematic | Simplified conceptual succession; not the published stratigraphic chart |
| Figure 3 | fig03_generic_workflow | Training and prediction workflow |
| Figure 4 | fig04_study_workflow | FL/ANN/GA workflow with the requested 3/2 split |
| Figure 5 | fig05_ENC1_logs | Facies/depth, GR, lab/Timur k, NPHI/RHOB, SWT, DT, RT, crossplot |
| Figure 6, first caption | fig06a_ENC1_predictions | FL/ANN/GA permeability versus depth |
| Figure 6, duplicate caption | fig06b_ENC1_crossplots | Three predicted/laboratory crossplots and metrics |
| Figures 7–9 | fig07–fig09, ENC2 | Same three plot families |
| Figures 10–12 | fig10–fig12, ENC3 | Same three plot families |
| Additional figures 13–15 | ENC4 files | Blind-test log panel, predictions and crossplots |
| Additional figures 16–18 | ENC5 files | Blind-test log panel, predictions and crossplots |
| Tables 2–4 | table02–table04 | Calculated FL/ANN/GA metrics and ranks for ENC1–ENC3 |
| Additional test tables | table05–table06 | Metrics and ranks for ENC4–ENC5 |
| Table 5 | table_best_models | Best model per well, expanded to five wells |

The repeated Figure 6 caption is preserved as 06a/06b rather than silently renumbering later figures. Crossplot mineral lines are simple ideal density mixing references, not a digitization of the Schlumberger tool-calibrated chart. Yellow neutron–density shading is a visual separation cue, not proof of hydrocarbons. Axes are adjusted to show synthetic ranges rather than clipping to the manuscript's ranges. The study's original 1-train/2-test split is deliberately replaced with the user's requested 3-train/2-test split. Table 1's real-reservoir properties are not republished as synthetic results.

## Files and outputs

- `permeability/data.py`: synthetic generator and exact sampling alignment.
- `permeability/models.py`: FL, ANN, GA, whole-well validation and evaluation.
- `permeability/figures.py`: every figure and result table.
- `run_pipeline.py`: complete experiment.
- `predict_new_logs.py`: reuse saved models on a log CSV.
- `tests/test_pipeline.py`: reproducibility, units, metric identities, leakage and split tests.
- `outputs/data/`: logs and laboratory CSVs.
- `outputs/models/`: fitted models and preprocessing, serialized with joblib. Load only trusted model files.
- `outputs/results/`: predictions, paired samples, metrics, training OOF predictions, GA convergence and audit JSON.
- `outputs/figures/`: PNGs, PDFs, gallery and figure index.

The audit records the exact package versions, predictor list, fold well IDs, training sample count and GA weights. It is suitable for inspecting the split before using results.

## GitHub

Upload the project source or initialize your own repository here. `.gitignore` excludes the private manuscript, inspection files, environment, generated outputs and archives. No GitHub repository is created or published automatically. The included GitHub Actions workflow runs the tests and complete pipeline and attaches results as an artifact. The supplied ZIP includes a generated example run for convenience; generated outputs need not be committed.
