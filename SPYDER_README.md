# Run the permeability experiment in Spyder

Open **spyder_permeability.py** in Spyder and click **Run file**. It is a standalone script: you can copy just this Python file to another folder or PC. It generates its own data and does not need the previous project files or manuscript.

## First run

1. Save the script in a writable folder on your PC, then open it in Spyder.
2. If dependencies are missing, run this in Spyder's **IPython Console**, then restart the kernel:

   ```python
   %pip install numpy pandas scipy matplotlib scikit-learn
   ```

3. For figures in the Plots pane, select **Preferences → IPython console → Graphics → Graphics backend → Inline**, then restart the console. See the [official Spyder documentation](https://docs.spyder-ide.org/current/panes/ipythonconsole.html).
4. Click **Run file** (normally F5). Run the entire file initially; later code cells depend on the earlier definitions.

No terminal arguments, GitHub account, or input files are required. Package installation must target the Python environment used by Spyder; the dependency error prints the current interpreter path.

## Editable settings

At the top of the script:

- `OUTPUT_FOLDER`: defaults to a new `spyder_outputs` folder beside the script, independently of Spyder's working directory. Matching output files are replaced on reruns.
- `GENERATE_FIGURES`: save all 19 figures and 6 tables as PNG and PDF; set to `False` for data/model runs only.
- `SHOW_FIGURES`: display figures in Spyder as well as saving them. Set to `False` for disk output only. With a GUI backend instead of Inline, figures open in separate windows.
- `CONFIG`: random seed, sampling counts, depth interval, model settings, and training/test well names.

The default split is **ENC1, ENC2, ENC3 for training**, **ENC4, ENC5 for blind testing**. All values are synthetic. Figures follow the manuscript's plot types; the location map and stratigraphy are schematic, and the manuscript's numerical results are not reproduced.

## Explore your results

The following variables remain available after execution:

| Variable | Contents |
|---|---|
| `well_logs` | 5,514 synthetic well-log rows |
| `laboratory_data` | 430 synthetic laboratory samples |
| `predictions` | Logs plus FL, ANN and GA permeability predictions |
| `metric_table` | Per-well and pooled train/test statistics |
| `paired_samples` | Lab samples matched to predicted permeability |
| `training_oof_predictions` | Held-out training-well predictions used to fit GA weights |
| `ga_history` | GA fitness by generation |
| `training_audit` | Split, predictors, weights and package versions |
| `fitted_models` | In-memory fitted models, preprocessing and GA weights |

Double-click data tables in the Variable Explorer. The CSV files and audit JSON are saved under `spyder_outputs/data` and `spyder_outputs/results`; figures and the gallery are under `spyder_outputs/figures`.

To predict from the same fitted models while the console remains open:

```python
my_logs = pd.read_csv(r"C:/my_data/new_logs.csv")
my_predictions = predict(fitted_models, my_logs)
my_predictions.to_csv(OUTPUT_FOLDER / "new_predictions.csv", index=False)
```

The new CSV must contain `GR_API`, `RHOB_g_cm3`, `NPHI_vv`, and `DT_us_ft` with the units stated in those names. The standalone version keeps fitted models in memory; it does not serialize script-local classes. Rerun the script after restarting the console. Use the original modular project for portable saved-model files.

The standalone file preserves training-only preprocessing, whole-well validation and GA calibration. Test labels are used only to calculate metrics and make plots. Predictive R² and squared Pearson correlation are reported separately. No performance ranking is forced.
