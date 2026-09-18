# -*- coding: utf-8 -*-
"""Synthetic permeability experiment -- open this file in Spyder and press Run.

This file is self-contained: it does not need the permeability package,
config.json, the manuscript, or any input data files.
Requires numpy, pandas, scipy, matplotlib and scikit-learn.
All generated data are synthetic. The regional map and stratigraphy are schematic.
"""

# %% SETTINGS -- edit these, then run the whole file
from pathlib import Path
import importlib.util
import importlib.metadata
import json
import sys

# Save this script on your PC first. Results are written beside it regardless
# of Spyder's working directory. You can replace this with Path(r"C:/my/results").
SCRIPT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = SCRIPT_FOLDER / "spyder_outputs"
GENERATE_FIGURES = True
SHOW_FIGURES = True  # Use Spyder's Inline backend to populate the Plots pane.

CONFIG = {'seed': 2026,
 'train_wells': ['ENC1', 'ENC2', 'ENC3'],
 'test_wells': ['ENC4', 'ENC5'],
 'log_samples_per_well': [1335, 1150, 929, 1100, 1000],
 'lab_samples_per_well': [100, 90, 80, 85, 75],
 'depth_interval_m': [0, 250],
 'features': ['GR_API', 'RHOB_g_cm3', 'NPHI_vv', 'DT_us_ft'],
 'fuzzy_rules': 9,
 'ann_hidden_layers': [16, 8],
 'ga_population': 80,
 'ga_generations': 160,
 'figure_dpi': 180}

# %% Dependency check (does not install anything or change your environment)
missing = [name for name in ("numpy", "pandas", "scipy", "matplotlib", "sklearn")
           if importlib.util.find_spec(name) is None]
if missing:
    raise ImportError(
        "Missing packages: " + ", ".join(missing) +
        "\nIn Spyder's IPython console run:\n"
        "%pip install numpy pandas scipy matplotlib scikit-learn\n"
        "Then restart the console/kernel and run this file again.\n"
        "Current Python: " + sys.executable
    )

# %% Synthetic well logs and laboratory samples
"""Generate depth-correlated logs and sparse, independent laboratory noise."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

WELLS = [f"ENC{i}" for i in range(1, 6)]
FACIES = ["Grainstone", "Grainstone/packstone", "Mudstone", "Packstone/wackestone"]
COLORS = ["#9756b5", "#428dcc", "#67b567", "#e3c957"]
FEATURES = ["GR_API", "RHOB_g_cm3", "NPHI_vv", "DT_us_ft"]


def validate_config(c):
    train, test = set(c["train_wells"]), set(c["test_wells"])
    if len(train) != 3 or len(test) != 2 or train & test or train | test != set(WELLS):
        raise ValueError("Use three distinct training wells and two distinct test wells from ENC1-ENC5.")
    if len(c["train_wells"]) != 3 or len(c["test_wells"]) != 2:
        raise ValueError("Duplicate well names are not allowed.")
    if c["features"] != FEATURES:
        raise ValueError("The manuscript experiment uses only GR, RHOB, NPHI and DT, in that order.")
    for key in ("log_samples_per_well", "lab_samples_per_well"):
        if len(c[key]) != 5 or any(not isinstance(x, int) or x < 20 for x in c[key]):
            raise ValueError(f"{key} must contain five integers >= 20.")
    if any(a > b for a, b in zip(c["lab_samples_per_well"], c["log_samples_per_well"])):
        raise ValueError("Laboratory counts cannot exceed log counts.")
    if c["depth_interval_m"][1] <= c["depth_interval_m"][0]:
        raise ValueError("Depth interval must increase.")
    if c["ga_population"] < 10 or c["ga_generations"] < 1 or c["fuzzy_rules"] < 2:
        raise ValueError("Invalid model configuration.")


def timur(phi, swirr):
    """k in mD; convert fractional porosity and irreducible saturation to percent."""
    return 0.136 * (100 * np.asarray(phi)) ** 4.4 / (100 * np.asarray(swirr)) ** 2


def generate(c):
    validate_config(c)
    logs, labs = [], []
    for i, well in enumerate(WELLS):
        rng = np.random.default_rng(c["seed"] + i * 101)
        n = c["log_samples_per_well"][i]
        z = np.linspace(*c["depth_interval_m"], n)
        relative = (z - z.min()) / np.ptp(z) * 250
        # Beds are generated before the physical properties. No model predictions
        # or desired metrics enter the generator.
        boundaries = np.r_[0, np.cumsum(rng.uniform(3, 15, 45)), 1000]
        beds = np.searchsorted(boundaries, relative, side="right") - 1
        categories = rng.choice(4, len(boundaries), p=[.33, .25, .20, .22])
        reservoir = (relative > 55 + 2 * i) & (relative < 135 + 3 * i)
        f = categories[beds]
        f[reservoir & (np.sin(relative / 10 + i / 2) > -.3)] = i % 2
        smooth = gaussian_filter1d(rng.normal(size=n), max(1, n / 150))
        smooth /= max(smooth.std(), 1e-6)
        phi = np.clip(np.take([.235, .19, .075, .13], f) + .018 * smooth
                      + .016 * np.sin(relative / 29 + i) - .003 * i, .035, .32)
        clay = np.clip(np.take([.035, .09, .40, .20], f) + .025 * smooth, .005, .6)
        dolomite = .10 + .18 * (np.sin(relative / 40 + i) > .6)
        matrix_density = 2.71 + .16 * dolomite
        swirr = np.clip(.07 + .55 * clay + .016 / phi, .08, .6)
        hydrocarbon_zone = np.exp(-((relative - 95 - i * 3) / 53) ** 4)
        swt = np.clip(1 - hydrocarbon_zone * (1 - swirr), swirr, 1)
        fluid_density = .82 * (1 - swt) + swt
        gr = np.clip(16 + 190 * clay + rng.normal(0, 3, n), 0, 150)
        rhob = matrix_density * (1 - phi) + fluid_density * phi + rng.normal(0, .013, n)
        nphi = np.clip(phi + .09 * clay - .02 * (1 - swt) + rng.normal(0, .009, n), -.02, .4)
        dt = 47.5 * (1 - phi) + 189 * phi + 10 * clay + rng.normal(0, 1.7, n)
        rt = np.clip(.11 / (phi ** 2 * swt ** 2) * np.exp(rng.normal(0, .13, n)), .1, 5000)
        # Permeability depends on porosity AND fabric, plus unresolved heterogeneity.
        logk = (-1.65 + 13.0 * phi + np.take([.45, .15, -.55, -.15], f)
                - .55 * clay + .12 * smooth + rng.normal(0, .12, n))
        truth = np.clip(10 ** logk, .002, 600)
        frame = pd.DataFrame({"well": well, "sample_id": np.arange(n), "depth_m": z,
            "split": "train" if well in c["train_wells"] else "test",
            "facies_code": f, "facies": np.take(FACIES, f), "GR_API": gr,
            "RHOB_g_cm3": rhob, "NPHI_vv": nphi, "DT_us_ft": dt,
            "RT_ohm_m": rt, "SWT_vv": swt, "SWIRR_vv": swirr,
            "PHI_vv": phi, "k_timur_mD": timur(phi, swirr)})
        ids = np.sort(rng.choice(n, c["lab_samples_per_well"][i], replace=False))
        lab = frame.iloc[ids][["well", "sample_id", "depth_m", "split", "facies"]].copy()
        lab["porosity_lab_vv"] = np.clip(phi[ids] + rng.normal(0, .006, len(ids)), .01, .4)
        lab["k_lab_mD"] = truth[ids] * 10 ** rng.normal(0, .09, len(ids))
        logs.append(frame)
        labs.append(lab)
    return pd.concat(logs, ignore_index=True), pd.concat(labs, ignore_index=True)


def align_laboratory(logs, labs):
    """Exact synthetic sampling keys avoid interpolation across lithologic beds."""
    data = labs.merge(logs.drop(columns=["depth_m", "split", "facies"]),
                      on=["well", "sample_id"], validate="one_to_one")
    if len(data) != len(labs):
        raise ValueError("Some laboratory measurements do not match a log sample.")
    return data

# %% FL, ANN, GA and blind evaluation
"""Train-only preprocessing, Gaussian Sugeno fuzzy model, MLP and real-coded GA."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FuzzyRegressor(RegressorMixin, BaseEstimator):
    """First-order Takagi-Sugeno rules with Gaussian product antecedents.

    K-means estimates centers on training data. Normalized Gaussian memberships
    weight per-rule linear consequents fitted jointly by ridge regression.
    Inputs arrive standardized; targets are log10(k/mD).
    """
    def __init__(self, n_rules=9, alpha=1.0, random_state=2026):
        self.n_rules = n_rules
        self.alpha = alpha
        self.random_state = random_state

    def _design(self, X):
        X = np.asarray(X)
        distance = ((X[:, None, :] - self.centers_[None, :, :]) / self.widths_) ** 2
        log_fire = -.5 * distance.sum(axis=2)
        log_fire -= log_fire.max(axis=1, keepdims=True)
        fire = np.exp(log_fire)
        fire /= fire.sum(axis=1, keepdims=True)
        augmented = np.column_stack([np.ones(len(X)), X])
        return (fire[:, :, None] * augmented[:, None, :]).reshape(len(X), -1)

    def fit(self, X, y):
        X = np.asarray(X)
        clusters = KMeans(n_clusters=self.n_rules, n_init=10, random_state=self.random_state).fit(X)
        self.centers_ = clusters.cluster_centers_
        self.widths_ = np.array([np.maximum(X[clusters.labels_ == j].std(axis=0), .55)
                                 for j in range(self.n_rules)])
        self.consequents_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(self._design(X), y)
        return self

    def predict(self, X):
        return self.consequents_.predict(self._design(X))


def base_models(c):
    return {
        "FL": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
            FuzzyRegressor(n_rules=c["fuzzy_rules"], random_state=c["seed"])),
        "ANN": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=tuple(c["ann_hidden_layers"]), activation="tanh",
                         solver="lbfgs", alpha=2.0, max_iter=3000, max_fun=50000,
                         random_state=c["seed"], tol=1e-5))}


def positive_prediction(model, X):
    # Fixed, broad numerical guardrails, not derived from test targets.
    return 10 ** np.clip(model.predict(X), -4, 5)


def genetic_weights(predictions, y, c):
    """Tournament selection, arithmetic crossover, Gaussian mutation, elitism.

    Search [0, 2]^2 for k_GA = w_FL*k_FL + w_ANN*k_ANN. Fitness is
    log10 RMSE of out-of-well predictions for the three TRAINING wells only.
    Nonnegative weights are not constrained to sum to one, as in the manuscript.
    """
    rng = np.random.default_rng(c["seed"])
    size = c["ga_population"]
    population = rng.uniform(0, 2, (size, 2))
    population[:3] = [[1, 0], [0, 1], [.5, .5]]
    target = np.log10(y)
    history = []
    def loss(p):
        estimates = np.maximum(predictions @ p.T, 1e-8)
        return np.sqrt(np.mean((np.log10(estimates) - target[:, None]) ** 2, axis=0))
    for generation in range(c["ga_generations"]):
        scores = loss(population)
        order = scores.argsort()
        history.append({"generation": generation, "oof_log10_rmse": float(scores[order[0]])})
        elite = population[order[:2]].copy()
        draws = rng.integers(0, size, (size - 2, 2, 3))
        winners = np.take_along_axis(draws, scores[draws].argmin(axis=2)[..., None], axis=2)[..., 0]
        parents = population[winners]
        mix = rng.random((size - 2, 1))
        child = mix * parents[:, 0] + (1 - mix) * parents[:, 1]
        child += (rng.random(child.shape) < .25) * rng.normal(0, .08, child.shape)
        population = np.vstack([elite, np.clip(child, 0, 2)])
    scores = loss(population)
    history.append({"generation": c["ga_generations"], "oof_log10_rmse": float(scores.min())})
    return population[scores.argmin()], pd.DataFrame(history)


def train(logs, labs, c):
    # Select BEFORE joining lab targets. Test labels never enter fitting.
    training = align_laboratory(logs[logs.well.isin(c["train_wells"])],
                                labs[labs.well.isin(c["train_wells"])]).reset_index(drop=True)
    X, y = training[c["features"]], training.k_lab_mD.to_numpy()
    oof = np.full((len(training), 2), np.nan)
    fold_records = []
    for fit_ids, val_ids in LeaveOneGroupOut().split(X, groups=training.well):
        models = base_models(c)
        for j, model in enumerate(models.values()):
            model.fit(X.iloc[fit_ids], np.log10(y[fit_ids]))
            oof[val_ids, j] = positive_prediction(model, X.iloc[val_ids])
        fold_records.append({"fit_wells": sorted(training.well.iloc[fit_ids].unique().tolist()),
                             "validation_well": training.well.iloc[val_ids[0]]})
    weights, history = genetic_weights(oof, y, c)
    models = base_models(c)
    for model in models.values():
        model.fit(X, np.log10(y))
    audit = {"training_wells": c["train_wells"], "blind_test_wells": c["test_wells"],
             "features": c["features"], "target": "log10(k_lab_mD)",
             "ga_objective": "training-well out-of-fold log10 RMSE",
             "ga_weights": dict(zip(["FL", "ANN"], weights.tolist())), "folds": fold_records,
             "training_lab_samples": len(training), "test_labels_used_for_fitting": False}
    out = training[["well", "sample_id", "depth_m", "k_lab_mD"]].copy()
    out[["k_FL_mD", "k_ANN_mD"]] = oof
    out["k_GA_mD"] = oof @ weights
    return {"models": models, "weights": weights, "features": c["features"]}, audit, out, history


def predict(bundle, logs):
    result = logs.copy()
    for name, model in bundle["models"].items():
        result[f"k_{name}_mD"] = positive_prediction(model, logs[bundle["features"]])
    result["k_GA_mD"] = result[["k_FL_mD", "k_ANN_mD"]].to_numpy() @ bundle["weights"]
    return result


def metrics(y, p):
    residual = np.asarray(p) - np.asarray(y)
    r = np.corrcoef(y, p)[0, 1] if np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {"n": len(y), "SSE_mD2": float(np.sum(residual ** 2)),
            "RMSE_mD": float(np.sqrt(np.mean(residual ** 2))),
            "R2": float(r2_score(y, p)), "Pearson_r2": float(r ** 2),
            "RMSE_log10": float(np.sqrt(np.mean((np.log10(p) - np.log10(y)) ** 2)))}


def evaluate(predictions, labs):
    paired = align_laboratory(predictions, labs)
    rows = []
    groups = [(well, group, group.split.iloc[0]) for well, group in paired.groupby("well")]
    groups += [(f"ALL_{split.upper()}", group, split) for split, group in paired.groupby("split")]
    for well, group, split in groups:
        for model in ["FL", "ANN", "GA", "timur"]:
            rows.append({"well": well, "split": split, "model": model,
                         **metrics(group.k_lab_mD, group[f"k_{model}_mD"])})
    return pd.DataFrame(rows), paired

# %% Manuscript-style figures
"""Original, manuscript-style figures, generated entirely from synthetic data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch

MODEL_COLORS = {"FL": "#30904b", "ANN": "#376cbe", "GA": "#d95151"}


class FigureWriter:
    def __init__(self, output, dpi):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.index = []
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                            "savefig.facecolor": "white", "axes.spines.top": False,
                            "axes.spines.right": False, "pdf.fonttype": 42})

    def save(self, fig, name, title, source):
        fig.suptitle(title, fontsize=14, fontweight="bold", y=.99)
        fig.text(.5, .008, "SYNTHETIC DATA / SCHEMATIC — not field measurements or original manuscript results",
                 ha="center", fontsize=8, color="#666666")
        for suffix in ("png", "pdf"):
            fig.savefig(self.path / f"{name}.{suffix}", dpi=self.dpi, bbox_inches="tight")
        if SHOW_FIGURES and "inline" in matplotlib.get_backend().lower():
            from IPython.display import display
            display(fig)
            plt.close(fig)
        elif SHOW_FIGURES and matplotlib.get_backend().lower() != "agg":
            plt.show(block=False)
        else:
            plt.close(fig)
        self.index.append({"file": f"{name}.png", "title": title, "manuscript_reference": source})


def context_figures(writer, c):
    train_label = ", ".join(c["train_wells"])
    test_label = ", ".join(c["test_wells"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=.35)
    ax = axes[0]
    ax.set_facecolor("#dceef5")
    # Local coordinates deliberately avoid implying surveyed coastline or well positions.
    ax.fill([-6, -6, -4.8, -4.2, -3.1, -2.2, -1.5, -1, -6],
            [-5, 5, 5, 3, 1.7, .8, -.8, -5, -5], color="#d5ceae")
    ax.text(-4.5, 1, "SE Brazil\nconceptual coast", ha="center")
    ax.add_patch(Rectangle((.1, -.9), 2.5, 3, fill=False, edgecolor="red", linestyle="--"))
    ax.text(1.35, 2.4, "Synthetic field", ha="center")
    ax.text(1.5, -3, "Atlantic Ocean", color="#397797")
    ax.set(xlim=(-6, 5), ylim=(-5, 5), title="A  Regional setting schematic",
           xlabel="Local schematic coordinate", ylabel="Local schematic coordinate")
    ax = axes[1]
    x, y = np.meshgrid(np.linspace(0, 4, 120), np.linspace(0, 3, 100))
    depth = 2420 - 100 * np.exp(-((x - 2) ** 2 / 2 + (y - 1.5) ** 2)) + 14 * x
    im = ax.contourf(x, y, depth, 16, cmap="viridis_r")
    contour = ax.contour(x, y, depth, 6, colors="white", linewidths=.5)
    ax.clabel(contour, fontsize=7)
    for well, xx, yy in zip(WELLS, [.7, 1.6, 2.7, 1.2, 3.3], [.7, 1.9, 1.3, 2.5, 2.4]):
        color = "#ffef75" if well in c["train_wells"] else "#ff7e71"
        ax.scatter(xx, yy, c=color, marker="^", s=70, edgecolors="black")
        ax.annotate(well, (xx, yy), xytext=(6, 5), textcoords="offset points", color="black",
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
    ax.set(title="B  Invented structure and five well positions", xlabel="Local x (km)", ylabel="Local y (km)")
    fig.colorbar(im, ax=ax, label="Synthetic structural depth (m)", shrink=.8)
    writer.save(fig, "fig01_location_schematic", "Five-well synthetic study layout", "Figure 1; schematic adaptation")

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    ax.set_axis_off()
    rows = [
        ["Upper Albian", "Macae Group", "Outeiro", "Marl / calcilutite", "Marine deepening"],
        ["Lower–Middle Albian", "Macae Group", "Quissama", "Grainstone / packstone", "Carbonate reservoir"],
        ["Aptian", "Underlying succession", "Evaporitic interval", "Salt / anhydrite", "Transitional setting"],
        ["Pre-Aptian", "Underlying succession", "Rift succession", "Mixed sediments", "Rift setting"]]
    table = ax.table(cellText=rows, colLabels=["Relative age", "Group / interval", "Unit", "Lithology", "Setting"],
                     colWidths=[.19, .22, .18, .22, .19], cellLoc="center", bbox=[.01, .25, .98, .60])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("#cfe3ef" if r == 0 else "#ece0ab" if r == 2 else "#f6f6f6")
    ax.text(.5, .14, "Simplified conceptual succession based on the supplied text; thicknesses and ages are not to scale.\n"
            "This is not a reproduction of the Winter et al. stratigraphic chart.", transform=ax.transAxes, ha="center")
    writer.save(fig, "fig02_stratigraphy_schematic", "Conceptual carbonate stratigraphy", "Figure 2; simplified schematic")

    flow(writer, "fig03_generic_workflow", "Machine learning workflow", "Figure 3", [
        ("Training data", .12, .65), ("Preprocessing\nand model fit", .40, .65),
        ("Frozen model", .72, .65), ("New well logs", .40, .25), ("Predictions\nand evaluation", .72, .25)],
        [(0, 1), (1, 2), (2, 4), (3, 4)])
    flow(writer, "fig04_study_workflow", "Three training wells and two blind test wells", "Figure 4; expanded split", [
        (f"{train_label}\nlogs + lab data", .11, .72), ("Leave one training\nwell out", .37, .72),
        ("FL + ANN\nout-of-fold predictions", .65, .72), ("GA learns two weights\nfrom training OOF only", .65, .40),
        (f"Refit FL + ANN\non {train_label}", .37, .40), (f"{test_label} logs\nfrozen models + weights", .37, .10),
        ("Blind predictions\nthen reveal test lab data", .76, .10)],
        [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (3, 6), (5, 6)])


def flow(writer, name, title, reference, nodes, edges):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=.87, bottom=.12, left=.07, right=.94)
    ax.set(xlim=(0, 1), ylim=(-.05, 1))
    ax.set_axis_off()
    half_width, half_height = .116, .072
    for label, x, y in nodes:
        ax.add_patch(Rectangle((x-half_width, y-half_height), 2*half_width, 2*half_height,
                              facecolor="#e7f0f5", edgecolor="#527589", clip_on=False))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    for a, b in edges:
        start, end = np.array(nodes[a][1:]), np.array(nodes[b][1:])
        direction = end-start
        scale = min(half_width / abs(direction[0]) if direction[0] else np.inf,
                    half_height / abs(direction[1]) if direction[1] else np.inf)
        offset = direction * scale
        ax.add_patch(FancyArrowPatch(start+offset, end-offset,
                     arrowstyle="->", mutation_scale=13, lw=1.5, color="#526475", shrinkA=3, shrinkB=3))
    writer.save(fig, name, title, reference)


def log_panel(writer, data, lab, well, number, reference):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 8, width_ratios=[.48, .85, 1.1, 1.15, .75, .85, .95, 3.0],
                          left=.045, right=.985, top=.84, bottom=.17, wspace=.42)
    axes = [fig.add_subplot(gs[0, j]) for j in range(8)]
    z = data.depth_m.to_numpy()
    for ax in axes[:7]:
        ax.set_ylim(z.max(), z.min())
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    axes[0].imshow(data.facies_code.to_numpy()[:, None], aspect="auto", cmap=ListedColormap(COLORS),
                    vmin=-.5, vmax=3.5, extent=[0, 1, z.max(), z.min()], interpolation="nearest")
    axes[0].set(xticks=[], xlabel="A / B\nFacies", ylabel="Relative depth (m)")
    for ax in axes[1:7]:
        ax.tick_params(labelleft=False)
    axes[1].plot(data.GR_API, z, color="green", lw=.6)
    axes[1].fill_betweenx(z, 0, data.GR_API, color="#f1d953", alpha=.6)
    axes[1].set(xlim=(0, 150), xlabel="C  GR\nAPI", xticks=[0, 75, 150])
    axes[2].semilogx(data.k_timur_mD, z, color="green", lw=.7, label="Timur")
    axes[2].scatter(lab.k_lab_mD, lab.depth_m, c="black", s=9, zorder=3, label="Lab")
    axes[2].set(xlim=(.001, 10000), xlabel="D  k\nmD", xticks=[.01, 1, 100, 10000])
    axes[2].legend(fontsize=7, loc="lower right")
    ax = axes[3]
    ax.plot(data.NPHI_vv, z, c="red", lw=.65)
    ax.set(xlim=(.4, -.15), xlabel="E  NPHI\nv/v", xticks=[.4, .1, -.15])
    twin = ax.twiny()
    twin.plot(data.RHOB_g_cm3, z, color="green", lw=.65)
    twin.set(xlim=(1.95, 2.95), xlabel="RHOB (g/cm³)", xticks=[1.95, 2.45, 2.95])
    twin.xaxis.set_label_position("bottom")
    twin.xaxis.tick_bottom()
    twin.tick_params(labelsize=7)
    # twiny resets the original axis ticks to the bottom; restore the neutron scale.
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    density_on_neutron_axis = .4 - (data.RHOB_g_cm3.to_numpy() - 1.95) * .55
    ax.fill_betweenx(z, data.NPHI_vv, density_on_neutron_axis,
                     where=data.NPHI_vv.to_numpy() < density_on_neutron_axis,
                     color="#f5d44e", alpha=.6)
    axes[4].plot(data.SWT_vv, z, c="#4399c6", lw=.7)
    axes[4].fill_betweenx(z, 0, data.SWT_vv, color="#9fdaee", alpha=.8)
    axes[4].set(xlim=(1, 0), xlabel="F  SWT\nv/v", xticks=[1, .5, 0])
    axes[5].plot(data.DT_us_ft, z, c="black", lw=.7)
    axes[5].set(xlim=(40, 130), xlabel="G  DT\nµs/ft", xticks=[40, 85, 130])
    axes[6].semilogx(data.RT_ohm_m, z, c="#3975c4", lw=.7)
    axes[6].set(xlim=(.1, 5000), xlabel="H  RT\nΩ m", xticks=[1, 100, 5000])
    ax = axes[7]
    ax.scatter(data.NPHI_vv, data.RHOB_g_cm3, c=data.facies_code, cmap=ListedColormap(COLORS),
               vmin=-.5, vmax=3.5, s=5, alpha=.45)
    phi = np.linspace(0, .4, 30)
    for density, label, color in [(2.65, "Quartz", "#c86159"), (2.71, "Calcite", "#388657"), (2.87, "Dolomite", "#6262b4")]:
        ax.plot(phi, density * (1 - phi) + phi, c=color, label=label, lw=1)
    ax.set(xlabel="NPHI (v/v)", ylabel="RHOB (g/cm³)", title="I  Neutron–density crossplot\nIdeal mixing reference lines",
           xlim=(-.02, .4), ylim=(2.95, 1.95))
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.legend(handles=[Patch(color=c, label=f) for f, c in zip(FACIES, COLORS)], loc="lower center",
               bbox_to_anchor=(.5, .055), ncol=4, frameon=False)
    role = "training" if data.split.iloc[0] == "train" else "blind test"
    writer.save(fig, f"fig{number}_{well}_logs", f"{well} — {role} well logs and synthetic laboratory data", reference)


def permeability_tracks(writer, data, lab, well, number, reference):
    fig, axes = plt.subplots(1, 3, figsize=(11, 8), sharey=True)
    fig.subplots_adjust(top=.89, bottom=.1, wspace=.2)
    lo = min(lab.k_lab_mD.min(), data[[f"k_{m}_mD" for m in MODEL_COLORS]].min().min())
    hi = max(lab.k_lab_mD.max(), data[[f"k_{m}_mD" for m in MODEL_COLORS]].max().max())
    for letter, (name, color), ax in zip("ABC", MODEL_COLORS.items(), axes):
        ax.semilogx(data[f"k_{name}_mD"], data.depth_m, c=color, lw=.8, label=name)
        ax.scatter(lab.k_lab_mD, lab.depth_m, color="black", s=10, label="Laboratory", zorder=3)
        ax.set(xlim=(10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))),
               xlabel="Permeability (mD)", title=f"{letter}  {name}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set(ylabel="Relative depth (m)", ylim=(data.depth_m.max(), data.depth_m.min()))
    role = "training fit" if data.split.iloc[0] == "train" else "blind test"
    writer.save(fig, f"fig{number}_{well}_predictions", f"{well} — permeability predictions ({role})", reference)


def crossplots(writer, paired, well, number, reference):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], top=.89, bottom=.1, hspace=.6, wspace=.2)
    for row, name in enumerate(MODEL_COLORS):
        ax, text = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        y, p = paired.k_lab_mD.to_numpy(), paired[f"k_{name}_mD"].to_numpy()
        slope, intercept = np.polyfit(y, p, 1)
        xx = np.linspace(0, max(y.max(), p.max()) * 1.05, 100)
        ax.scatter(y, p, c="black", s=10, alpha=.65, label="Paired samples")
        ax.plot(xx, slope * xx + intercept, color="#3578be", label="Descriptive linear fit")
        ax.plot(xx, xx, color="#999999", ls="--", lw=.8, label="1:1")
        ax.set(xlabel="Laboratory permeability (mD)", ylabel="Predicted k (mD)", title=f"{'ABC'[row]}  {name}")
        ax.grid(alpha=.2)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left")
        m = metrics(y, p)
        text.set_axis_off()
        text.text(0, .88, f"k = {slope:.3f} × kLAB {intercept:+.3f}\n\n"
                  f"n = {m['n']}\nSSE = {m['SSE_mD2']:.2f} mD²\n"
                  f"RMSE = {m['RMSE_mD']:.3f} mD\n"
                  f"Predictive R² = {m['R2']:.3f}\nPearson r² = {m['Pearson_r2']:.3f}\n"
                  f"log10 RMSE = {m['RMSE_log10']:.3f}", va="top", fontsize=9)
    role = "training fit" if paired.split.iloc[0] == "train" else "blind test"
    writer.save(fig, f"fig{number}_{well}_crossplots", f"{well} — predicted versus laboratory permeability ({role})", reference)


def result_tables(writer, scores):
    for i, well in enumerate(WELLS):
        data = scores[(scores.well == well) & (scores.model != "timur")].copy()
        data["SSE rank"] = data.SSE_mD2.rank(method="min").astype(int)
        data["r² rank"] = data.Pearson_r2.rank(ascending=False, method="min").astype(int)
        data["RMSE rank"] = data.RMSE_mD.rank(method="min").astype(int)
        table_plot(writer, data[["model", "SSE_mD2", "SSE rank", "Pearson_r2", "r² rank", "RMSE_mD", "RMSE rank"]],
                   f"table{i+2:02d}_{well}_metrics", f"{well} — synthetic permeability model comparison",
                   f"Table {i+2}" if i < 3 else "Additional test-well table")
    ml = scores[scores.well.isin(WELLS) & (scores.model != "timur")]
    best = ml.loc[ml.groupby("well").RMSE_mD.idxmin()]
    table_plot(writer, best[["well", "split", "model", "SSE_mD2", "Pearson_r2", "RMSE_mD"]],
               "table_best_models", "Best model per well by linear-space RMSE", "Table 5; expanded to five wells")


def table_plot(writer, data, name, title, source):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_axis_off()
    cells = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in data.to_numpy()]
    table = ax.table(cellText=cells, colLabels=data.columns, cellLoc="center", bbox=[0, .12, 1, .65])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d3d3d3")
        cell.set_facecolor("#dce9f0" if r == 0 else "#f5f5f5" if r % 2 else "white")
    writer.save(fig, name, title, source)


def make_figures(predictions, labs, paired, scores, output, c):
    writer = FigureWriter(output, c["figure_dpi"])
    context_figures(writer, c)
    numbers = [("05", "06a", "06b"), ("07", "08", "09"), ("10", "11", "12"),
               ("13", "14", "15"), ("16", "17", "18")]
    for i, well in enumerate(WELLS):
        data, lab = predictions[predictions.well == well], labs[labs.well == well]
        a, b, d = numbers[i]
        refs = ([f"Figure {a}", f"Figure {b}", f"Figure {d}"] if i < 3 else ["Additional blind-test well"] * 3)
        if i == 0:
            refs = ["Figure 5", "Figure 6 (first caption)", "Figure 6 (second caption; text calls it Figure 7)"]
        log_panel(writer, data, lab, well, a, refs[0])
        permeability_tracks(writer, data, lab, well, b, refs[1])
        crossplots(writer, paired[paired.well == well], well, d, refs[2])
    result_tables(writer, scores)
    pd.DataFrame(writer.index).to_csv(writer.path / "figure_index.csv", index=False)
    lines = ["# Synthetic figure gallery", "", "All results are synthetic. Context figures are schematic.", ""]
    for item in writer.index:
        lines += [f"## {item['title']}", "", f"Source counterpart: {item['manuscript_reference']}", "",
                  f"![{item['title']}]({item['file']})", ""]
    (writer.path / "GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    return writer.index

# %% Full experiment
def run(config, output, skip_figures=False):
    validate_config(config)
    output = Path(output)
    for name in ["data", "results"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    logs, labs = generate(config)
    logs.to_csv(output / "data/well_logs.csv", index=False)
    labs.to_csv(output / "data/laboratory.csv", index=False)
    print(f"Generated {len(logs)} log rows and {len(labs)} laboratory measurements.", flush=True)
    bundle, audit, oof, history = train(logs, labs, config)
    predictions = predict(bundle, logs)
    scores, paired = evaluate(predictions, labs)
    predictions.to_csv(output / "results/predictions.csv", index=False)
    paired.to_csv(output / "results/paired_laboratory_predictions.csv", index=False)
    scores.to_csv(output / "results/metrics.csv", index=False)
    oof.to_csv(output / "results/training_out_of_fold.csv", index=False)
    history.to_csv(output / "results/ga_history.csv", index=False)
    audit["versions"] = {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]}
    (output / "results/audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Models fitted. Blind test results:", flush=True)
    print(scores[scores.well == "ALL_TEST"].to_string(index=False), flush=True)
    if not skip_figures:
        index = make_figures(predictions, labs, paired, scores, output / "figures", config)
        print(f"Saved {len(index)} figure/table pairs (PNG + PDF).", flush=True)
    return logs, labs, predictions, scores, paired, bundle, audit, oof, history


# %% RUN -- resulting tables are available in Spyder's Variable Explorer
if __name__ == "__main__":
    # Closing these figures only affects plots created in this Python console.
    plt.close("all")
    plt.rcParams["figure.max_open_warning"] = 0
    (well_logs, laboratory_data, predictions, metric_table, paired_samples,
     fitted_models, training_audit, training_oof_predictions, ga_history) = run(
        CONFIG, OUTPUT_FOLDER, skip_figures=not GENERATE_FIGURES
    )
    print("\nFinished. Output folder:", OUTPUT_FOLDER.resolve())
    print("Inspect well_logs, laboratory_data, predictions and metric_table "
          "in Spyder's Variable Explorer.")
    if SHOW_FIGURES and matplotlib.get_backend().lower() == "agg":
        print("Figures were saved to disk. To display them in Spyder, select "
              "the Inline graphics backend and restart the console.")
