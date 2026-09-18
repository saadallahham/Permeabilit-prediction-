"""Maintainer utility: bundle the tested implementation into one readable script."""
import json
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parent


def build():
    header = '''# -*- coding: utf-8 -*-
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

'''
    header += "CONFIG = " + pformat(json.loads((ROOT / "config.json").read_text()), sort_dicts=False) + "\n\n"
    header += '''# %% Dependency check (does not install anything or change your environment)
missing = [name for name in ("numpy", "pandas", "scipy", "matplotlib", "sklearn")
           if importlib.util.find_spec(name) is None]
if missing:
    raise ImportError(
        "Missing packages: " + ", ".join(missing) +
        "\\nIn Spyder's IPython console run:\\n"
        "%pip install numpy pandas scipy matplotlib scikit-learn\\n"
        "Then restart the console/kernel and run this file again.\\n"
        "Current Python: " + sys.executable
    )

'''
    sections = [header]
    for filename, title in [("data.py", "Synthetic well logs and laboratory samples"),
                            ("models.py", "FL, ANN, GA and blind evaluation"),
                            ("figures.py", "Manuscript-style figures")]:
        code = (ROOT / "permeability" / filename).read_text(encoding="utf-8")
        code = "\n".join(line for line in code.splitlines()
                         if not line.startswith("from .") and line != 'matplotlib.use("Agg")')
        if filename == "figures.py":
            code = code.replace("        plt.close(fig)", '''        if SHOW_FIGURES and "inline" in matplotlib.get_backend().lower():
            from IPython.display import display
            display(fig)
            plt.close(fig)
        elif SHOW_FIGURES and matplotlib.get_backend().lower() != "agg":
            plt.show(block=False)
        else:
            plt.close(fig)''')
        sections.append("# %% " + title + "\n" + code + "\n\n")
    runner = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")
    runner = runner[runner.index("def run("):runner.index('if __name__ == "__main__":')]
    runner = runner.replace('["data", "models", "results"]', '["data", "results"]')
    runner = runner.replace('    joblib.dump(bundle, output / "models/permeability_models.joblib")\n', '')
    runner = runner.replace(', "joblib"]', ']')
    runner = runner.replace('    return scores', '''    return logs, labs, predictions, scores, paired, bundle, audit, oof, history''')
    sections.append("# %% Full experiment\n" + runner)
    sections.append('''# %% RUN -- resulting tables are available in Spyder's Variable Explorer
if __name__ == "__main__":
    # Closing these figures only affects plots created in this Python console.
    plt.close("all")
    plt.rcParams["figure.max_open_warning"] = 0
    (well_logs, laboratory_data, predictions, metric_table, paired_samples,
     fitted_models, training_audit, training_oof_predictions, ga_history) = run(
        CONFIG, OUTPUT_FOLDER, skip_figures=not GENERATE_FIGURES
    )
    print("\\nFinished. Output folder:", OUTPUT_FOLDER.resolve())
    print("Inspect well_logs, laboratory_data, predictions and metric_table "
          "in Spyder's Variable Explorer.")
    if SHOW_FIGURES and matplotlib.get_backend().lower() == "agg":
        print("Figures were saved to disk. To display them in Spyder, select "
              "the Inline graphics backend and restart the console.")
''')
    target = ROOT / "spyder_permeability.py"
    target.write_text("".join(sections), encoding="utf-8")
    print(f"Created {target.name}")


if __name__ == "__main__":
    build()
