"""One command generates data, trains models, evaluates and renders every figure."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import joblib
from permeability.data import generate, validate_config
from permeability.models import train, predict, evaluate
from permeability.figures import make_figures


def run(config, output, skip_figures=False):
    validate_config(config)
    output = Path(output)
    for name in ["data", "models", "results"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    logs, labs = generate(config)
    logs.to_csv(output / "data/well_logs.csv", index=False)
    labs.to_csv(output / "data/laboratory.csv", index=False)
    print(f"Generated {len(logs)} log rows and {len(labs)} laboratory measurements.", flush=True)
    bundle, audit, oof, history = train(logs, labs, config)
    joblib.dump(bundle, output / "models/permeability_models.joblib")
    predictions = predict(bundle, logs)
    scores, paired = evaluate(predictions, labs)
    predictions.to_csv(output / "results/predictions.csv", index=False)
    paired.to_csv(output / "results/paired_laboratory_predictions.csv", index=False)
    scores.to_csv(output / "results/metrics.csv", index=False)
    oof.to_csv(output / "results/training_out_of_fold.csv", index=False)
    history.to_csv(output / "results/ga_history.csv", index=False)
    audit["versions"] = {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "joblib"]}
    (output / "results/audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Models fitted. Blind test results:", flush=True)
    print(scores[scores.well == "ALL_TEST"].to_string(index=False), flush=True)
    if not skip_figures:
        index = make_figures(predictions, labs, paired, scores, output / "figures", config)
        print(f"Saved {len(index)} figure/table pairs (PNG + PDF).", flush=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.skip_figures)
