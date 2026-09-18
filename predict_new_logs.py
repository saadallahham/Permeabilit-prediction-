"""Apply saved models to a CSV containing the four required well-log columns."""
import argparse
import joblib
import pandas as pd
from permeability.models import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", help="Input CSV")
    parser.add_argument("--model", default="outputs/models/permeability_models.joblib")
    parser.add_argument("--output", default="new_predictions.csv")
    args = parser.parse_args()
    # Load only a model file you trust; joblib uses pickle serialization.
    bundle = joblib.load(args.model)
    logs = pd.read_csv(args.logs)
    missing = set(bundle["features"]) - set(logs)
    if missing:
        parser.error(f"Missing required log columns: {sorted(missing)}")
    predict(bundle, logs).to_csv(args.output, index=False)
