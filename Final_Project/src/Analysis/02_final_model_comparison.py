import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(".")
OUT = ROOT / "results/analysis/final_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

clean = pd.read_csv("data/processed/clean.csv")

results = []
checks = []

for split_type in SPLITS:
    for seed in SEEDS:

        print(f"\n===== {split_type.upper()} | Seed {seed} =====")

        expected = pd.read_csv(
            f"splits/{split_type}/seed_{seed}/test.csv"
        )["row_index"].to_numpy(dtype=int)

        predictions = {}

        for model in MODELS:

            path = (
                ROOT / "results/models" / model /
                split_type / f"seed_{seed}" /
                "predictions_test.csv"
            )

            assert path.exists(), f"Missing: {path}"

            df = pd.read_csv(path)

            assert not df["row_index"].duplicated().any()

            df = df.sort_values("row_index").reset_index(drop=True)

            idx = df["row_index"].to_numpy(dtype=int)

            assert np.array_equal(idx, np.sort(expected)), (
                f"Test index mismatch: {model}, {split_type}, {seed}"
            )

            y_true = df["y_true"].to_numpy(dtype=float)
            y_pred = df["y_pred"].to_numpy(dtype=float)

            reference = clean.loc[idx, "PAMPA_clipped"].to_numpy()

            assert np.allclose(y_true, reference, atol=1e-7)
            assert np.isfinite(y_pred).all()

            predictions[model] = (idx, y_true, y_pred)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            pearson = np.corrcoef(y_true, y_pred)[0, 1]

            results.append({
                "split": split_type,
                "seed": seed,
                "model": model,
                "n_test": len(y_true),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Pearson_R": pearson
            })

            print(
                f"{model:10s} "
                f"MAE={mae:.4f} "
                f"RMSE={rmse:.4f} "
                f"R2={r2:.4f} "
                f"Pearson={pearson:.4f}"
            )

            checks.append({
                "split": split_type,
                "seed": seed,
                "model": model,
                "n_test": len(idx),
                "index_match": True,
                "target_match": True,
                "finite_predictions": True,
                "status": "PASS"
            })

        # Verify paired evaluation across all models
        for model in MODELS[1:]:
            assert np.array_equal(
                predictions["rf"][0],
                predictions[model][0]
            )
            assert np.allclose(
                predictions["rf"][1],
                predictions[model][1]
            )

# Save detailed metrics
metrics = pd.DataFrame(results)
metrics.to_csv(
    OUT / "model_metrics_recalculated.csv",
    index=False
)

# Mean and standard deviation across three seeds
summary = (
    metrics.groupby(["split", "model"])
    [["MAE", "RMSE", "R2", "Pearson_R"]]
    .agg(["mean", "std"])
)

summary.to_csv(OUT / "model_metrics_summary.csv")

# Calculate Chemprop improvement
improvements = []

for split in SPLITS:

    subset = metrics[metrics["split"] == split]

    means = subset.groupby("model")["MAE"].mean()

    for baseline in ["rf", "xgboost"]:

        improvement = means[baseline] - means["chemprop"]

        improvements.append({
            "split": split,
            "baseline": baseline,
            "chemprop_MAE": means["chemprop"],
            "baseline_MAE": means[baseline],
            "MAE_improvement": improvement,
            "relative_improvement_percent":
                improvement / means[baseline] * 100
        })

pd.DataFrame(improvements).to_csv(
    OUT / "chemprop_mae_improvement.csv",
    index=False
)

pd.DataFrame(checks).to_csv(
    OUT / "prediction_integrity_check.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON COMPLETED")
print("=" * 65)

print("\nMean metrics across seeds:")
print(summary.round(4).to_string())

print("\nChemprop improvement:")
print(pd.DataFrame(improvements).round(4).to_string(index=False))

print("\nIntegrity checks:")
print(pd.DataFrame(checks)["status"].value_counts())

print("\nOutput:", OUT)
