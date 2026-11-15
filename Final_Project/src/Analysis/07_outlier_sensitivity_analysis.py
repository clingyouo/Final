import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

# Seven previously identified molecules
FLAGGED = [6879, 6791, 6792, 6794, 6793, 6790, 6798]

MODELS = ["rf", "xgboost", "chemprop"]

FINGERPRINTS = {
    "achiral": ["max_sim_achiral", "top5_sim_achiral"],
    "chiral": ["max_sim_chiral", "top5_sim_chiral"]
}

SOURCE = OUT / "cross_seed_similarity_molecule_level.csv"

df = pd.read_csv(SOURCE)

required = [
    "split", "seed", "row_index", "PAMPA_clipped",
    "pred_rf", "pred_xgboost", "pred_chemprop",
    "max_sim_achiral", "top5_sim_achiral",
    "max_sim_chiral", "top5_sim_chiral"
]

assert all(c in df.columns for c in required)

assert not df.duplicated(
    ["split", "seed", "row_index"]
).any()

# Recalculate absolute errors from predictions
for model in MODELS:
    df[f"abs_error_{model}"] = (
        df[f"pred_{model}"] - df["PAMPA_clipped"]
    ).abs()

df["is_flagged"] = df["row_index"].isin(FLAGGED)

# Three analysis groups
groups = {
    "all": df,
    "without_flagged": df[~df["is_flagged"]],
    "flagged_only": df[df["is_flagged"]]
}

metric_records = []
correlation_records = []
bin_records = []
advantage_records = []
membership_records = []

# Similarity bins
BIN_EDGES = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]

BIN_LABELS = [
    "0-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0"
]

for (split, seed), subset in df.groupby(["split", "seed"]):

    print(f"\nProcessing {split} | Seed {seed}")

    flagged_subset = subset[subset["is_flagged"]]

    for _, row in flagged_subset.iterrows():
        membership_records.append({
            "split": split,
            "seed": seed,
            "row_index": row["row_index"],
            "Source": row["Source"],
            "PAMPA_clipped": row["PAMPA_clipped"],
            "present_in_evaluation": True
        })

    for group_name, group in groups.items():

        selected = group[
            (group["split"] == split) &
            (group["seed"] == seed)
        ].copy()

        if selected.empty:
            continue

        # 1. Model performance
        for model in MODELS:

            errors = selected[f"abs_error_{model}"].to_numpy()

            residuals = (
                selected[f"pred_{model}"] -
                selected["PAMPA_clipped"]
            ).to_numpy()

            metric_records.append({
                "split": split,
                "seed": seed,
                "group": group_name,
                "model": model,
                "n": len(selected),
                "MAE": mean_absolute_error(
                    selected["PAMPA_clipped"],
                    selected[f"pred_{model}"]
                ),
                "RMSE": np.sqrt(np.mean(errors ** 2)),
                "bias": residuals.mean()
            })

        if group_name == "flagged_only":
            continue

        # 2. Similarity-error correlations
        for fingerprint, columns in FINGERPRINTS.items():

            for sim_col in columns:

                for model in MODELS:

                    rho, p = spearmanr(
                        selected[sim_col],
                        selected[f"abs_error_{model}"]
                    )

                    correlation_records.append({
                        "split": split,
                        "seed": seed,
                        "group": group_name,
                        "fingerprint": fingerprint,
                        "similarity_type": sim_col,
                        "model": model,
                        "n": len(selected),
                        "spearman_rho": rho,
                        "p_value": p
                    })

        # 3. Error by similarity bins
        for fingerprint, columns in FINGERPRINTS.items():

            for sim_col in columns:

                selected["similarity_bin"] = pd.cut(
                    selected[sim_col],
                    bins=BIN_EDGES,
                    labels=BIN_LABELS,
                    right=False,
                    include_lowest=True
                )

                for bin_name, bin_df in selected.groupby(
                    "similarity_bin",
                    observed=True
                ):

                    for model in MODELS:

                        errors = bin_df[
                            f"abs_error_{model}"
                        ]

                        bin_records.append({
                            "split": split,
                            "seed": seed,
                            "group": group_name,
                            "fingerprint": fingerprint,
                            "similarity_type": sim_col,
                            "similarity_bin": str(bin_name),
                            "model": model,
                            "n": len(bin_df),
                            "MAE": errors.mean(),
                            "median_error": errors.median()
                        })

        # 4. Chemprop advantage over baseline models
        for model in ["rf", "xgboost"]:

            advantage_records.append({
                "split": split,
                "seed": seed,
                "group": group_name,
                "baseline": model,
                "n": len(selected),
                "baseline_MAE": selected[
                    f"abs_error_{model}"
                ].mean(),
                "chemprop_MAE": selected[
                    "abs_error_chemprop"
                ].mean(),
                "MAE_improvement": (
                    selected[f"abs_error_{model}"].mean()
                    - selected["abs_error_chemprop"].mean()
                )
            })

# Save all results
metric_df = pd.DataFrame(metric_records)
correlation_df = pd.DataFrame(correlation_records)
bin_df = pd.DataFrame(bin_records)
advantage_df = pd.DataFrame(advantage_records)
membership_df = pd.DataFrame(membership_records)

metric_df.to_csv(
    OUT / "outlier_sensitivity_metrics.csv",
    index=False
)

correlation_df.to_csv(
    OUT / "outlier_sensitivity_correlations.csv",
    index=False
)

bin_df.to_csv(
    OUT / "outlier_sensitivity_bins.csv",
    index=False
)

advantage_df.to_csv(
    OUT / "outlier_sensitivity_advantage.csv",
    index=False
)

membership_df.to_csv(
    OUT / "outlier_sensitivity_membership.csv",
    index=False
)

# Summary: Chemprop improvement before and after exclusion
print("\n" + "=" * 70)
print("MODEL PERFORMANCE: ALL VS WITHOUT FLAGGED MOLECULES")
print("=" * 70)

for split in ["random", "scaffold"]:

    for model in MODELS:

        result = metric_df[
            (metric_df["split"] == split) &
            (metric_df["model"] == model) &
            (metric_df["group"].isin(
                ["all", "without_flagged"]
            ))
        ]

        print(f"\n{split.upper()} | {model.upper()}")

        print(
            result[
                ["seed", "group", "n", "MAE", "RMSE"]
            ].to_string(index=False)
        )

print("\n" + "=" * 70)
print("CHEMPROP ADVANTAGE")
print("=" * 70)

print(
    advantage_df[
        ["split", "seed", "group", "baseline",
         "n", "MAE_improvement"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("FLAGGED MOLECULE DISTRIBUTION")
print("=" * 70)

print(
    df.groupby(
        ["split", "seed"]
    )["is_flagged"].sum().to_string()
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput directory:", OUT)
