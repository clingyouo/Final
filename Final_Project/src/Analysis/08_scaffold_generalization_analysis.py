import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

FLAGGED = [6879, 6791, 6792, 6794, 6793, 6790, 6798]

SOURCE = Path(
    "results/analysis/representation_scaffold/"
    "scaffold_molecule_predictions.csv"
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))

pred = pd.read_csv(SOURCE)

required = [
    "row_index", "split", "seed",
    "rf_true", "rf_pred",
    "xgboost_true", "xgboost_pred",
    "chemprop_true", "chemprop_pred",
    "Murcko_Scaffold"
]

assert all(c in pred.columns for c in required)

assert clean["row_index"].is_unique
assert not pred.duplicated(
    ["split", "seed", "row_index"]
).any()

# Merge source and year information
meta = clean[
    [
        "row_index",
        "Source",
        "Year",
        "PAMPA_clipped",
        "Murcko_Scaffold"
    ]
].copy()

pred = pred.rename(
    columns={"Murcko_Scaffold": "prediction_scaffold"}
)

# The prediction file already contains PAMPA_clipped.
# Verify it against clean.csv before removing the duplicate column.
if "PAMPA_clipped" in pred.columns:

    target_check = pred[
        ["row_index", "PAMPA_clipped"]
    ].merge(
        meta[["row_index", "PAMPA_clipped"]],
        on="row_index",
        how="left",
        suffixes=("_prediction", "_clean"),
        validate="many_to_one"
    )

    assert np.allclose(
        target_check["PAMPA_clipped_prediction"],
        target_check["PAMPA_clipped_clean"],
        atol=1e-8
    ), "Prediction targets do not match clean.csv"

    pred = pred.drop(columns=["PAMPA_clipped"])

df = pred.merge(
    meta,
    on="row_index",
    how="left",
    validate="many_to_one"
)

assert df["Source"].notna().all()
assert df["Murcko_Scaffold"].notna().all()

assert (
    df["prediction_scaffold"].astype(str).to_numpy()
    == df["Murcko_Scaffold"].astype(str).to_numpy()
).all()

# Verify all model targets
for model in MODELS:
    assert np.allclose(
        df[f"{model}_true"],
        df["PAMPA_clipped"],
        atol=1e-8
    )

# Recalculate errors
for model in MODELS:
    df[f"{model}_residual"] = (
        df[f"{model}_pred"] - df["PAMPA_clipped"]
    )

    df[f"{model}_abs_error"] = (
        df[f"{model}_residual"].abs()
    )

# Full dataset scaffold sizes
scaffold_counts = (
    clean.groupby("Murcko_Scaffold")
    .size()
    .to_dict()
)

flagged_scaffolds = set(
    clean.loc[
        clean["row_index"].isin(FLAGGED),
        "Murcko_Scaffold"
    ]
)

df["is_flagged"] = df["row_index"].isin(FLAGGED)

# --------------------------------------------------
# 2. Scaffold size classification
# --------------------------------------------------

def size_bin(n):
    if n <= 2:
        return "1-2"
    elif n <= 9:
        return "3-9"
    elif n <= 49:
        return "10-49"
    else:
        return "50+"


# --------------------------------------------------
# 3. Scaffold-level performance
# --------------------------------------------------

scaffold_records = []

for (split, seed, scaffold), group in df.groupby(
    ["split", "seed", "Murcko_Scaffold"]
):

    total_n = scaffold_counts[scaffold]

    row = {
        "split": split,
        "seed": seed,
        "scaffold": scaffold,
        "n_test": len(group),
        "scaffold_total_n": total_n,
        "scaffold_size_bin": size_bin(total_n),
        "n_sources": group["Source"].nunique(),
        "sources": ";".join(
            sorted(group["Source"].unique())
        ),
        "target_mean": group["PAMPA_clipped"].mean(),
        "target_std": group["PAMPA_clipped"].std(),
        "is_flagged_scaffold": (
            scaffold in flagged_scaffolds
        )
    }

    for model in MODELS:

        residual = group[
            f"{model}_residual"
        ].to_numpy()

        errors = np.abs(residual)

        row[f"MAE_{model}"] = errors.mean()

        row[f"RMSE_{model}"] = np.sqrt(
            np.mean(residual ** 2)
        )

        row[f"bias_{model}"] = residual.mean()

    maes = {
        model: row[f"MAE_{model}"]
        for model in MODELS
    }

    best_mae = min(maes.values())

    winners = [
        model for model in MODELS
        if np.isclose(
            maes[model],
            best_mae,
            rtol=1e-10,
            atol=1e-12
        )
    ]

    row["best_model"] = ";".join(winners)
    row["n_winning_models"] = len(winners)

    scaffold_records.append(row)

scaffold_df = pd.DataFrame(scaffold_records)

scaffold_df.to_csv(
    OUT / "scaffold_generalization_metrics.csv",
    index=False
)

# --------------------------------------------------
# 4. Macro and micro performance
# --------------------------------------------------

summary_records = []

for (split, seed), group in df.groupby(
    ["split", "seed"]
):

    assert len(group) == 690
    assert group["row_index"].nunique() == 690

    scaffold_subset = scaffold_df[
        (scaffold_df["split"] == split) &
        (scaffold_df["seed"] == seed)
    ]

    n_scaffolds = len(scaffold_subset)

    for model in MODELS:

        scaffold_mae = scaffold_subset[
            f"MAE_{model}"
        ]

        scaffold_rmse = scaffold_subset[
            f"RMSE_{model}"
        ]

        wins = (
            scaffold_subset["best_model"]
            .str.split(";")
            .apply(
                lambda x: (
                    1 / len(x) if model in x else 0
                )
            )
        )

        residual = group[
            f"{model}_residual"
        ].to_numpy()

        summary_records.append({
            "split": split,
            "seed": seed,
            "model": model,
            "n_molecules": len(group),
            "n_scaffolds": n_scaffolds,

            "micro_MAE": np.abs(residual).mean(),

            "micro_RMSE": np.sqrt(
                np.mean(residual ** 2)
            ),

            "macro_MAE": scaffold_mae.mean(),

            "macro_RMSE": scaffold_rmse.mean(),

            "median_scaffold_MAE": scaffold_mae.median(),

            "scaffold_win_fraction": wins.mean(),

            "scaffold_wins": wins.sum()
        })

summary_df = pd.DataFrame(summary_records)

summary_df.to_csv(
    OUT / "scaffold_generalization_summary.csv",
    index=False
)

# --------------------------------------------------
# 5. Performance by scaffold size
# --------------------------------------------------

size_records = []

for keys, group in scaffold_df.groupby(
    ["split", "seed", "scaffold_size_bin"],
    observed=True
):

    split, seed, size_group = keys

    for model in MODELS:

        n_molecules = group["n_test"].sum()

        size_records.append({
            "split": split,
            "seed": seed,
            "scaffold_size_bin": size_group,
            "model": model,

            "n_scaffolds": len(group),

            "n_molecules": n_molecules,

            "macro_MAE": group[
                f"MAE_{model}"
            ].mean(),

            "micro_MAE": np.average(
                group[f"MAE_{model}"],
                weights=group["n_test"]
            ),

            "macro_RMSE": group[
                f"RMSE_{model}"
            ].mean(),

            "micro_RMSE": np.sqrt(
                np.average(
                    group[f"RMSE_{model}"] ** 2,
                    weights=group["n_test"]
                )
            )
        })

size_df = pd.DataFrame(size_records)

size_df.to_csv(
    OUT / "scaffold_size_performance.csv",
    index=False
)

# --------------------------------------------------
# 6. Scaffold size and prediction error correlation
# --------------------------------------------------

correlation_records = []

for (split, seed), group in scaffold_df.groupby(
    ["split", "seed"]
):

    for model in MODELS:

        for size_column, label in [
            ("scaffold_total_n", "full_scaffold_size"),
            ("n_test", "test_scaffold_size")
        ]:

            if group[size_column].nunique() < 2:
                continue

            rho, p = spearmanr(
                np.log10(group[size_column]),
                group[f"MAE_{model}"]
            )

            correlation_records.append({
                "split": split,
                "seed": seed,
                "model": model,
                "size_definition": label,
                "n_scaffolds": len(group),
                "spearman_rho": rho,
                "p_value": p
            })

correlation_df = pd.DataFrame(
    correlation_records
)

correlation_df.to_csv(
    OUT / "scaffold_size_error_correlations.csv",
    index=False
)

# --------------------------------------------------
# 7. Identify difficult scaffolds
# --------------------------------------------------

hard_records = []

for (split, seed), group in scaffold_df.groupby(
    ["split", "seed"]
):

    hardest = group.nlargest(
        20,
        "MAE_chemprop"
    )

    for _, row in hardest.iterrows():

        hard_records.append({
            "split": split,
            "seed": seed,
            "scaffold": row["scaffold"],
            "n_test": row["n_test"],
            "scaffold_total_n": row["scaffold_total_n"],
            "scaffold_size_bin": row["scaffold_size_bin"],
            "n_sources": row["n_sources"],
            "sources": row["sources"],
            "is_flagged_scaffold": row["is_flagged_scaffold"],
            "MAE_rf": row["MAE_rf"],
            "MAE_xgboost": row["MAE_xgboost"],
            "MAE_chemprop": row["MAE_chemprop"],
            "RMSE_chemprop": row["RMSE_chemprop"],
            "bias_chemprop": row["bias_chemprop"]
        })

hard_df = pd.DataFrame(hard_records)

hard_df.to_csv(
    OUT / "hard_scaffolds_top20.csv",
    index=False
)

# --------------------------------------------------
# 8. Flagged scaffold vs other scaffolds
# --------------------------------------------------

flagged_records = []

for (split, seed), group in scaffold_df.groupby(
    ["split", "seed"]
):

    for flag, subset in group.groupby(
        "is_flagged_scaffold"
    ):

        for model in MODELS:

            flagged_records.append({
                "split": split,
                "seed": seed,
                "group": (
                    "flagged_scaffold"
                    if flag else "other_scaffolds"
                ),
                "model": model,
                "n_scaffolds": len(subset),
                "n_molecules": subset["n_test"].sum(),
                "macro_MAE": subset[
                    f"MAE_{model}"
                ].mean(),
                "micro_MAE": np.average(
                    subset[f"MAE_{model}"],
                    weights=subset["n_test"]
                )
            })

flagged_df = pd.DataFrame(flagged_records)

flagged_df.to_csv(
    OUT / "flagged_vs_other_scaffolds.csv",
    index=False
)

# --------------------------------------------------
# 9. Across-seed descriptive summary
# --------------------------------------------------

across_seed = (
    summary_df.groupby(
        ["split", "model"]
    )[
        [
            "micro_MAE",
            "macro_MAE",
            "micro_RMSE",
            "macro_RMSE",
            "scaffold_win_fraction"
        ]
    ]
    .agg(["mean", "std"])
)

across_seed.columns = [
    f"{metric}_{stat}"
    for metric, stat in across_seed.columns
]

across_seed = across_seed.reset_index()

across_seed.to_csv(
    OUT / "scaffold_across_seed_summary.csv",
    index=False
)

# --------------------------------------------------
# 10. Print key results
# --------------------------------------------------

print("\n" + "=" * 75)
print("SCAFFOLD GENERALIZATION ANALYSIS COMPLETED")
print("=" * 75)

print("\n===== Macro vs Micro Performance =====")

print(
    summary_df[
        [
            "split", "seed", "model",
            "micro_MAE", "macro_MAE",
            "scaffold_win_fraction"
        ]
    ].round(4).to_string(index=False)
)

print("\n===== Across-seed Summary =====")

print(
    across_seed.round(4).to_string(index=False)
)

print("\n===== Performance by Scaffold Size =====")

print(
    size_df[
        [
            "split", "seed", "scaffold_size_bin",
            "model", "n_scaffolds",
            "n_molecules", "macro_MAE"
        ]
    ].round(4).to_string(index=False)
)

print("\n===== Scaffold Size vs Error =====")

print(
    correlation_df.round(4).to_string(index=False)
)

print("\n===== Flagged Scaffold Comparison =====")

print(
    flagged_df.round(4).to_string(index=False)
)

print("\n===== Top 20 Difficult Scaffolds =====")

print(
    hard_df[
        [
            "split", "seed", "n_test",
            "scaffold_size_bin",
            "MAE_rf", "MAE_xgboost",
            "MAE_chemprop"
        ]
    ].head(30).round(4).to_string(index=False)
)

print("\n" + "=" * 75)
print("ALL ANALYSES SAVED")
print("=" * 75)

print("Output:", OUT)
