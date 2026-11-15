import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 3000
RANDOM_SEED = 20261105

rng = np.random.default_rng(RANDOM_SEED)

MODELS = ["rf", "xgboost", "chemprop"]
BASELINES = ["rf", "xgboost"]

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))

pred = pd.read_csv(
    "results/analysis/representation_scaffold/"
    "scaffold_molecule_predictions.csv"
)

assert len(pred) == 4140
assert set(MODELS) == set(
    ["rf", "xgboost", "chemprop"]
)

# Check paired predictions
assert (
    pred["rf_true"].to_numpy()
    == pred["chemprop_true"].to_numpy()
).all()

assert (
    pred["rf_true"].to_numpy()
    == pred["xgboost_true"].to_numpy()
).all()

# --------------------------------------------------
# 2. Full-dataset scaffold characteristics
# --------------------------------------------------

def size_bin(n):
    if n <= 2:
        return "1-2"
    elif n <= 9:
        return "3-9"
    elif n <= 49:
        return "10-49"
    return "50+"

meta = (
    clean.groupby("Murcko_Scaffold")
    .agg(
        scaffold_total_n=("row_index", "size"),
        target_mean=("PAMPA_clipped", "mean"),
        target_std=("PAMPA_clipped", "std"),
        target_min=("PAMPA_clipped", "min"),
        target_max=("PAMPA_clipped", "max")
    )
    .reset_index()
)

meta["scaffold_size_bin"] = (
    meta["scaffold_total_n"].apply(size_bin)
)

meta["target_range"] = (
    meta["target_max"] - meta["target_min"]
)

meta["target_variability_bin"] = "single_molecule"

valid = meta["target_std"].notna()

if valid.sum() >= 3:
    meta.loc[valid, "target_variability_bin"] = (
        pd.qcut(
            meta.loc[valid, "target_std"],
            q=3,
            duplicates="drop"
        ).astype(str)
    )

# --------------------------------------------------
# 3. Bootstrap helper
# --------------------------------------------------

def bootstrap_ci(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    samples = rng.choice(
        values,
        size=(N_BOOT, len(values)),
        replace=True
    )

    means = samples.mean(axis=1)

    return (
        np.percentile(means, 2.5),
        np.percentile(means, 97.5)
    )

# --------------------------------------------------
# 4. Analyze every split and seed
# --------------------------------------------------

summary = []
scaffold_details = []
correlations = []

for split in ["random", "scaffold"]:

    for seed in [0, 1, 2]:

        print(f"\nProcessing {split} | Seed {seed}")

        d = pred[
            (pred["split"] == split)
            & (pred["seed"] == seed)
        ].copy()

        assert len(d) == 690
        assert d["row_index"].is_unique

        # Absolute prediction errors
        for model in MODELS:
            d[f"{model}_abs_error"] = (
                d[f"{model}_error"].abs()
            )

        # Molecule-level -> scaffold-level
        scaffold = (
            d.groupby("Murcko_Scaffold")
            .agg(
                n_eval=("row_index", "size"),
                rf_mae=("rf_abs_error", "mean"),
                xgboost_mae=("xgboost_abs_error", "mean"),
                chemprop_mae=("chemprop_abs_error", "mean")
            )
            .reset_index()
        )

        scaffold = scaffold.merge(
            meta,
            on="Murcko_Scaffold",
            how="left",
            validate="one_to_one"
        )

        assert scaffold["scaffold_total_n"].notna().all()

        scaffold["split"] = split
        scaffold["seed"] = seed

        for baseline in BASELINES:

            scaffold[f"{baseline}_improvement"] = (
                scaffold[f"{baseline}_mae"]
                - scaffold["chemprop_mae"]
            )

        scaffold_details.append(scaffold)

        # --------------------------------------------------
        # 5. Factor analysis
        # --------------------------------------------------

        factors = [
            "scaffold_size_bin",
            "target_variability_bin"
        ]

        for factor in factors:

            for category, group in scaffold.groupby(
                factor,
                observed=True
            ):

                if len(group) < 3:
                    continue

                scaffold_ids = set(
                    group["Murcko_Scaffold"]
                )

                mol_group = d[
                    d["Murcko_Scaffold"].isin(scaffold_ids)
                ]

                for baseline in BASELINES:

                    improvement = group[
                        f"{baseline}_improvement"
                    ].to_numpy()

                    ci_low, ci_high = bootstrap_ci(
                        improvement
                    )

                    result = {
                        "split": split,
                        "seed": seed,
                        "factor": factor,
                        "category": str(category),
                        "baseline": baseline,
                        "n_scaffolds": len(group),
                        "n_molecules": len(mol_group),

                        "macro_MAE_baseline":
                            group[f"{baseline}_mae"].mean(),

                        "macro_MAE_chemprop":
                            group["chemprop_mae"].mean(),

                        "macro_MAE_improvement":
                            improvement.mean(),

                        "CI95_lower": ci_low,
                        "CI95_upper": ci_high,

                        "scaffold_win_fraction":
                            np.mean(improvement > 0),

                        "micro_MAE_baseline":
                            mol_group[
                                f"{baseline}_abs_error"
                            ].mean(),

                        "micro_MAE_chemprop":
                            mol_group[
                                "chemprop_abs_error"
                            ].mean()
                    }

                    summary.append(result)

        # --------------------------------------------------
        # 6. Continuous scaffold-level correlations
        # --------------------------------------------------

        for baseline in BASELINES:

            improvement = scaffold[
                f"{baseline}_improvement"
            ]

            for feature in [
                "scaffold_total_n",
                "target_std",
                "target_range"
            ]:

                valid_data = scaffold[
                    [feature]
                ].copy()

                valid_mask = (
                    valid_data[feature].notna()
                    & improvement.notna()
                )

                x = valid_data.loc[
                    valid_mask, feature
                ]

                y = improvement.loc[valid_mask]

                if len(x) < 5 or x.nunique() < 2:
                    continue

                rho, pvalue = spearmanr(x, y)

                correlations.append({
                    "split": split,
                    "seed": seed,
                    "baseline": baseline,
                    "feature": feature,
                    "n_scaffolds": len(x),
                    "spearman_rho": rho,
                    "p_value": pvalue
                })

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

details_df = pd.concat(
    scaffold_details,
    ignore_index=True
)

corr_df = pd.DataFrame(correlations)

summary_df.to_csv(
    OUT / "scaffold_factor_summary.csv",
    index=False
)

details_df.to_csv(
    OUT / "scaffold_factor_details.csv",
    index=False
)

corr_df.to_csv(
    OUT / "scaffold_factor_correlations.csv",
    index=False
)

# --------------------------------------------------
# 8. Display
# --------------------------------------------------

print("\n" + "=" * 70)
print("SCAFFOLD FACTOR ANALYSIS COMPLETED")
print("=" * 70)

print("\n===== Factor-level performance =====")

print(
    summary_df[
        [
            "split",
            "seed",
            "factor",
            "category",
            "baseline",
            "n_scaffolds",
            "n_molecules",
            "macro_MAE_improvement",
            "CI95_lower",
            "CI95_upper",
            "scaffold_win_fraction"
        ]
    ].to_string(index=False)
)

print("\n===== Spearman correlations =====")

print(corr_df.to_string(index=False))

print("\nOutput files:")
print(OUT / "scaffold_factor_summary.csv")
print(OUT / "scaffold_factor_details.csv")
print(OUT / "scaffold_factor_correlations.csv")

