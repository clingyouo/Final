import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))

reference = clean[
    ["row_index", "PAMPA_clipped", "Murcko_Scaffold"]
]

all_scaffold_metrics = []
all_summary = []
all_advantages = []
all_wins = []

for split in SPLITS:
    for seed in SEEDS:

        print(f"\n===== {split.upper()} | Seed {seed} =====")

        model_scaffolds = {}

        for model in MODELS:

            path = (
                Path("results/models") / model /
                split / f"seed_{seed}" /
                "predictions_test.csv"
            )

            pred = pd.read_csv(path)

            df = pred.merge(
                reference,
                on="row_index",
                how="left",
                validate="one_to_one"
            )

            assert len(df) == len(pred)
            assert np.allclose(
                df["y_true"],
                df["PAMPA_clipped"],
                atol=1e-7
            )

            df["abs_error"] = (
                df["y_true"] - df["y_pred"]
            ).abs()

            df["squared_error"] = (
                df["y_true"] - df["y_pred"]
            ) ** 2

            # Calculate metrics for each scaffold
            grouped = df.groupby(
                "Murcko_Scaffold",
                dropna=False
            )

            records = []

            for scaffold, group in grouped:

                record = {
                    "split": split,
                    "seed": seed,
                    "model": model,
                    "scaffold": scaffold,
                    "n_molecules": len(group),
                    "MAE": group["abs_error"].mean(),
                    "RMSE": np.sqrt(
                        group["squared_error"].mean()
                    ),
                    "bias": (
                        group["y_pred"] - group["y_true"]
                    ).mean()
                }

                records.append(record)
                all_scaffold_metrics.append(record)

            scaffold_df = pd.DataFrame(records)
            model_scaffolds[model] = scaffold_df

            # Micro and macro metrics
            all_summary.append({
                "split": split,
                "seed": seed,
                "model": model,
                "n_molecules": len(df),
                "n_scaffolds": len(scaffold_df),
                "micro_MAE": df["abs_error"].mean(),
                "micro_RMSE": np.sqrt(
                    df["squared_error"].mean()
                ),
                "macro_MAE": scaffold_df["MAE"].mean(),
                "macro_RMSE": scaffold_df["RMSE"].mean(),
                "median_scaffold_MAE": (
                    scaffold_df["MAE"].median()
                )
            })

        # Compare models scaffold by scaffold
        pivot = {}

        for model in MODELS:
            pivot[model] = (
                model_scaffolds[model]
                .set_index("scaffold")["MAE"]
            )

        comparison = pd.DataFrame(pivot)

        for baseline in ["rf", "xgboost"]:

            improvement = (
                comparison[baseline] -
                comparison["chemprop"]
            )

            all_advantages.append({
                "split": split,
                "seed": seed,
                "baseline": baseline,
                "n_scaffolds": len(comparison),
                "mean_scaffold_MAE_improvement": (
                    improvement.mean()
                ),
                "median_scaffold_MAE_improvement": (
                    improvement.median()
                ),
                "chemprop_better_fraction": (
                    improvement.gt(0).mean()
                ),
                "chemprop_better_count": (
                    improvement.gt(0).sum()
                )
            })

        # Count scaffold-level wins
        winners = comparison.idxmin(axis=1)

        for model in MODELS:
            count = (winners == model).sum()

            all_wins.append({
                "split": split,
                "seed": seed,
                "model": model,
                "scaffold_wins": count,
                "win_fraction": count / len(winners)
            })

# Save outputs
pd.DataFrame(all_scaffold_metrics).to_csv(
    OUT / "per_scaffold_metrics.csv", index=False
)

summary = pd.DataFrame(all_summary)
summary.to_csv(
    OUT / "micro_macro_metrics_by_seed.csv",
    index=False
)

aggregate = (
    summary.groupby(["split", "model"])
    .agg(
        micro_MAE_mean=("micro_MAE", "mean"),
        micro_MAE_std=("micro_MAE", "std"),
        macro_MAE_mean=("macro_MAE", "mean"),
        macro_MAE_std=("macro_MAE", "std"),
        macro_RMSE_mean=("macro_RMSE", "mean"),
        median_scaffold_MAE_mean=(
            "median_scaffold_MAE", "mean"
        )
    )
)

aggregate.to_csv(
    OUT / "micro_macro_summary.csv"
)

pd.DataFrame(all_advantages).to_csv(
    OUT / "paired_scaffold_advantage.csv",
    index=False
)

pd.DataFrame(all_wins).to_csv(
    OUT / "scaffold_win_counts.csv",
    index=False
)

print("\n" + "=" * 65)
print("SCAFFOLD MACRO/MICRO ANALYSIS COMPLETED")
print("=" * 65)

print("\n===== Micro vs Macro =====")
print(aggregate.round(4).to_string())

print("\n===== Chemprop scaffold advantage =====")
print(
    pd.DataFrame(all_advantages)
    .round(4)
    .to_string(index=False)
)

print("\n===== Scaffold-level wins =====")
print(
    pd.DataFrame(all_wins)
    .to_string(index=False)
)

print("\nOutput:", OUT)
