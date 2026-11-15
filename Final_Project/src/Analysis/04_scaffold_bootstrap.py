import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 10000
SEED = 20261105

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

source = pd.read_csv(
    OUT / "per_scaffold_metrics.csv"
)

rng = np.random.default_rng(SEED)

results = []

for split in SPLITS:
    for seed in SEEDS:

        subset = source[
            (source["split"] == split) &
            (source["seed"] == seed)
        ]

        pivot = subset.pivot(
            index="scaffold",
            columns="model",
            values="MAE"
        )

        assert set(MODELS).issubset(pivot.columns)
        assert not pivot[MODELS].isna().any().any()

        n = len(pivot)

        print(f"\n===== {split.upper()} | Seed {seed} =====")
        print("Number of scaffolds:", n)

        for baseline in ["rf", "xgboost"]:

            # Positive difference favors Chemprop
            diff = (
                pivot[baseline].to_numpy() -
                pivot["chemprop"].to_numpy()
            )

            observed = diff.mean()

            # Resample scaffold units, not individual molecules
            sampled = rng.choice(
                diff,
                size=(N_BOOT, n),
                replace=True
            )

            boot_means = sampled.mean(axis=1)

            lower, upper = np.percentile(
                boot_means, [2.5, 97.5]
            )

            record = {
                "split": split,
                "seed": seed,
                "baseline": baseline,
                "n_scaffolds": n,
                "mean_MAE_improvement": observed,
                "CI95_lower": lower,
                "CI95_upper": upper,
                "bootstrap_fraction_positive": (
                    boot_means > 0
                ).mean(),
                "scaffold_win_fraction": (
                    diff > 0
                ).mean()
            }

            results.append(record)

            print(
                f"{baseline} vs Chemprop:"
                f" improvement={observed:.4f},"
                f" 95% CI=[{lower:.4f}, {upper:.4f}],"
                f" win fraction={(diff > 0).mean():.1%}"
            )

result = pd.DataFrame(results)

result.to_csv(
    OUT / "scaffold_bootstrap_ci.csv",
    index=False
)

# Descriptive summary across seeds.
# Do not treat seeds as independent bootstrap samples.
summary = (
    result.groupby(["split", "baseline"])
    .agg(
        mean_improvement=("mean_MAE_improvement", "mean"),
        min_improvement=("mean_MAE_improvement", "min"),
        max_improvement=("mean_MAE_improvement", "max"),
        mean_win_fraction=("scaffold_win_fraction", "mean"),
        seeds=("seed", "nunique")
    )
)

summary.to_csv(
    OUT / "scaffold_bootstrap_summary.csv"
)

print("\n" + "=" * 65)
print("SCAFFOLD BOOTSTRAP COMPLETED")
print("=" * 65)

print("\nDetailed results:")
print(result.round(4).to_string(index=False))

print("\nAcross-seed descriptive summary:")
print(summary.round(4).to_string())

print("\nOutput:", OUT)
