import numpy as np
import pandas as pd

from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))

# Generate Morgan fingerprints
mols = [
    Chem.MolFromSmiles(s)
    for s in clean["canonical_smiles"]
]

assert all(m is not None for m in mols)

generators = {
    "achiral": rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    ),
    "chiral": rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
}

fingerprints = {
    name: [generator.GetFingerprint(m) for m in mols]
    for name, generator in generators.items()
}

molecule_records = []

for split in SPLITS:
    for seed in SEEDS:

        print(f"\nProcessing {split} | Seed {seed}")

        train_idx = pd.read_csv(
            f"splits/{split}/seed_{seed}/train.csv"
        )["row_index"].astype(int).tolist()

        test_idx = pd.read_csv(
            f"splits/{split}/seed_{seed}/test.csv"
        )["row_index"].astype(int).tolist()

        # Calculate maximum and top-5 similarity
        similarity_results = {}

        for fp_name, fps in fingerprints.items():

            train_fps = [fps[i] for i in train_idx]

            for idx in test_idx:

                sims = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        fps[idx], train_fps
                    )
                )

                top5 = np.sort(sims)[-5:]

                similarity_results.setdefault(idx, {})[
                    f"max_sim_{fp_name}"
                ] = float(sims.max())

                similarity_results[idx][
                    f"top5_sim_{fp_name}"
                ] = float(top5.mean())

        # Read model predictions
        predictions = {}

        for model in MODELS:

            path = Path(
                f"results/models/{model}/{split}/seed_{seed}/"
                "predictions_test.csv"
            )

            pred = pd.read_csv(path)

            assert set(pred["row_index"].astype(int)) == set(test_idx)

            predictions[model] = pred.set_index("row_index")

        for idx in test_idx:

            row = clean.iloc[idx]

            record = {
                "split": split,
                "seed": seed,
                "row_index": idx,
                "Source": row["Source"],
                "Year": row["Year"],
                "Murcko_Scaffold": row["Murcko_Scaffold"],
                "PAMPA_clipped": row["PAMPA_clipped"],
            }

            record.update(similarity_results[idx])

            for model in MODELS:

                pred = predictions[model].loc[idx]

                record[f"pred_{model}"] = pred["y_pred"]

                record[f"error_{model}"] = abs(
                    pred["y_true"] - pred["y_pred"]
                )

            molecule_records.append(record)

# Save molecule-level results
df = pd.DataFrame(molecule_records)

df.to_csv(
    OUT / "cross_seed_similarity_molecule_level.csv",
    index=False
)

# Spearman correlation
correlations = []

for (split, seed), group in df.groupby(["split", "seed"]):

    for fp_name in ["achiral", "chiral"]:

        sim_col = f"max_sim_{fp_name}"

        for model in MODELS:

            rho, p = spearmanr(
                group[sim_col],
                group[f"error_{model}"]
            )

            correlations.append({
                "split": split,
                "seed": seed,
                "fingerprint": fp_name,
                "model": model,
                "n": len(group),
                "spearman_rho": rho,
                "p_value": p
            })

corr_df = pd.DataFrame(correlations)

corr_df.to_csv(
    OUT / "cross_seed_similarity_correlations.csv",
    index=False
)

# Error by nearest-neighbor similarity
bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]

labels = [
    "0-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0"
]

bin_records = []

for fp_name in ["achiral", "chiral"]:

    sim_col = f"max_sim_{fp_name}"

    df["similarity_bin"] = pd.cut(
        df[sim_col],
        bins=bins,
        labels=labels,
        right=False
    )

    for (split, seed, sim_bin), group in df.groupby(
        ["split", "seed", "similarity_bin"],
        observed=True
    ):

        for model in MODELS:

            bin_records.append({
                "split": split,
                "seed": seed,
                "fingerprint": fp_name,
                "similarity_bin": str(sim_bin),
                "model": model,
                "n": len(group),
                "MAE": group[f"error_{model}"].mean(),
                "median_error": group[f"error_{model}"].median()
            })

bin_df = pd.DataFrame(bin_records)

bin_df.to_csv(
    OUT / "cross_seed_similarity_bins.csv",
    index=False
)

# Paired model advantage in each similarity bin
advantage_records = []

for (split, seed, fp_name, sim_bin), group in bin_df.groupby(
    ["split", "seed", "fingerprint", "similarity_bin"]
):

    values = group.set_index("model")["MAE"]

    if not all(m in values.index for m in MODELS):
        continue

    advantage_records.append({
        "split": split,
        "seed": seed,
        "fingerprint": fp_name,
        "similarity_bin": sim_bin,
        "n": int(group["n"].iloc[0]),
        "RF_MAE": values["rf"],
        "XGBoost_MAE": values["xgboost"],
        "Chemprop_MAE": values["chemprop"],
        "RF_improvement": values["rf"] - values["chemprop"],
        "XGBoost_improvement":
            values["xgboost"] - values["chemprop"]
    })

adv_df = pd.DataFrame(advantage_records)

adv_df.to_csv(
    OUT / "cross_seed_similarity_model_advantage.csv",
    index=False
)

print("\n" + "=" * 65)
print("CROSS-SEED SIMILARITY ANALYSIS COMPLETED")
print("=" * 65)

print("\n===== Spearman correlations =====")
print(corr_df.to_string(index=False))

print("\n===== Model advantage by similarity =====")
print(adv_df.to_string(index=False))

print("\nOutput directory:", OUT)
