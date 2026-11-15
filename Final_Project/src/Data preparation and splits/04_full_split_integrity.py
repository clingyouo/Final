import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

OUT = Path("results/analysis/final_comparison")
OUT.mkdir(parents=True, exist_ok=True)

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))
clean = clean.set_index("row_index", drop=False)

PARTITIONS = ["train", "val", "test"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

pair_results = []
partition_results = []
coverage_results = []

def valid_set(series):
    values = series.dropna().astype(str).str.strip()
    return set(values[~values.isin(["", "nan", "None"])])

for split in SPLITS:
    for seed in SEEDS:

        print(f"\n===== {split.upper()} | Seed {seed} =====")

        data = {}

        for name in PARTITIONS:
            idx = pd.read_csv(
                f"splits/{split}/seed_{seed}/{name}.csv"
            )["row_index"].astype(int).tolist()

            assert len(idx) == len(set(idx))

            data[name] = clean.loc[idx]

            partition_results.append({
                "split": split,
                "seed": seed,
                "partition": name,
                "n_molecules": len(idx),
                "n_unique_smiles":
                    data[name]["canonical_smiles"].nunique(),
                "n_unique_scaffolds":
                    len(valid_set(data[name]["Murcko_Scaffold"])),
                "duplicate_smiles":
                    int(data[name]["canonical_smiles"].duplicated().sum())
            })

        # Check full dataset coverage
        all_indices = set()

        for name in PARTITIONS:
            all_indices.update(data[name].index)

        expected_indices = set(clean.index)

        missing = expected_indices - all_indices
        extra = all_indices - expected_indices

        coverage_results.append({
            "split": split,
            "seed": seed,
            "total_rows": len(clean),
            "assigned_rows": len(all_indices),
            "missing_rows": len(missing),
            "extra_rows": len(extra)
        })

        # Pairwise comparisons
        for a, b in combinations(PARTITIONS, 2):

            A = data[a]
            B = data[b]

            row_overlap = set(A.index) & set(B.index)

            smiles_overlap = (
                set(A["canonical_smiles"])
                & set(B["canonical_smiles"])
            )

            scaffold_A = valid_set(A["Murcko_Scaffold"])
            scaffold_B = valid_set(B["Murcko_Scaffold"])

            shared_scaffolds = scaffold_A & scaffold_B

            molecules_shared = B[
                B["Murcko_Scaffold"].astype(str).isin(scaffold_A)
            ]

            result = {
                "split": split,
                "seed": seed,
                "partition_A": a,
                "partition_B": b,
                "row_overlap": len(row_overlap),
                "exact_smiles_overlap": len(smiles_overlap),
                "shared_scaffolds": len(shared_scaffolds),
                "molecules_sharing_scaffold":
                    len(molecules_shared)
            }

            pair_results.append(result)

            print(
                f"{a} vs {b}: "
                f"rows={len(row_overlap)}, "
                f"SMILES={len(smiles_overlap)}, "
                f"scaffolds={len(shared_scaffolds)}"
            )

# Save results
pair_df = pd.DataFrame(pair_results)
partition_df = pd.DataFrame(partition_results)
coverage_df = pd.DataFrame(coverage_results)

pair_df.to_csv(
    OUT / "full_split_pairwise_integrity.csv",
    index=False
)

partition_df.to_csv(
    OUT / "full_split_partition_summary.csv",
    index=False
)

coverage_df.to_csv(
    OUT / "full_split_coverage.csv",
    index=False
)

print("\n" + "=" * 65)
print("FULL SPLIT INTEGRITY COMPLETED")
print("=" * 65)

print("\n===== Pairwise integrity =====")
print(pair_df.to_string(index=False))

print("\n===== Partition summary =====")
print(partition_df.to_string(index=False))

print("\n===== Dataset coverage =====")
print(coverage_df.to_string(index=False))

print("\nSaved to:", OUT)
