import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(".")
OUT = ROOT / "results/analysis/representation_scaffold"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgboost", "chemprop"]
SPLITS = ["random", "scaffold"]
SEEDS = [0, 1, 2]

clean = pd.read_csv("data/processed/clean.csv")
clean["row_index"] = np.arange(len(clean))

# 每种 scaffold 在完整数据集中的样本数
scaffold_counts = clean["Murcko_Scaffold"].value_counts()

def size_bin(n):
    if n <= 2:
        return "1-2"
    elif n <= 9:
        return "3-9"
    elif n <= 49:
        return "10-49"
    return "50+"

clean["scaffold_total_n"] = clean["Murcko_Scaffold"].map(
    scaffold_counts
)

clean["scaffold_size_bin"] = clean["scaffold_total_n"].map(
    size_bin
)

scaffold_rows = []
bin_rows = []
molecule_rows = []

for split in SPLITS:
    for seed in SEEDS:

        print(f"\nProcessing {split}, seed {seed}")

        split_dir = ROOT / "splits" / split / f"seed_{seed}"
        expected = pd.read_csv(split_dir / "test.csv")

        expected_ids = set(expected["row_index"])

        # 以 row_index 为唯一键合并三个模型
        merged = None

        for model in MODELS:

            path = (
                ROOT / "results/models" /
                model / split / f"seed_{seed}" /
                "predictions_test.csv"
            )

            pred = pd.read_csv(path)

            if pred["row_index"].duplicated().any():
                raise ValueError(f"Duplicate index in {path}")

            if set(pred["row_index"]) != expected_ids:
                raise ValueError(f"Test index mismatch: {path}")

            pred = pred.set_index("row_index")

            if merged is None:
                merged = pred[
                    ["canonical_smiles", "y_true", "y_pred"]
                ].rename(columns={
                    "canonical_smiles": f"{model}_smiles",
                    "y_true": f"{model}_true",
                    "y_pred": f"{model}_pred"
                })
            else:
                selected = pred[
                    ["canonical_smiles", "y_true", "y_pred"]
                ].rename(columns={
                    "canonical_smiles": f"{model}_smiles",
                    "y_true": f"{model}_true",
                    "y_pred": f"{model}_pred"
                })

                merged = merged.join(
                    selected, validate="one_to_one"
                )

        merged = merged.reset_index()

        # 加入冻结数据中的 scaffold
        merged = merged.merge(
            clean[
                [
                    "row_index",
                    "canonical_smiles",
                    "PAMPA_clipped",
                    "Murcko_Scaffold",
                    "scaffold_total_n",
                    "scaffold_size_bin"
                ]
            ],
            on="row_index",
            validate="one_to_one"
        )

        # 样本与目标一致性检查
        for model in MODELS:

            if not (
                merged[f"{model}_smiles"]
                == merged["canonical_smiles"]
            ).all():
                raise ValueError(f"SMILES mismatch: {model}")

            if not np.allclose(
                merged[f"{model}_true"],
                merged["PAMPA_clipped"],
                atol=1e-8
            ):
                raise ValueError(f"Target mismatch: {model}")

            merged[f"{model}_error"] = np.abs(
                merged["PAMPA_clipped"]
                - merged[f"{model}_pred"]
            )

        merged["split"] = split
        merged["seed"] = seed

        molecule_rows.append(merged)

        # 逐 scaffold 计算模型表现
        for scaffold, group in merged.groupby("Murcko_Scaffold"):

            row = {
                "split": split,
                "seed": seed,
                "scaffold": scaffold,
                "n_test": len(group),
                "scaffold_total_n": int(
                    group["scaffold_total_n"].iloc[0]
                ),
                "scaffold_size_bin": (
                    group["scaffold_size_bin"].iloc[0]
                ),
                "mean_PAMPA": group["PAMPA_clipped"].mean(),
                "std_PAMPA": group["PAMPA_clipped"].std()
            }

            for model in MODELS:
                row[f"MAE_{model}"] = group[
                    f"{model}_error"
                ].mean()

                row[f"RMSE_{model}"] = np.sqrt(
                    np.mean(group[f"{model}_error"] ** 2)
                )

            row["improvement_RF"] = (
                row["MAE_rf"] - row["MAE_chemprop"]
            )

            row["improvement_XGBoost"] = (
                row["MAE_xgboost"] - row["MAE_chemprop"]
            )

            row["chemprop_better_fraction_RF"] = (
                group["chemprop_error"]
                < group["rf_error"]
            ).mean()

            row["chemprop_better_fraction_XGBoost"] = (
                group["chemprop_error"]
                < group["xgboost_error"]
            ).mean()

            scaffold_rows.append(row)

        # 按完整数据集 scaffold 大小分组
        for bin_name, group in merged.groupby("scaffold_size_bin"):

            row = {
                "split": split,
                "seed": seed,
                "scaffold_size_bin": bin_name,
                "n_test": len(group),
                "n_scaffolds": group["Murcko_Scaffold"].nunique()
            }

            for model in MODELS:
                row[f"MAE_{model}"] = mean_absolute_error(
                    group["PAMPA_clipped"],
                    group[f"{model}_pred"]
                )

                row[f"RMSE_{model}"] = np.sqrt(
                    mean_squared_error(
                        group["PAMPA_clipped"],
                        group[f"{model}_pred"]
                    )
                )

            row["improvement_RF"] = (
                row["MAE_rf"] - row["MAE_chemprop"]
            )

            row["improvement_XGBoost"] = (
                row["MAE_xgboost"] - row["MAE_chemprop"]
            )

            bin_rows.append(row)

# 保存逐 scaffold 结果
scaffold_df = pd.DataFrame(scaffold_rows)

scaffold_df.to_csv(
    OUT / "scaffold_level_metrics.csv",
    index=False
)

# 保存 scaffold 大小分组结果
bin_df = pd.DataFrame(bin_rows)

bin_df.to_csv(
    OUT / "scaffold_size_bin_metrics.csv",
    index=False
)

# 提取有一定样本量的优势和劣势 scaffold
eligible = scaffold_df[
    scaffold_df["n_test"] >= 5
].copy()

top_bottom = []

for (split, seed), group in eligible.groupby(["split", "seed"]):

    for comparison in ["RF", "XGBoost"]:

        col = f"improvement_{comparison}"

        for rank, (_, row) in enumerate(
            group.nlargest(10, col).iterrows(), 1
        ):
            result = row.to_dict()
            result["comparison"] = comparison
            result["ranking"] = "top"
            result["rank"] = rank
            top_bottom.append(result)

        for rank, (_, row) in enumerate(
            group.nsmallest(10, col).iterrows(), 1
        ):
            result = row.to_dict()
            result["comparison"] = comparison
            result["ranking"] = "bottom"
            result["rank"] = rank
            top_bottom.append(result)

pd.DataFrame(top_bottom).to_csv(
    OUT / "top_bottom_scaffolds.csv",
    index=False
)

# 保存分子级结果，便于后续绘图
pd.concat(molecule_rows, ignore_index=True).to_csv(
    OUT / "scaffold_molecule_predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("SCAFFOLD ANALYSIS COMPLETED")
print("=" * 60)

print("\nScaffold size-bin results:")
print(
    bin_df.groupby(
        ["split", "scaffold_size_bin"]
    )[
        [
            "MAE_rf",
            "MAE_xgboost",
            "MAE_chemprop",
            "improvement_RF",
            "improvement_XGBoost"
        ]
    ].mean().round(4)
)

print("\nScaffold rows:", len(scaffold_df))
print("Eligible scaffold rows (n >= 5):", len(eligible))

print("\nOutput:", OUT)
