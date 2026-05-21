#!/usr/bin/env python3
"""
Reproduce ELAPSE paper statistical tables from released raw t-test results.

This script is the command-line counterpart of
`code/statistics/selection-impact-t-test-w-correction.ipynb`.
It expects the raw CSV files produced by `selection-impact-t-test.ipynb` or by
an equivalent run of the selection-impact analysis, namely files named:

    <dataset>_ttest_results_epochs_intervals_raw_effect_size.csv

By default, the script:
  1. loads all raw t-test/effect-size CSV files;
  2. applies Holm-Bonferroni correction globally across all p-values;
  3. applies practical post-processing thresholds;
  4. exports impact tables with and without Random;
  5. exports effect-size summaries;
  6. exports CSV and LaTeX versions of the paper tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


DEFAULT_DATASETS = [
    "ars", "dc", "kdd", "adult", "mobiact",
    "celeba", "fairface", "audiomnist", "voxceleb",
]

DEFAULT_METRICS = [
    "time", "acc", "f1", "precision", "recall",
    "SPD_gender", "EOD_gender", "AOD_gender", "DI_gender", "DcI_gender",
    "SPD_age", "EOD_age", "AOD_age", "DI_age", "DcI_age",
    "SPD_race", "EOD_race", "AOD_race", "DI_race", "DcI_race",
]

UTILITY_METRICS = ["acc", "f1", "precision", "recall"]
UTILITY_AVG_COLS = {
    "acc": "Accuracy_avg",
    "f1": "F1_score_avg",
    "precision": "Precision_avg",
    "recall": "Recall_avg",
}

FAIRNESS_METRICS = [m for m in DEFAULT_METRICS if m.startswith(("SPD_", "EOD_", "AOD_", "DI_", "DcI_"))]
FAIRNESS_AVG_COLS = {m: f"{m}_avg" for m in FAIRNESS_METRICS}

LABEL_VALUES = {"positive", "negative", "insignificant"}
INSIGNIFICANT_FLAGS = {
    "insignificant-p", "insignificant-p2", "insignificant-p3",
    "insignificant-pu", "insignificant-pu2",
}


def _label_from_significance(significant: bool, effect_size: float) -> str:
    if not significant or pd.isna(effect_size):
        return "insignificant"
    if effect_size > 0:
        return "positive"
    if effect_size < 0:
        return "negative"
    return "insignificant"


def _read_raw_results(input_dir: Path, datasets: Sequence[str]) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        path = input_dir / f"{dataset}_ttest_results_epochs_intervals_raw_effect_size.csv"
        if not path.exists():
            print(f"Missing raw file: {path}")
            continue
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = dataset
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No raw t-test files were found in {input_dir}. "
            "Expected files named <dataset>_ttest_results_epochs_intervals_raw_effect_size.csv."
        )

    return pd.concat(frames, ignore_index=True)


def apply_holm_global(df: pd.DataFrame, metrics: Sequence[str], alpha: float) -> pd.DataFrame:
    """Apply Holm-Bonferroni globally across all available raw p-values."""
    out = df.copy()
    p_values: list[float] = []
    index_map: list[tuple[int, str]] = []

    for idx, row in out.iterrows():
        if row.get("system") == "Full":
            continue
        for metric in metrics:
            p_col = f"test_{metric}_p_raw"
            effect_col = f"test_{metric}_effect_size_raw"
            if p_col not in out.columns or effect_col not in out.columns:
                continue
            p_value = pd.to_numeric(row.get(p_col), errors="coerce")
            if pd.isna(p_value):
                continue
            p_values.append(float(p_value))
            index_map.append((idx, metric))

    if not p_values:
        print("No p-values found for Holm correction. Returning input dataframe unchanged.")
        return out

    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method="holm")

    for i, (idx, metric) in enumerate(index_map):
        effect_col = f"test_{metric}_effect_size_raw"
        effect_size = pd.to_numeric(out.loc[idx, effect_col], errors="coerce")

        out.loc[idx, f"test_{metric}_p_holm_global"] = p_corr[i]
        out.loc[idx, f"test_{metric}_significant_holm_global"] = bool(reject[i])
        out.loc[idx, f"test_{metric}_effect_size_holm_global"] = effect_size

        drm_col = f"test_{metric}_effect_size_drm_raw"
        corr_col = f"test_{metric}_correlation_r_raw"
        if drm_col in out.columns:
            out.loc[idx, f"test_{metric}_effect_size_drm_holm_global"] = out.loc[idx, drm_col]
        if corr_col in out.columns:
            out.loc[idx, f"test_{metric}_correlation_r_holm_global"] = out.loc[idx, corr_col]

        out.loc[idx, f"test_{metric}_holm_global"] = _label_from_significance(bool(reject[i]), effect_size)

    return out


def _reference_rows(df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    refs = {}
    full_df = df[df["system"] == "Full"] if "system" in df.columns else pd.DataFrame()
    for _, row in full_df.iterrows():
        refs[(row.get("dataset"), row.get("model"))] = row
    return refs


def apply_postprocessing_threshold(df: pd.DataFrame, threshold: float, suffix: str = "holm_global") -> pd.DataFrame:
    """Turn statistically significant but practically tiny effects into insignificant labels.

    `threshold` is expressed in the same scale as the stored CSV values. In the ELAPSE
    result CSVs, accuracy/F1/precision/recall and most fairness metrics are expressed in
    percentages, while DI is a ratio. Therefore, for non-DI metrics we use threshold*100;
    for DI metrics we use threshold directly.
    """
    out = df.copy()
    refs = _reference_rows(out)

    for idx, row in out.iterrows():
        if row.get("system") == "Full":
            continue
        ref = refs.get((row.get("dataset"), row.get("model")))
        if ref is None:
            continue

        # Accuracy guard from the original notebook: when accuracy degradation is large,
        # positive fairness changes are treated cautiously.
        acc_col = UTILITY_AVG_COLS["acc"]
        if acc_col in out.columns and pd.notna(row.get(acc_col)) and pd.notna(ref.get(acc_col)):
            acc_drop = float(ref[acc_col]) - float(row[acc_col])
            if acc_drop >= threshold * 100:
                for metric in FAIRNESS_METRICS:
                    test_col = f"test_{metric}_{suffix}"
                    if test_col in out.columns and row.get(test_col) == "positive":
                        out.at[idx, test_col] = "insignificant-p"

        # Utility labels: positive/negative changes below the practical threshold.
        for metric, avg_col in UTILITY_AVG_COLS.items():
            test_col = f"test_{metric}_{suffix}"
            if test_col not in out.columns or avg_col not in out.columns:
                continue
            if row.get(test_col) not in {"positive", "negative"}:
                continue
            if pd.isna(row.get(avg_col)) or pd.isna(ref.get(avg_col)):
                continue
            if abs(float(ref[avg_col]) - float(row[avg_col])) < threshold * 100:
                out.at[idx, test_col] = "insignificant-pu" if row.get(test_col) == "positive" else "insignificant-pu2"

        # Fairness labels: positive/negative changes below the practical threshold.
        for metric, avg_col in FAIRNESS_AVG_COLS.items():
            test_col = f"test_{metric}_{suffix}"
            if test_col not in out.columns or avg_col not in out.columns:
                continue
            if row.get(test_col) not in {"positive", "negative"}:
                continue
            if pd.isna(row.get(avg_col)) or pd.isna(ref.get(avg_col)):
                continue
            metric_threshold = threshold if metric.startswith("DI_") else threshold * 100
            if abs(float(ref[avg_col]) - float(row[avg_col])) < metric_threshold:
                out.at[idx, test_col] = "insignificant-p2" if row.get(test_col) == "positive" else "insignificant-p3"

    return out


def normalize_insignificant_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.replace({flag: "insignificant" for flag in INSIGNIFICANT_FLAGS})


def impact_columns(metrics: Sequence[str], suffix: str = "holm_global") -> list[str]:
    return [f"test_{metric}_{suffix}" for metric in metrics]


def export_impact_files(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    metrics: Sequence[str],
    suffix: str = "holm_global",
) -> pd.DataFrame:
    label = str(threshold).replace(".", "p")
    cols = ["dataset", "model", "system", "ratio"] + [c for c in impact_columns(metrics, suffix) if c in df.columns]
    out = df[cols].copy()

    out.to_csv(output_dir / f"table_impact_{suffix}_{label}_with_flags.csv", index=False)
    out = normalize_insignificant_labels(out)
    out.to_csv(output_dir / f"table_impact_{suffix}_{label}.csv", index=False)

    main = out[(out["system"] != "Full") & (pd.to_numeric(out["ratio"], errors="coerce") != 0.5)].copy()
    main.to_csv(output_dir / f"table_impact_{suffix}_{label}_with_random.csv", index=False)
    no_random = main[main["system"] != "Random"].copy()
    no_random.to_csv(output_dir / f"table_impact_{suffix}_{label}_wo_random.csv", index=False)

    with open(output_dir / f"table_impact_{suffix}_{label}_wo_random.tex", "w", encoding="utf-8") as f:
        f.write(no_random.to_latex(index=False, escape=False))

    return no_random


def build_effect_size_wide(df: pd.DataFrame, metrics: Sequence[str], suffix: str = "holm_global") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        out = {
            "dataset": row.get("dataset"),
            "model": row.get("model"),
            "system": row.get("system"),
            "ratio": row.get("ratio"),
        }
        for metric in metrics:
            col = f"test_{metric}_effect_size_{suffix}"
            out[f"effect_size_{metric}"] = row.get(col, np.nan) if col in df.columns else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_effect_sizes(
    impact_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    metrics: Sequence[str],
    excluded_spd_di_datasets: Sequence[str],
    suffix: str = "holm_global",
) -> pd.DataFrame:
    impact_df = impact_df[impact_df["system"] != "Random"].copy()
    effect_df = effect_df[effect_df["system"] != "Random"].copy()
    df = impact_df.merge(effect_df, on=["dataset", "model", "system", "ratio"], how="left")

    rows = []
    for metric in metrics:
        impact_col = f"test_{metric}_{suffix}"
        effect_col = f"effect_size_{metric}"
        if impact_col not in df.columns or effect_col not in df.columns:
            continue
        tmp = df[["dataset", impact_col, effect_col]].copy()
        if metric.startswith("SPD_") or metric.startswith("DI_"):
            tmp = tmp[~tmp["dataset"].isin(excluded_spd_di_datasets)]
        tmp[impact_col] = tmp[impact_col].replace({flag: "insignificant" for flag in INSIGNIFICANT_FLAGS})
        tmp[effect_col] = pd.to_numeric(tmp[effect_col], errors="coerce")
        total = tmp[impact_col].isin(LABEL_VALUES).sum()
        if total == 0:
            continue
        row = {"metric": metric, "total_cases": int(total)}
        for label in ["negative", "positive", "insignificant"]:
            subset = tmp[tmp[impact_col] == label]
            row[f"{label}_count"] = int(len(subset))
            row[f"{label}_percentage"] = 100 * len(subset) / total
            row[f"{label}_mean_effect_size"] = subset[effect_col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_counts(df: pd.DataFrame, metrics: Sequence[str], suffix: str = "holm_global") -> pd.DataFrame:
    rows = []
    for metric in metrics:
        col = f"test_{metric}_{suffix}"
        if col not in df.columns:
            continue
        values = df[col].replace({flag: "insignificant" for flag in INSIGNIFICANT_FLAGS})
        total = values.isin(LABEL_VALUES).sum()
        if total == 0:
            continue
        rows.append({
            "metric": metric,
            "negative_count": int((values == "negative").sum()),
            "negative_percentage": 100 * (values == "negative").sum() / total,
            "positive_count": int((values == "positive").sum()),
            "positive_percentage": 100 * (values == "positive").sum() / total,
            "insignificant_count": int((values == "insignificant").sum()),
            "insignificant_percentage": 100 * (values == "insignificant").sum() / total,
            "total_cases": int(total),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    metrics = args.metrics

    raw = _read_raw_results(input_dir, datasets)
    raw.to_csv(output_dir / "all_datasets_raw_before_postprocessing.csv", index=False)

    holm = apply_holm_global(raw, metrics, alpha=args.alpha)
    holm.to_csv(output_dir / "all_datasets_holm_global_before_postprocessing.csv", index=False)

    for dataset in datasets:
        tmp = holm[holm["dataset"] == dataset]
        if len(tmp) > 0:
            tmp.to_csv(output_dir / f"{dataset}_ttest_results_epochs_intervals_holm_global.csv", index=False)

    effect_wide = build_effect_size_wide(holm, metrics, suffix="holm_global")
    effect_wide.to_csv(output_dir / "effect_size_wide_holm_global.csv", index=False)

    threshold_outputs = []
    for threshold in args.post_thresholds:
        processed = apply_postprocessing_threshold(holm, threshold=threshold, suffix="holm_global")
        processed_path = output_dir / f"all_datasets_holm_global_postprocessed_threshold_{str(threshold).replace('.', 'p')}.csv"
        processed.to_csv(processed_path, index=False)
        table = export_impact_files(processed, output_dir, threshold, metrics, suffix="holm_global")
        summary = summarize_counts(table, metrics, suffix="holm_global")
        summary.to_csv(output_dir / f"summary_counts_holm_global_threshold_{str(threshold).replace('.', 'p')}.csv", index=False)
        threshold_outputs.append((threshold, table))

    main_threshold = args.main_threshold
    processed_main = apply_postprocessing_threshold(holm, threshold=main_threshold, suffix="holm_global")
    main_table = export_impact_files(processed_main, output_dir, main_threshold, metrics, suffix="holm_global")
    effect_summary = summarize_effect_sizes(
        impact_df=main_table,
        effect_df=effect_wide,
        metrics=metrics,
        excluded_spd_di_datasets=args.exclude_spd_di_datasets,
        suffix="holm_global",
    )
    effect_summary.to_csv(output_dir / "table_effect_size_summary.csv", index=False)
    with open(output_dir / "table_effect_size_summary.tex", "w", encoding="utf-8") as f:
        f.write(effect_summary.to_latex(index=False, escape=False))

    print("Saved reproduced tables in:", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce ELAPSE paper tables from raw t-test CSV files.")
    parser.add_argument("--results-root", default="results", help="Root results directory, kept for README compatibility.")
    parser.add_argument("--input-dir", default="results/test-p-value-005-raw-effect-size", help="Directory containing raw t-test CSV files.")
    parser.add_argument("--output-dir", default="results/paper_tables", help="Directory where reproduced tables are saved.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for Holm correction.")
    parser.add_argument("--correction", default="holm", choices=["holm"], help="Multiple-comparison correction method.")
    parser.add_argument("--post-thresholds", nargs="+", type=float, default=[0.005, 0.0075, 0.01, 0.02, 0.03], help="Practical post-processing thresholds.")
    parser.add_argument("--main-threshold", type=float, default=0.01, help="Main threshold used for effect-size summary and primary table.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to include.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to include.")
    parser.add_argument("--exclude-spd-di-datasets", nargs="+", default=["voxceleb", "fairface"], help="Datasets excluded from SPD/DI summaries.")
    parser.add_argument("--exclude-random", action="store_true", help="Kept for README compatibility; no-random files are always exported.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
