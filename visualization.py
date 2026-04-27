import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tabulate import tabulate


def collect_metrics(experiment_folder: str = "output") -> pd.DataFrame:
    all_metrics = []
    metrics_pattern = os.path.join(experiment_folder, "*", "*", "metrics.json")
    metrics_files = glob.glob(metrics_pattern)
    for metrics_file in sorted(metrics_files):
        path_parts = Path(metrics_file).parts
        if len(path_parts) < 3:
            continue
        image_name = path_parts[-3]
        method = path_parts[-2]
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            for prompt_key, metrics in metrics_data.items():
                row = {
                    "image_name": image_name,
                    "prompt": prompt_key,
                    "method": method,
                    "clip_score": metrics.get("clip_score", None),
                    "lpips_score": metrics.get("lpips_score", None),
                    "time": metrics.get("time", None),
                    "clip_score_original": metrics.get("clip_score_original", None),
                    "lpips_score_original": metrics.get("lpips_score_original", None),
                    "metrics_file": metrics_file,
                }
                all_metrics.append(row)
        except Exception:
            continue
    if not all_metrics:
        return pd.DataFrame()
    return pd.DataFrame(all_metrics)


def make_figure(df: pd.DataFrame, title: Optional[str] = None):
    sns.set(style="whitegrid")
    method_map = {
        "hardflow": "HardFlow",
        "gradient_guidance": "Gradient Guidance",
        "projection_relaxed": "Projection-Relaxed + Gradient Guidance",
        "oc_flow": "OC-Flow",
    }
    if not df.empty and "method" in df.columns:
        df = df.copy()
        df["method"] = df["method"].replace(method_map)
    if not df.empty and "prompt" in df.columns:
        df = df.copy()
        df["prompt_display"] = df["prompt"].astype(str).str.capitalize()
        prompts = [p.capitalize() for p in sorted(df["prompt"].astype(str).unique())]
    else:
        prompts = []
    methods = df["method"].unique() if not df.empty and "method" in df.columns else []
    num_methods = len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    show_fliers_flag = False

    if num_methods <= 1:
        sns.boxplot(
            data=df,
            x="prompt_display",
            y="clip_score",
            order=prompts,
            ax=axes[0],
            showfliers=show_fliers_flag,
        )
    else:
        sns.boxplot(
            data=df,
            x="prompt_display",
            y="clip_score",
            hue="method",
            order=prompts,
            ax=axes[0],
            showfliers=show_fliers_flag,
        )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CLIP")
    axes[0].tick_params(axis="x", rotation=0)
    if num_methods <= 1:
        sns.boxplot(
            data=df,
            x="prompt_display",
            y="lpips_score",
            order=prompts,
            ax=axes[1],
            showfliers=show_fliers_flag,
        )
    else:
        sns.boxplot(
            data=df,
            x="prompt_display",
            y="lpips_score",
            hue="method",
            order=prompts,
            ax=axes[1],
            showfliers=show_fliers_flag,
        )
    axes[1].axhline(
        y=0.06,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="LPIPS upper bound",
        zorder=2,
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("LPIPS")
    axes[1].tick_params(axis="x", rotation=0)
    if axes[0].legend_:
        axes[0].legend_.remove()
    if axes[1].legend_:
        axes[1].legend_.remove()
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handle_map = {}
    for h, l in list(zip(handles0, labels0)) + list(zip(handles1, labels1)):
        if l:
            handle_map[l] = h
    row1_labels = [
        "OC-Flow",
        "Projection-Relaxed + Gradient Guidance",
    ]
    row2_labels = [
        "HardFlow (ours)",
        "Gradient Guidance",
        "LPIPS upper bound",
    ]
    row1_handles = [handle_map.get(l, Line2D([0], [0])) for l in row1_labels]
    row2_handles = [handle_map.get(l, Line2D([0], [0])) for l in row2_labels]
    leg1 = fig.legend(
        row1_handles,
        row1_labels,
        loc="lower left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.0, 0.0),
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.2,
    )
    leg2 = fig.legend(
        row2_handles,
        row2_labels,
        loc="lower left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.0, 0.0),
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.2,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    b1 = leg1.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )
    b2 = leg2.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )
    width = max(b1.width, b2.width)
    left = 0.5 - width / 2.0
    gap = 0.002
    y2 = 0.018
    y1 = y2 + b2.height + gap
    leg1.set_bbox_to_anchor((left, y1))
    leg2.set_bbox_to_anchor((left, y2))
    if title:
        pass
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Create CLIP/LPIPS visualization figure"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="output",
        help="Root folder containing experiment outputs (default: output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/visualization.png",
        help="Output image file path (default: visualization.png)",
    )
    parser.add_argument(
        "--result_table",
        type=str,
        default="result/result_table.txt",
        help="Output text file path for time summary table (default: time_table.txt)",
    )
    parser.add_argument("--title", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    df = collect_metrics(args.experiment_folder)
    if df.empty:
        print("No metrics found to visualize.")
        return
    fig, _ = make_figure(df, title=None)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    if not df.empty:
        method_map = {
            "hardflow": "HardFlow",
            "gradient_guidance": "Gradient Guidance",
            "projection_relaxed": "Projection-Relaxed + Gradient Guidance",
            "oc_flow": "OC-Flow",
        }
        df_tbl = df.copy()
        df_tbl["method"] = df_tbl["method"].replace(method_map)
        per_prompt = (
            df_tbl.groupby(["method", "prompt"])
            .agg(
                clip_mean=("clip_score", "mean"),
                lpips_mean=("lpips_score", "mean"),
                time_mean=("time", "mean"),
                safety_rate=("lpips_score", lambda x: (x <= 0.06).mean()),
            )
            .reset_index()
        )
        summary = (
            per_prompt.groupby("method")
            .agg(
                clip_mean=("clip_mean", "mean"),
                clip_std=("clip_mean", "std"),
                lpips_mean=("lpips_mean", "mean"),
                lpips_std=("lpips_mean", "std"),
                time_mean=("time_mean", "mean"),
                time_std=("time_mean", "std"),
                safety_rate=("safety_rate", "mean"),
            )
            .reset_index()
        )
        order = [
            "OC-Flow",
            "Gradient Guidance",
            "Projection-Relaxed + Gradient Guidance",
            "HardFlow",
        ]
        summary["order"] = summary["method"].apply(
            lambda m: order.index(m) if m in order else len(order)
        )
        summary = (
            summary.sort_values(["order"])
            .drop(columns=["order"])
            .reset_index(drop=True)
        )
        table_rows = []
        for _, row in summary.iterrows():
            safety_str = (
                f"{row['safety_rate']:.3f}"
                if not pd.isna(row["safety_rate"])
                else "N/A"
            )
            lpips_str = (
                f"{row['lpips_mean']:.3f} ± {row['lpips_std']:.3f}"
                if not pd.isna(row["lpips_mean"])
                else "N/A"
            )
            clip_str = (
                f"{row['clip_mean']:.3f} ± {row['clip_std']:.3f}"
                if not pd.isna(row["clip_mean"])
                else "N/A"
            )
            time_str = "N/A"
            if not pd.isna(row["time_mean"]):
                time_str = f"{row['time_mean']:.3f} ± {row['time_std']:.3f}"
            table_rows.append(
                [row["method"], safety_str, lpips_str, clip_str, time_str]
            )
        headers = ["Method", "Safety Rate", "LPIPS", "CLIP", "Time (s)"]
        text = tabulate(
            table_rows,
            headers=headers,
            tablefmt="github",
            stralign="left",
            numalign="right",
        )
        print("\nMetrics Summary (mean ± std across prompts):\n" + text)
        tpath = Path(args.result_table)
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with open(tpath, "w") as f:
            f.write(text + "\n")
        print(f"Saved time table to: {tpath}")


if __name__ == "__main__":
    main()
