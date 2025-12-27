#!/usr/bin/env python
# coding: UTF-8
"""
Calculate mean results across multiple experiment runs.
This script aggregates results from multiple runs and computes mean and standard deviation.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from os.path import join
from collections import defaultdict


def load_results_from_directory(results_dir):
    """
    Load all results.json files from the results directory.

    Returns:
        dict: Nested dictionary {dataset_name: {run_id: results}}
    """
    all_results = defaultdict(dict)

    # Walk through the directory structure
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            # Extract dataset name and run ID from path
            # Expected structure: results_dir/dataset_name/run_X_timestamp/results.json
            parts = root.replace(results_dir, "").strip(os.sep).split(os.sep)

            if len(parts) >= 2:
                dataset_name = parts[0]
                run_folder = parts[1]

                # Extract run ID from folder name (e.g., run_1_20231227_120000)
                if run_folder.startswith("run_"):
                    try:
                        run_id = int(run_folder.split("_")[1])
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse run ID from {run_folder}")
                        continue

                    # Load results
                    results_file = join(root, "results.json")
                    try:
                        with open(results_file, "r") as f:
                            results = json.load(f)
                            all_results[dataset_name][run_id] = results
                    except Exception as e:
                        print(f"Warning: Could not load {results_file}: {e}")

    return dict(all_results)


def calculate_statistics(values):
    """
    Calculate mean and standard deviation for a list of values.

    Args:
        values: List of numeric values

    Returns:
        dict: {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    values = [v for v in values if v is not None]

    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values),
    }


def aggregate_results(all_results):
    """
    Aggregate results across runs for each dataset.

    Args:
        all_results: Nested dict {dataset_name: {run_id: results}}

    Returns:
        dict: Aggregated statistics for each dataset
    """
    aggregated = {}

    for dataset_name, runs in all_results.items():
        # Collect values for each metric across runs
        metrics = defaultdict(list)

        for run_id, results in runs.items():
            for metric, value in results.items():
                if value is not None:
                    metrics[metric].append(value)

        # Calculate statistics for each metric
        dataset_stats = {}
        for metric, values in metrics.items():
            dataset_stats[metric] = calculate_statistics(values)

        dataset_stats["num_runs"] = len(runs)
        aggregated[dataset_name] = dataset_stats

    return aggregated


def format_metric(stats, precision=3):
    """Format a metric with mean ± std."""
    if stats["mean"] is None:
        return "N/A"

    mean = stats["mean"]
    std = stats["std"]

    if std > 0:
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    else:
        return f"{mean:.{precision}f}"


def print_results_table(aggregated_results):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("AGGREGATED RESULTS ACROSS ALL RUNS")
    print("=" * 100 + "\n")

    # Define metrics to display
    metrics_to_display = [
        ("test_at_best_valid_hit1", "Test Hit@1 @ Best Valid"),
        ("test_at_best_valid_hit10", "Test Hit@10 @ Best Valid"),
        ("best_valid_hit1", "Best Valid Hit@1"),
        ("best_valid_hit10", "Best Valid Hit@10"),
    ]

    for dataset_name, stats in sorted(aggregated_results.items()):
        print(f"\nDataset: {dataset_name}")
        print(f"Number of runs: {stats['num_runs']}")
        print("-" * 80)

        for metric_key, metric_label in metrics_to_display:
            if metric_key in stats:
                metric_stats = stats[metric_key]
                formatted = format_metric(metric_stats)
                count = metric_stats["count"]
                print(f"  {metric_label:30s}: {formatted:20s} (n={count})")

        print()

    print("=" * 100 + "\n")


def create_csv_report(aggregated_results, output_file):
    """Create a CSV report of the results."""
    rows = []

    for dataset_name, stats in sorted(aggregated_results.items()):
        row = {"dataset": dataset_name, "num_runs": stats["num_runs"]}

        # Add all metrics
        for metric_key, metric_stats in stats.items():
            if metric_key != "num_runs" and isinstance(metric_stats, dict):
                row[f"{metric_key}_mean"] = metric_stats.get("mean")
                row[f"{metric_key}_std"] = metric_stats.get("std")
                row[f"{metric_key}_min"] = metric_stats.get("min")
                row[f"{metric_key}_max"] = metric_stats.get("max")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"CSV report saved to: {output_file}")


def create_latex_table(aggregated_results, output_file):
    """Create a LaTeX table of the results."""
    with open(output_file, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experimental Results on Custom Datasets}\n")
        f.write("\\label{tab:custom_results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write(
            "Dataset & Runs & Valid Hit@1 & Valid Hit@10 & Test Hit@1 @ Best Valid \\\\\n"
        )
        f.write("\\hline\n")

        for dataset_name, stats in sorted(aggregated_results.items()):
            num_runs = stats["num_runs"]

            valid_hit1 = format_metric(stats.get("best_valid_hit1", {}), precision=3)
            valid_hit10 = format_metric(stats.get("best_valid_hit10", {}), precision=3)
            test_hit1 = format_metric(
                stats.get("test_at_best_valid_hit1", {}), precision=3
            )

            f.write(
                f"{dataset_name} & {num_runs} & {valid_hit1} & {valid_hit10} & {test_hit1} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean results from multiple experiment runs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files (default: same as results_dir)",
    )
    parser.add_argument("--csv", action="store_true", help="Generate CSV report")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument(
        "--json", action="store_true", help="Save aggregated results as JSON"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"ERROR: Results directory {args.results_dir} does not exist!")
        return

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")

    # Load all results
    all_results = load_results_from_directory(args.results_dir)

    if not all_results:
        print("No results found!")
        return

    print(f"Found results for {len(all_results)} dataset(s)")
    for dataset, runs in all_results.items():
        print(f"  - {dataset}: {len(runs)} run(s)")

    # Aggregate results
    print("\nAggregating results...")
    aggregated_results = aggregate_results(all_results)

    # Print results table
    print_results_table(aggregated_results)

    # Save JSON if requested
    if args.json:
        json_file = join(args.output_dir, "aggregated_results.json")
        with open(json_file, "w") as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"JSON results saved to: {json_file}")

    # Create CSV report if requested
    if args.csv:
        csv_file = join(args.output_dir, "aggregated_results.csv")
        create_csv_report(aggregated_results, csv_file)

    # Create LaTeX table if requested
    if args.latex:
        latex_file = join(args.output_dir, "results_table.tex")
        create_latex_table(aggregated_results, latex_file)

    # Always create a summary text file
    summary_file = join(args.output_dir, "results_summary.txt")
    with open(summary_file, "w") as f:
        import sys

        old_stdout = sys.stdout
        sys.stdout = f
        print_results_table(aggregated_results)
        sys.stdout = old_stdout
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
