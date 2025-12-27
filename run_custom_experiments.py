#!/usr/bin/env python
# coding: UTF-8
"""
Run SelfKG experiments on custom datasets with multiple repetitions.
This script runs experiments for all datasets in data/custom/ with N repetitions.
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from os.path import join
from settings import DATA_DIR, PROJ_DIR


def get_custom_datasets():
    """Get list of all custom dataset folders."""
    custom_dir = join(DATA_DIR, "custom")

    if not os.path.exists(custom_dir):
        print(f"Custom data directory {custom_dir} does not exist!")
        return []

    datasets = [d for d in os.listdir(custom_dir) if os.path.isdir(join(custom_dir, d))]

    return sorted(datasets)


def check_preprocessing(dataset_name):
    """Check if dataset has been preprocessed."""
    dataset_path = join(DATA_DIR, "custom", dataset_name)

    required_files = [
        "cleaned_ent_ids_1",
        "cleaned_ent_ids_2",
        "raw_LaBSE_emb_1.pkl",
        "raw_LaBSE_emb_2.pkl",
        "test.ref",
        "valid.ref",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(join(dataset_path, file)):
            missing_files.append(file)

    return len(missing_files) == 0, missing_files


def run_single_experiment(
    dataset_name, run_id, device, epochs, batch_size, queue_length, log_dir
):
    """
    Run a single experiment.

    Args:
        dataset_name: Name of the dataset
        run_id: Run number (1, 2, 3, ...)
        device: CUDA device to use
        epochs: Number of epochs
        batch_size: Batch size
        queue_length: Queue length for negative samples
        log_dir: Directory to save logs

    Returns:
        dict: Results containing hit@1 and hit@10 scores
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {dataset_name} - Run {run_id}")
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*80}\n")

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = join(log_dir, dataset_name, f"run_{run_id}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)

    # Prepare command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "run_LaBSE_neighbor_custom.py",
        "--device",
        device,
        "--language",
        dataset_name,
        "--data_dir",
        "custom",
        "--epoch",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--queue_length",
        str(queue_length),
        "--log_dir",
        exp_log_dir,
        "--run_id",
        str(run_id),
    ]

    # Run experiment
    log_file = join(exp_log_dir, "output.log")

    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Print and save output in real-time
            for line in process.stdout:
                print(line, end="")
                f.write(line)

            process.wait()

            if process.returncode != 0:
                print(f"ERROR: Experiment failed with return code {process.returncode}")
                return None

        # Parse results from log file
        results = parse_results_from_log(log_file)

        # Save results to JSON
        results_file = join(exp_log_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_file}")
        return results

    except Exception as e:
        print(f"ERROR running experiment: {e}")
        return None


def parse_results_from_log(log_file):
    """Parse experimental results from log file."""
    results = {
        "valid_hit1": None,
        "valid_hit10": None,
        "test_hit1": None,
        "test_hit10": None,
        "best_valid_hit1": None,
        "best_valid_hit10": None,
        "test_at_best_valid_hit1": None,
        "test_at_best_valid_hit10": None,
    }

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Look for final results in the log
        for i, line in enumerate(lines):
            # Pattern: "Test @ Best Valid = X(Y) at epoch Z batch W"
            if "Test @ Best Valid" in line:
                parts = line.split("=")[1].split("at")[0].strip()
                if "(" in parts:
                    hit1, hit10 = parts.replace(")", "").split("(")
                    results["test_at_best_valid_hit1"] = float(hit1)
                    results["test_at_best_valid_hit10"] = float(hit10)

            # Pattern: "Best Valid Hit@1  = X(Y) at epoch Z"
            if "Best Valid Hit@1" in line and "=" in line:
                parts = line.split("=")[1].split("at")[0].strip()
                if "(" in parts:
                    hit1, hit10 = parts.replace(")", "").split("(")
                    results["best_valid_hit1"] = float(hit1)

            # Pattern: "Best Valid Hit@10 = X(Y) at epoch Z"
            if "Best Valid Hit@10" in line and "=" in line:
                parts = line.split("=")[1].split("at")[0].strip()
                if "(" in parts:
                    hit10, hit1 = parts.replace(")", "").split("(")
                    results["best_valid_hit10"] = float(hit10)

    except Exception as e:
        print(f"Warning: Could not parse results from log: {e}")

    return results


def run_experiments_for_dataset(
    dataset_name, num_runs, device, epochs, batch_size, queue_length, log_dir
):
    """
    Run multiple experiments for a single dataset.

    Returns:
        list: Results from all runs
    """
    print(f"\n{'#'*80}")
    print(f"# Starting experiments for dataset: {dataset_name}")
    print(f"# Number of runs: {num_runs}")
    print(f"{'#'*80}\n")

    # Check preprocessing
    is_ready, missing_files = check_preprocessing(dataset_name)
    if not is_ready:
        print(f"ERROR: Dataset {dataset_name} is not preprocessed!")
        print(f"Missing files: {missing_files}")
        print(
            f"Please run: python preprocess_custom_datasets.py --dataset {dataset_name}"
        )
        return []

    all_results = []

    for run_id in range(1, num_runs + 1):
        results = run_single_experiment(
            dataset_name, run_id, device, epochs, batch_size, queue_length, log_dir
        )
        if results:
            all_results.append(results)
        else:
            print(f"WARNING: Run {run_id} failed!")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run SelfKG experiments on custom datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to run (e.g., fr_en). If not specified, runs all.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of repetitions for each dataset (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs (default: 150)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--queue_length",
        type=int,
        default=64,
        help="Queue length for negative samples (default: 64)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save logs (default: logs/custom_experiments/TIMESTAMP)",
    )

    args = parser.parse_args()

    # Set up log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = join(PROJ_DIR, "logs", "custom_experiments", timestamp)

    os.makedirs(args.log_dir, exist_ok=True)

    # Get datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = get_custom_datasets()

    if not datasets:
        print("No datasets found to process!")
        return

    print(f"Found {len(datasets)} dataset(s) to process: {datasets}")
    print(f"Each dataset will be run {args.num_runs} time(s)")
    print(f"Logs will be saved to: {args.log_dir}")

    # Save experiment configuration
    config = {
        "datasets": datasets,
        "num_runs": args.num_runs,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "queue_length": args.queue_length,
        "timestamp": datetime.now().isoformat(),
    }

    config_file = join(args.log_dir, "experiment_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments for each dataset
    all_dataset_results = {}

    for dataset_name in datasets:
        results = run_experiments_for_dataset(
            dataset_name,
            args.num_runs,
            args.device,
            args.epochs,
            args.batch_size,
            args.queue_length,
            args.log_dir,
        )
        all_dataset_results[dataset_name] = results

    # Save summary of all results
    summary_file = join(args.log_dir, "all_results_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_dataset_results, f, indent=2)

    print(f"\n{'#'*80}")
    print(f"# All experiments completed!")
    print(f"# Results saved to: {args.log_dir}")
    print(f"# Summary file: {summary_file}")
    print(f"{'#'*80}\n")

    print("\nNext step: Calculate mean results using:")
    print(f"  python calculate_mean_results.py --results_dir {args.log_dir}")


if __name__ == "__main__":
    main()
