# Running SelfKG Experiments on Custom Datasets

This guide explains how to set up and run SelfKG experiments on custom datasets across multiple runs to calculate mean results.

## Overview

The workflow consists of three main steps:
1. **Preprocessing**: Convert your custom datasets into the format required by SelfKG
2. **Running Experiments**: Execute the experiments multiple times (e.g., 3 runs) for each dataset
3. **Aggregating Results**: Calculate mean and standard deviation across runs

## Prerequisites

### Required Files

Download and set up the LaBSE model first:
```bash
cd data
bash getdata.sh
```

This will download the LaBSE model to `data/LaBSE/`.

### Python Environment

Install dependencies:
```bash
conda create -n selfkg python=3.8
conda activate selfkg
bash setup.sh
```

Or manually install:
```bash
pip install torch==1.9.0
pip install faiss-cpu==1.7.1
pip install numpy==1.19.2
pip install pandas==1.0.5
pip install tqdm==4.61.1
pip install transformers==4.8.2
pip install torchtext==0.10.0
```

## Step 1: Prepare Your Custom Dataset

Your custom dataset should be placed in `data/custom/<dataset_name>/` with the following files:

```
data/custom/<dataset_name>/
├── ent_ids_1          # Entity IDs for KG1: <id>\t<uri>
├── ent_ids_2          # Entity IDs for KG2: <id>\t<uri>
├── triples_1          # Triples for KG1: <head>\t<relation>\t<tail>
├── triples_2          # Triples for KG2: <head>\t<relation>\t<tail>
├── ref_pairs          # Test entity alignment pairs: <id_kg1>\t<id_kg2>
└── sup_pairs          # Validation entity alignment pairs (optional)
```

**Example** (already provided in `data/custom/fr_en/`):
- `ent_ids_1`: French DBpedia entities
- `ent_ids_2`: English DBpedia entities
- `triples_1`: French knowledge graph triples
- `triples_2`: English knowledge graph triples
- `ref_pairs`: Ground truth entity alignments for testing
- `sup_pairs`: Supervised pairs for validation (if available)

## Step 2: Preprocess Custom Datasets

Run preprocessing to generate embeddings and required files:

```bash
# Preprocess all datasets in data/custom/
python preprocess_custom_datasets.py --device cuda:0

# Or preprocess a specific dataset
python preprocess_custom_datasets.py --dataset fr_en --device cuda:0
```

**What this does:**
- Creates `cleaned_ent_ids_1` and `cleaned_ent_ids_2` from entity URIs
- Generates `test.ref` and `valid.ref` reference files
- Computes LaBSE embeddings: `raw_LaBSE_emb_1.pkl` and `raw_LaBSE_emb_2.pkl`

**Time estimate:** 5-30 minutes per dataset depending on size and GPU

## Step 3: Run Experiments

### Single Dataset, Single Run

```bash
python run_LaBSE_neighbor_custom.py \
    --device cuda:0 \
    --language fr_en \
    --data_dir custom \
    --epoch 150 \
    --batch_size 64
```

### All Datasets, Multiple Runs (Recommended)

```bash
# Run all datasets in data/custom/ with 3 repetitions each
python run_custom_experiments.py \
    --num_runs 3 \
    --device cuda:0 \
    --epochs 150 \
    --batch_size 64 \
    --queue_length 64
```

**Parameters:**
- `--num_runs 3`: Run each dataset 3 times
- `--device cuda:0`: GPU device to use
- `--epochs 150`: Number of training epochs (original paper uses 150)
- `--batch_size 64`: Batch size (adjust based on GPU memory)
- `--queue_length 64`: Queue length for negative samples

**Output:**
Results are saved to `logs/custom_experiments/<timestamp>/`
```
logs/custom_experiments/20231227_120000/
├── experiment_config.json
├── all_results_summary.json
├── fr_en/
│   ├── run_1_<timestamp>/
│   │   ├── output.log
│   │   └── results.json
│   ├── run_2_<timestamp>/
│   └── run_3_<timestamp>/
└── <other_datasets>/
```

**Time estimate:** 2-8 hours per run per dataset (depending on dataset size and GPU)

### Run Specific Dataset Only

```bash
python run_custom_experiments.py \
    --dataset fr_en \
    --num_runs 3 \
    --device cuda:0 \
    --epochs 150
```

## Step 4: Calculate Mean Results

After all experiments complete, aggregate the results:

```bash
python calculate_mean_results.py \
    --results_dir logs/custom_experiments/<timestamp> \
    --csv \
    --latex \
    --json
```

**Parameters:**
- `--results_dir`: Directory containing experiment results
- `--csv`: Generate CSV report
- `--latex`: Generate LaTeX table for paper
- `--json`: Save aggregated JSON

**Output:**
```
logs/custom_experiments/<timestamp>/
├── aggregated_results.json     # Detailed statistics
├── aggregated_results.csv      # CSV format
├── results_table.tex           # LaTeX table
└── results_summary.txt         # Human-readable summary
```

**Example output:**
```
=====================================
AGGREGATED RESULTS ACROSS ALL RUNS
=====================================

Dataset: fr_en
Number of runs: 3
--------------------------------------------------------------------------------
  Test Hit@1 @ Best Valid     : 0.856 ± 0.003        (n=3)
  Test Hit@10 @ Best Valid    : 0.923 ± 0.002        (n=3)
  Best Valid Hit@1            : 0.842 ± 0.004        (n=3)
  Best Valid Hit@10           : 0.915 ± 0.003        (n=3)
```

## Setup for Different GPU Machine

### 1. Transfer Code and Data

```bash
# On local machine: package the code
tar czf selfkg.tar.gz \
    --exclude='logs' \
    --exclude='checkpoints' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    SelfKG/

# Transfer to GPU machine
scp selfkg.tar.gz user@gpu-machine:/path/to/workspace/

# On GPU machine: extract
ssh user@gpu-machine
cd /path/to/workspace/
tar xzf selfkg.tar.gz
cd SelfKG
```

### 2. Set Up Environment

```bash
# Create conda environment
conda create -n selfkg python=3.8
conda activate selfkg

# Install dependencies
bash setup.sh

# Or install manually
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 3. Download LaBSE Model

```bash
cd data
bash getdata.sh
cd ..
```

**Alternative**: If download is slow, download locally and transfer:
```bash
# On local machine
cd data
bash getdata.sh
tar czf LaBSE.tar.gz LaBSE/

# Transfer
scp LaBSE.tar.gz user@gpu-machine:/path/to/SelfKG/data/
ssh user@gpu-machine
cd /path/to/SelfKG/data/
tar xzf LaBSE.tar.gz
```

### 4. Copy Your Custom Datasets

```bash
# On local machine
cd data
tar czf custom_datasets.tar.gz custom/

# Transfer
scp custom_datasets.tar.gz user@gpu-machine:/path/to/SelfKG/data/
ssh user@gpu-machine
cd /path/to/SelfKG/data/
tar xzf custom_datasets.tar.gz
```

### 5. Run Preprocessing on GPU Machine

```bash
# Activate environment
conda activate selfkg

# Preprocess all custom datasets
python preprocess_custom_datasets.py --device cuda:0
```

### 6. Run Experiments

```bash
# Run all experiments with 3 repetitions
nohup python run_custom_experiments.py \
    --num_runs 3 \
    --device cuda:0 \
    --epochs 150 \
    --batch_size 64 \
    > experiment.log 2>&1 &

# Monitor progress
tail -f experiment.log

# Or use screen/tmux for long-running experiments
screen -S selfkg_experiments
python run_custom_experiments.py --num_runs 3 --device cuda:0 --epochs 150
# Detach: Ctrl+A, D
# Reattach: screen -r selfkg_experiments
```

### 7. Download Results

```bash
# On GPU machine: package results
cd /path/to/SelfKG
tar czf results.tar.gz logs/

# On local machine: download
scp user@gpu-machine:/path/to/SelfKG/results.tar.gz .
tar xzf results.tar.gz

# Calculate mean results locally
python calculate_mean_results.py \
    --results_dir logs/custom_experiments/<timestamp> \
    --csv --latex --json
```

## GPU Memory Considerations

If you encounter out-of-memory errors, adjust these parameters:

```bash
# Reduce batch size
python run_custom_experiments.py \
    --batch_size 32 \
    --queue_length 32 \
    --device cuda:0

# Or use a smaller neighbor size by modifying settings.py:
# NEIGHBOR_SIZE = 10  # default is 20
```

## Quick Reference Commands

```bash
# 1. Preprocess all custom datasets
python preprocess_custom_datasets.py --device cuda:0

# 2. Run experiments (3 runs per dataset)
python run_custom_experiments.py --num_runs 3 --device cuda:0 --epochs 150

# 3. Calculate mean results
python calculate_mean_results.py \
    --results_dir logs/custom_experiments/<timestamp> \
    --csv --latex --json

# 4. Single dataset workflow
python preprocess_custom_datasets.py --dataset fr_en --device cuda:0
python run_custom_experiments.py --dataset fr_en --num_runs 3 --device cuda:0
python calculate_mean_results.py --results_dir logs/custom_experiments/<timestamp>
```

## Expected Results Format

After running 3 experiments for each dataset, you'll get:

**Metrics reported:**
- **Test Hit@1 @ Best Valid**: Primary metric - Hit@1 on test set at the epoch with best validation Hit@1
- **Test Hit@10 @ Best Valid**: Hit@10 on test set at the epoch with best validation Hit@1  
- **Best Valid Hit@1**: Best Hit@1 achieved on validation set
- **Best Valid Hit@10**: Best Hit@10 achieved on validation set

Each metric includes:
- Mean across runs
- Standard deviation
- Min/max values
- Number of runs

## Troubleshooting

### Preprocessing fails
```bash
# Check if LaBSE model exists
ls data/LaBSE/

# Check dataset format
head data/custom/fr_en/ent_ids_1
head data/custom/fr_en/triples_1
```

### CUDA out of memory
```bash
# Reduce batch size
python run_custom_experiments.py --batch_size 32 --queue_length 32
```

### Experiments taking too long
```bash
# Reduce epochs for testing
python run_custom_experiments.py --epochs 50 --num_runs 1
```

### Results not found
```bash
# Check log directory structure
ls -R logs/custom_experiments/<timestamp>/

# Verify results.json files exist
find logs/custom_experiments/ -name "results.json"
```

## Citation

If you use this code for your master's thesis, please cite the original SelfKG paper:

```bibtex
@inproceedings{selfkg2022,
  title={SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs},
  author={Xiao, Zequn and Shi, Zijing and Wang, Jiahao and Huang, Qingheng and Liu, Wei},
  booktitle={Proceedings of the Web Conference 2022},
  year={2022}
}
```
