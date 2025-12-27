# Quick Start: Custom Datasets Experiments

This is a streamlined guide for running SelfKG experiments on your custom datasets with multiple repetitions.

## TL;DR - Complete Workflow

```bash
# 1. Setup (one time)
bash setup_custom.sh

# 2. Preprocess your custom datasets
conda activate selfkg
python preprocess_custom_datasets.py --device cuda:0

# 3. Run experiments (3 runs per dataset)
python run_custom_experiments.py --num_runs 3 --device cuda:0 --epochs 150

# 4. Calculate mean results
python calculate_mean_results.py \
    --results_dir logs/custom_experiments/<timestamp> \
    --csv --latex --json
```

## Your Custom Dataset Structure

Place your dataset in `data/custom/<dataset_name>/`:

```
data/custom/fr_en/          # Example: French-English alignment
├── ent_ids_1               # KG1 entities: <id>\t<uri>
├── ent_ids_2               # KG2 entities: <id>\t<uri>
├── triples_1               # KG1 triples: <head>\t<relation>\t<tail>
├── triples_2               # KG2 triples: <head>\t<relation>\t<tail>
├── ref_pairs               # Test alignments: <id_kg1>\t<id_kg2>
└── sup_pairs               # Validation alignments (optional)
```

## Setup on GPU Machine

### Option 1: Quick Setup Script

```bash
bash setup_custom.sh
```

### Option 2: Manual Setup

```bash
# Create environment
conda create -n selfkg python=3.8 -y
conda activate selfkg

# Install dependencies
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install faiss-cpu==1.7.1 numpy==1.19.2 pandas==1.0.5 tqdm==4.61.1
pip install transformers==4.8.2 torchtext==0.10.0

# Download LaBSE model
cd data && bash getdata.sh && cd ..
```

## Running Experiments

### For All Datasets in data/custom/

```bash
conda activate selfkg

# Run 3 repetitions per dataset
python run_custom_experiments.py \
    --num_runs 3 \
    --device cuda:0 \
    --epochs 150 \
    --batch_size 64
```

### For a Specific Dataset

```bash
# Preprocess only fr_en
python preprocess_custom_datasets.py --dataset fr_en --device cuda:0

# Run experiments only for fr_en
python run_custom_experiments.py \
    --dataset fr_en \
    --num_runs 3 \
    --device cuda:0 \
    --epochs 150
```

### Background Execution (Recommended)

For long-running experiments:

```bash
# Using nohup
nohup python run_custom_experiments.py \
    --num_runs 3 --device cuda:0 --epochs 150 \
    > experiment.log 2>&1 &

# Monitor progress
tail -f experiment.log

# Or using screen
screen -S selfkg
python run_custom_experiments.py --num_runs 3 --device cuda:0 --epochs 150
# Detach: Ctrl+A then D
# Reattach later: screen -r selfkg
```

## Results

### Directory Structure

```
logs/custom_experiments/20231227_120000/
├── experiment_config.json          # Experiment configuration
├── all_results_summary.json        # All raw results
├── fr_en/
│   ├── run_1_20231227_120030/
│   │   ├── output.log             # Full training log
│   │   └── results.json           # Results for this run
│   ├── run_2_20231227_140030/
│   └── run_3_20231227_160030/
└── ...other datasets.../
```

### Calculate Mean Results

```bash
python calculate_mean_results.py \
    --results_dir logs/custom_experiments/20231227_120000 \
    --csv --latex --json
```

**Output files:**
- `results_summary.txt` - Human-readable summary
- `aggregated_results.json` - Detailed statistics  
- `aggregated_results.csv` - Spreadsheet format
- `results_table.tex` - LaTeX table for thesis

**Example output:**
```
Dataset: fr_en
Number of runs: 3
--------------------------------------------------------------------------------
  Test Hit@1 @ Best Valid     : 0.856 ± 0.003        (n=3)
  Test Hit@10 @ Best Valid    : 0.923 ± 0.002        (n=3)
  Best Valid Hit@1            : 0.842 ± 0.004        (n=3)
  Best Valid Hit@10           : 0.915 ± 0.003        (n=3)
```

## File Checklist

**Created files for custom experiments:**
- ✅ `preprocess_custom_datasets.py` - Preprocessing script
- ✅ `run_custom_experiments.py` - Experiment runner
- ✅ `run_LaBSE_neighbor_custom.py` - Custom dataset training script
- ✅ `calculate_mean_results.py` - Results aggregation
- ✅ `loader/CustomRawNeighbors.py` - Custom data loader
- ✅ `setup_custom.sh` - Setup script
- ✅ `CUSTOM_DATASETS_GUIDE.md` - Detailed documentation
- ✅ `QUICKSTART_CUSTOM.md` - This file

## Troubleshooting

### CUDA Error: "no kernel image is available"

**This is the most common issue!** See **CUDA_FIX_GUIDE.md** for complete solutions.

**Quick fix:**
```bash
# 1. Run diagnostic
python check_cuda_compatibility.py

# 2. Reinstall PyTorch (recommended)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Or use CPU (slower but always works)
python preprocess_custom_datasets.py --device cpu
python run_custom_experiments.py --device cpu --num_runs 3
```

### Out of Memory Error
```bash
# Reduce batch size and queue length
python run_custom_experiments.py \
    --batch_size 32 \
    --queue_length 32 \
    --device cuda:0
```

### Preprocessing Fails
```bash
# Verify LaBSE model exists
ls data/LaBSE/

# Check dataset files
ls data/custom/fr_en/
```

### No Results Found
```bash
# List all results directories
ls -la logs/custom_experiments/

# Find all results.json files
find logs/custom_experiments/ -name "results.json"
```

## Time Estimates

| Task | Time (per dataset) |
|------|-------------------|
| Preprocessing | 5-30 minutes |
| Single run (150 epochs) | 2-8 hours |
| 3 runs total | 6-24 hours |

*Times vary based on dataset size and GPU*

## GPU Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Minimum**: NVIDIA GPU with 4GB VRAM (reduce batch_size to 32)
- **CPU fallback**: Use `--device cpu` (much slower)

## Support

For detailed documentation, see `CUSTOM_DATASETS_GUIDE.md`

For issues with the original SelfKG code, see the [original repository](https://github.com/THUDM/SelfKG)
