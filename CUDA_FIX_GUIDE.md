# CUDA Error Fix Guide

## Problem: "no kernel image is available for execution on the device"

This error means PyTorch was compiled for GPU architectures that don't match your GPU.

## Quick Fix

### Option 1: Reinstall PyTorch (Recommended)

```bash
# Activate your environment
conda activate selfkg

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install newer PyTorch with broader GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Check and Install Correct CUDA Version

First, run the diagnostic:
```bash
python check_cuda_compatibility.py
```

Then install the matching PyTorch version:

```bash
# For CUDA 11.8 (most recent)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 11.7
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.1
pip uninstall torch torchvision torchaudio -y
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 10.2
pip uninstall torch torchvision torchaudio -y
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Option 3: Use CPU (Slower but Always Works)

```bash
# Preprocess with CPU
python preprocess_custom_datasets.py --device cpu

# Run experiments with CPU
python run_custom_experiments.py --device cpu --num_runs 3 --epochs 150
```

**Note:** CPU is much slower (~10-50x) but will work on any machine.

## Detailed Diagnosis Steps

### Step 1: Check Your Setup

```bash
python check_cuda_compatibility.py
```

This will show:
- Your GPU model and compute capability
- PyTorch version and CUDA version
- Whether CUDA operations work

### Step 2: Identify the Issue

The diagnostic script will tell you:
- ✓ If everything is working
- ✗ If there's a CUDA mismatch
- Suggested fix command

### Step 3: Apply the Fix

Follow the suggestions from the diagnostic script.

## Common Scenarios

### Scenario 1: Old GPU (Compute Capability < 3.5)

```bash
# Use CPU mode
python preprocess_custom_datasets.py --device cpu
python run_custom_experiments.py --device cpu
```

### Scenario 2: New GPU (RTX 30XX, 40XX, A100, etc.)

```bash
# Install latest PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Scenario 3: Server with Specific CUDA Version

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch - see table below
```

## PyTorch CUDA Compatibility Table

| CUDA Version | Install Command |
|--------------|-----------------|
| 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| 11.7 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117` |
| 11.6 | `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116` |
| 11.3 | `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113` |
| 10.2 | `pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html` |
| CPU only | `pip install torch torchvision torchaudio` |

## Verification

After fixing, verify it works:

```bash
# Test 1: Check CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test 2: Test GPU operation
python -c "import torch; x = torch.randn(10, 10).cuda(); print('✓ GPU test passed')"

# Test 3: Full diagnostic
python check_cuda_compatibility.py
```

## Still Having Issues?

### Debug Mode

Run with blocking CUDA calls to see exact error location:

```bash
export CUDA_LAUNCH_BLOCKING=1
python preprocess_custom_datasets.py --device cuda:0
```

### Alternative: Use Docker

```bash
# Use NVIDIA's PyTorch container (includes all CUDA libraries)
docker pull nvcr.io/nvidia/pytorch:22.12-py3

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:22.12-py3 \
  bash

# Inside container
cd /workspace
pip install faiss-cpu transformers
python preprocess_custom_datasets.py --device cuda:0
```

### Contact for Help

If you're still stuck, run:
```bash
python check_cuda_compatibility.py > cuda_diagnostic.txt 2>&1
```

And share `cuda_diagnostic.txt` with:
- Your GPU model
- Operating system
- How you installed PyTorch
