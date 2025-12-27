#!/bin/bash
# Quick setup script for running SelfKG experiments on a GPU machine

set -e  # Exit on error

echo "========================================"
echo "SelfKG Custom Dataset Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $PYTHON_VERSION"

# Setup conda environment
echo ""
echo "Step 1: Setting up conda environment..."
read -p "Create new conda environment 'selfkg'? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda create -n selfkg python=3.8 -y
    echo "Environment 'selfkg' created."
fi

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate selfkg

# Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
read -p "Install Python dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if CUDA is available
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo "CUDA version detected: $CUDA_VERSION"
        
        # Install PyTorch with CUDA support
        if [[ "$CUDA_VERSION" == "11."* ]]; then
            pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
        else
            pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
        fi
    else
        echo "CUDA not detected, installing CPU version"
        pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
    fi
    
    # Install other dependencies
    pip install faiss-cpu==1.7.1
    pip install numpy==1.19.2
    pip install pandas==1.0.5
    pip install tqdm==4.61.1
    pip install transformers==4.8.2
    pip install torchtext==0.10.0
    
    echo "Dependencies installed successfully!"
fi

# Check for LaBSE model
echo ""
echo "Step 3: Checking for LaBSE model..."
if [ ! -d "data/LaBSE" ]; then
    echo "LaBSE model not found."
    read -p "Download LaBSE model? (This may take a while) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd data
        bash getdata.sh
        cd ..
        echo "LaBSE model downloaded."
    else
        echo "WARNING: LaBSE model is required. Please download it later by running:"
        echo "  cd data && bash getdata.sh"
    fi
else
    echo "LaBSE model found at data/LaBSE/"
fi

# Check for custom datasets
echo ""
echo "Step 4: Checking for custom datasets..."
if [ -d "data/custom" ]; then
    NUM_DATASETS=$(find data/custom -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Found $NUM_DATASETS custom dataset(s):"
    ls -1 data/custom/
else
    echo "No custom datasets found in data/custom/"
    echo "Please add your datasets to data/custom/<dataset_name>/"
fi

# Test GPU
echo ""
echo "Step 5: Testing GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate selfkg"
echo ""
echo "2. Add your custom datasets to data/custom/<dataset_name>/"
echo "   Required files: ent_ids_1, ent_ids_2, triples_1, triples_2, ref_pairs"
echo ""
echo "3. Preprocess your datasets:"
echo "   python preprocess_custom_datasets.py --device cuda:0"
echo ""
echo "4. Run experiments (3 runs per dataset):"
echo "   python run_custom_experiments.py --num_runs 3 --device cuda:0 --epochs 150"
echo ""
echo "5. Calculate mean results:"
echo "   python calculate_mean_results.py --results_dir logs/custom_experiments/<timestamp> --csv --latex --json"
echo ""
echo "For more details, see CUSTOM_DATASETS_GUIDE.md"
echo ""
