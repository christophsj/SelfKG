#!/bin/bash
# Transfer script for setting up SelfKG on a remote GPU machine
# Usage: bash transfer_to_gpu.sh <user@remote-machine>

if [ -z "$1" ]; then
    echo "Usage: bash transfer_to_gpu.sh <user@remote-machine>"
    echo "Example: bash transfer_to_gpu.sh user@gpu-server.example.com"
    exit 1
fi

REMOTE=$1
REMOTE_DIR="~/SelfKG"

echo "========================================"
echo "Transferring SelfKG to $REMOTE"
echo "========================================"
echo ""

# Create archive excluding unnecessary files
echo "Step 1: Creating archive..."
tar czf selfkg_transfer.tar.gz \
    --exclude='logs' \
    --exclude='checkpoints' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='data/DBP15K' \
    --exclude='data/DWY100K' \
    --exclude='data/LaBSE' \
    *.py *.sh *.md requirements.txt settings.py \
    data/custom/ \
    loader/ \
    model/ \
    script/ \
    data/getdata.sh

echo "Archive created: selfkg_transfer.tar.gz"
echo "Size: $(du -h selfkg_transfer.tar.gz | cut -f1)"

# Transfer to remote machine
echo ""
echo "Step 2: Transferring to $REMOTE..."
scp selfkg_transfer.tar.gz $REMOTE:~/

# Create setup script on remote
echo ""
echo "Step 3: Creating setup script on remote..."
ssh $REMOTE << 'ENDSSH'
mkdir -p ~/SelfKG
cd ~/SelfKG
tar xzf ../selfkg_transfer.tar.gz
chmod +x *.sh

echo ""
echo "========================================"
echo "Transfer complete!"
echo "========================================"
echo ""
echo "Next steps on the GPU machine:"
echo ""
echo "1. SSH to the machine:"
echo "   ssh $HOSTNAME"
echo ""
echo "2. Navigate to the directory:"
echo "   cd ~/SelfKG"
echo ""
echo "3. Run setup:"
echo "   bash setup_custom.sh"
echo ""
echo "4. Follow the instructions in QUICKSTART_CUSTOM.md"
echo ""
ENDSSH

# Clean up local archive
echo ""
read -p "Delete local archive selfkg_transfer.tar.gz? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm selfkg_transfer.tar.gz
    echo "Archive deleted."
fi

echo ""
echo "========================================"
echo "Setup Instructions Summary"
echo "========================================"
echo ""
echo "On the GPU machine, run:"
echo "  cd ~/SelfKG"
echo "  bash setup_custom.sh"
echo ""
echo "Then follow QUICKSTART_CUSTOM.md for running experiments."
echo ""
