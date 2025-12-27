#!/usr/bin/env python
"""
Diagnostic script to check CUDA compatibility and suggest fixes.
"""

import sys

print("=" * 80)
print("CUDA Compatibility Diagnostic")
print("=" * 80)

# Check PyTorch
try:
    import torch

    print(f"\n✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch not installed!")
    sys.exit(1)

# Check CUDA availability
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n⚠ CUDA is not available. Possible reasons:")
    print("  1. No NVIDIA GPU detected")
    print("  2. NVIDIA drivers not installed")
    print("  3. PyTorch CPU-only version installed")
    print("\nSolution:")
    print("  Install PyTorch with CUDA support:")
    print("  pip uninstall torch torchvision torchaudio")
    print(
        "  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
    )
    sys.exit(1)

# Get GPU information
print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")

# Check PyTorch CUDA version
print(f"\n✓ PyTorch CUDA version: {torch.version.cuda}")

# Try to get system CUDA version
try:
    import subprocess

    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                print(f"✓ System CUDA version: {line.strip()}")
                break
except:
    print("⚠ Could not detect system CUDA version (nvcc not found)")

# Test GPU operation
print("\n" + "=" * 80)
print("Testing GPU Operations")
print("=" * 80)

try:
    # Test basic tensor operations
    print("\nTest 1: Creating tensor on GPU...")
    x = torch.randn(100, 100).cuda()
    print("✓ Success")

    print("\nTest 2: Matrix multiplication on GPU...")
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print("✓ Success")

    print("\nTest 3: Testing transformers library...")
    try:
        from transformers import AutoTokenizer, AutoModel

        print("✓ Transformers library available")

        # Check if this is the problematic operation
        print("\nTest 4: Loading a simple model on GPU...")
        print("(This may fail if there's a CUDA architecture mismatch)")

        # Try loading a minimal model
        try:
            model = torch.nn.Linear(10, 10).cuda()
            x = torch.randn(1, 10).cuda()
            output = model(x)
            print("✓ Simple model works on GPU")
        except RuntimeError as e:
            print(f"✗ CUDA error with simple model: {e}")
            print("\n" + "=" * 80)
            print("DIAGNOSIS: CUDA Architecture Mismatch")
            print("=" * 80)
            print(
                "\nYour GPU's compute capability is not supported by your PyTorch build."
            )
            print("\nSOLUTION:")
            print("\n1. Check your GPU's compute capability above")
            print("2. Reinstall PyTorch with correct CUDA version:")
            print("\n   # For CUDA 11.1")
            print("   pip uninstall torch torchvision torchaudio")
            print(
                "   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
            )
            print("\n   # For CUDA 10.2")
            print("   pip uninstall torch torchvision torchaudio")
            print(
                "   pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
            )
            print("\n   # For CPU only (slower)")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0")
            print("\n3. Or use newer PyTorch version (recommended):")
            print("   pip uninstall torch torchvision torchaudio")
            print(
                "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
            sys.exit(1)

    except ImportError:
        print("⚠ Transformers library not installed")

    print("\n" + "=" * 80)
    print("✓ All GPU tests passed successfully!")
    print("=" * 80)
    print("\nYour CUDA setup appears to be working correctly.")
    print("If you're still experiencing errors, try:")
    print("  1. Set environment variable: export CUDA_LAUNCH_BLOCKING=1")
    print("  2. Run with CPU: python your_script.py --device cpu")
    print("  3. Update to newer PyTorch: pip install torch --upgrade")

except RuntimeError as e:
    print(f"\n✗ GPU test failed: {e}")
    print("\n" + "=" * 80)
    print("DIAGNOSIS: CUDA Runtime Error")
    print("=" * 80)

    if "no kernel image" in str(e).lower():
        print("\nThe error 'no kernel image is available' means:")
        print("PyTorch was compiled for different GPU architectures than yours.")
        print("\nSOLUTION:")
        print("\n1. Reinstall PyTorch with broader architecture support:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio")
        print("\n2. Or use newer PyTorch (recommended):")
        print(
            "   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118"
        )
        print("\n3. Or run on CPU (slower but works):")
        print("   python preprocess_custom_datasets.py --device cpu")
        print("   python run_custom_experiments.py --device cpu")
    else:
        print("\nGeneral CUDA troubleshooting:")
        print("  1. Update NVIDIA drivers")
        print("  2. Reinstall PyTorch with correct CUDA version")
        print("  3. Try running with CPU: --device cpu")

    sys.exit(1)

except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
