# GPU Setup Guide for LLM Fine-Tuning

This guide covers setting up AMD ROCm and NVIDIA CUDA environments for LLM fine-tuning, including both modern and older GPU architectures.

## Table of Contents

1. [AMD ROCm Setup](#amd-rocm-setup)
   - [Modern GPUs (MI100, MI200, MI300)](#modern-amd-gpus)
   - [Older GPUs (MI50, Vega, Polaris)](#older-amd-gpus-mi50-vega)
2. [NVIDIA CUDA Setup](#nvidia-cuda-setup)
   - [Modern GPUs (A100, H100, RTX 40xx)](#modern-nvidia-gpus)
   - [Older GPUs (Tesla V100, P100, GTX 10xx)](#older-nvidia-gpus)
3. [Python Environment Setup](#python-environment-setup)
4. [Verification Steps](#verification-steps)

---

## AMD ROCm Setup

### Prerequisites

- Ubuntu 22.04 LTS or 24.04 LTS (recommended)
- Kernel 5.15+ (Ubuntu HWE kernel recommended for older GPUs)
- Secure Boot disabled (required for kernel modules)

### Modern AMD GPUs

**Supported GPUs**: MI100 (gfx908), MI200 (gfx90a), MI300 (gfx940/gfx941/gfx942), Radeon VII

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROCm prerequisites
sudo apt install -y wget gnupg2 software-properties-common

# Add ROCm repository (ROCm 6.x)
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo apt install -y ./amdgpu-install_6.1.60103-1_all.deb

# Install ROCm
sudo amdgpu-install -y --usecase=rocm,hip --no-dkms

# Add user to video and render groups
sudo usermod -aG video,render $USER

# Reboot required
sudo reboot
```

After reboot, verify installation:

```bash
rocminfo
rocm-smi
```

### Older AMD GPUs (MI50, Vega)

**Supported GPUs**: MI50 (gfx906), MI60 (gfx906), Radeon VII (gfx906), Vega 56/64 (gfx900)

For older architectures like MI50 (gfx906), you need to set environment variables to override the GFX version:

```bash
# Install ROCm (same as modern)
sudo amdgpu-install -y --usecase=rocm,hip --no-dkms

# Add user to groups
sudo usermod -aG video,render $USER
sudo reboot
```

**Critical Environment Variables for MI50/gfx906:**

Create a file `~/.rocm_env`:

```bash
# ROCm environment for MI50 (gfx906)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3  # Adjust for your GPU count
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_DEVICE_ORDINAL=0,1,2,3

# Prevent hipBLASLt errors on unsupported architecture
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

Add to your `~/.bashrc`:

```bash
echo 'source ~/.rocm_env' >> ~/.bashrc
source ~/.bashrc
```

**Verification for MI50:**

```bash
# Check GFX version
rocminfo | grep -i "gfx"

# Should show something like:
#   Name:                    gfx906
```

### Multi-GPU Setup (AMD)

For multi-GPU training, ensure all GPUs are visible:

```bash
# List all GPUs
rocm-smi --showid

# Check GPU topology
rocm-smi --showtopo

# Set visible devices (example for 4 GPUs)
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
```

---

## NVIDIA CUDA Setup

### Prerequisites

- Ubuntu 22.04 LTS or 24.04 LTS
- GCC 11 or 12
- Linux kernel headers

### Modern NVIDIA GPUs

**Supported GPUs**: A100, H100, L40, RTX 3090/4090, RTX A6000

```bash
# Remove any existing NVIDIA drivers
sudo apt remove --purge nvidia-* -y
sudo apt autoremove -y

# Install prerequisites
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r)

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.x and driver
sudo apt install -y cuda-toolkit-12-4 nvidia-driver-550

# Reboot
sudo reboot
```

After reboot:

```bash
# Verify installation
nvidia-smi
nvcc --version
```

### Older NVIDIA GPUs

**Supported GPUs**: Tesla V100, P100, Tesla K80, GTX 1080 Ti, GTX 1070

Older GPUs require specific CUDA versions:

| GPU | Architecture | Recommended CUDA |
|-----|-------------|------------------|
| V100 | Volta (sm_70) | CUDA 11.8 or 12.x |
| P100 | Pascal (sm_60) | CUDA 11.8 |
| K80 | Kepler (sm_37) | CUDA 11.4 |
| GTX 1080 Ti | Pascal (sm_61) | CUDA 11.8 |

**For Tesla V100/P100:**

```bash
# Install CUDA 11.8 (better compatibility)
sudo apt install -y cuda-toolkit-11-8 nvidia-driver-535

# Set CUDA path
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**For Tesla K80 (Kepler):**

```bash
# K80 requires older CUDA
# Install CUDA 11.4 from archive
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
sudo sh cuda_11.4.4_470.82.01_linux.run --toolkit --silent
```

### Multi-GPU Setup (NVIDIA)

```bash
# List all GPUs
nvidia-smi -L

# Check GPU topology for optimal placement
nvidia-smi topo --matrix

# Set visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Python Environment Setup

### Create Virtual Environment

```bash
# Install Python 3.11 or 3.12
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project directory
mkdir -p ~/llm_training
cd ~/llm_training

# Create virtual environment with uv
uv venv --python 3.11
source .venv/bin/activate
```

### Install PyTorch

**For AMD ROCm:**

```bash
# Install PyTorch for ROCm 6.x
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# For ROCm 5.x (older systems)
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**For NVIDIA CUDA:**

```bash
# Install PyTorch for CUDA 12.x
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install Training Dependencies

```bash
# Install Axolotl and dependencies
uv pip install axolotl[flash-attn,deepspeed]

# Install additional packages
uv pip install transformers datasets accelerate peft bitsandbytes
uv pip install wandb tensorboard  # For logging

# For model conversion
uv pip install llama-cpp-python sentencepiece
```

---

## Verification Steps

### 1. Verify GPU Detection

**AMD ROCm:**

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch version: 2.x.x+rocmX.X
ROCm available: True
GPU count: 4  # Your GPU count
```

**NVIDIA CUDA:**

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Verify Memory Allocation

```bash
python3 << 'EOF'
import torch

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")

    # Test allocation
    tensor = torch.randn(1024, 1024, device=f'cuda:{i}')
    print(f"  Test tensor created successfully")
    del tensor
    torch.cuda.empty_cache()
EOF
```

### 3. Verify Multi-GPU Communication

```bash
python3 << 'EOF'
import torch
import torch.distributed as dist

if torch.cuda.device_count() > 1:
    # Test NCCL backend
    print(f"Testing {torch.cuda.device_count()} GPUs...")

    tensors = []
    for i in range(torch.cuda.device_count()):
        t = torch.randn(100, 100, device=f'cuda:{i}')
        tensors.append(t)

    # Simple GPU-to-GPU copy test
    for i in range(1, len(tensors)):
        tensors[i].copy_(tensors[0])

    print("Multi-GPU communication test: PASSED")
else:
    print("Single GPU detected, skipping multi-GPU test")
EOF
```

### 4. Quick Training Test

```bash
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading small test model...")
model_name = "microsoft/phi-2"  # Small model for testing

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded on: {model.device}")
print(f"Model dtype: {model.dtype}")

# Test inference
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
print(f"Test generation: {tokenizer.decode(outputs[0])}")

print("\nGPU setup verification: PASSED")
EOF
```

---

## Troubleshooting

### AMD ROCm Issues

**Issue**: `hipErrorNoBinaryForGpu` error
```
Solution: Set HSA_OVERRIDE_GFX_VERSION for your architecture
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # For MI50
```

**Issue**: Out of memory errors on MI50
```
Solution: Reduce batch size and enable gradient checkpointing
MI50 has 16GB VRAM, use micro_batch_size: 1-2
```

**Issue**: `hipBLASLt` warnings
```
Solution: Set TORCH_BLAS_PREFER_HIPBLASLT=0
This is expected on gfx906 architecture
```

### NVIDIA CUDA Issues

**Issue**: CUDA version mismatch
```
Solution: Match PyTorch CUDA version to installed CUDA
Check with: nvcc --version
Install matching PyTorch: uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: Out of memory on older GPUs
```
Solution: Use 8-bit quantization with bitsandbytes
uv pip install bitsandbytes
load_in_8bit: true in config
```

---

## Next Steps

Once your GPU environment is verified:
1. Proceed to [02-STEP-BY-STEP-PIPELINE.md](./02-STEP-BY-STEP-PIPELINE.md) for detailed training steps
2. Or use [03-AUTOMATED-PIPELINE.md](./03-AUTOMATED-PIPELINE.md) for single-script execution
