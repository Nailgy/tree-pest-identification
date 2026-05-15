# PyTorch Installation for CUDA 12.x

If you get this error when running `pip install -r requirements.txt`:
```
ERROR: Could not find a version that satisfies the requirement torch==2.5.1+cu124
ERROR: No matching distribution found for torch==2.5.1+cu124
```

**This is normal!** The issue is that PyTorch with CUDA support needs to be installed from a special index.

## Fix: Install PyTorch First (Separate Step)

Run this BEFORE `pip install -r requirements.txt`:

```bash
# For CUDA 12.4 (recommended for RTX 4070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# OR for CUDA 12.1 (if your system uses it)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR for CUDA 12.0 (older systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
```

Then install remaining dependencies:

```bash
# Install remaining packages (without PyTorch, which is already installed)
pip install -r requirements.txt
```

## Complete Installation Steps

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate
                        # Linux: source venv/bin/activate

# 2. Upgrade pip (important!)
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.4 (primary option)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install all other dependencies from requirements.txt
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Verify GPU is Working

After installation, run:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Version: 12.4
```

## If Still Having Issues

### Check NVIDIA Drivers
```bash
nvidia-smi
```

Should show your GPU and CUDA version.

### If CUDA 12.4 doesn't work, try 12.1:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### As Last Resort - CPU version (slower but works):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

This will install CPU-only PyTorch. Training will be very slow but at least you can test the pipeline.

---

## Why This Happens

PyTorch distributes CUDA-enabled wheels differently than regular Python packages. The `+cu124` notation in `requirements.txt` doesn't work with standard pip because:

1. PyTorch wheels are hosted on a custom index
2. Pip needs to be told to look at `https://download.pytorch.org/whl/cu124`
3. Standard pip indexes don't have CUDA versions

That's why we install PyTorch separately first, then the rest of the dependencies.

---

**After this is done, proceed with the training steps in INFO.md!**
