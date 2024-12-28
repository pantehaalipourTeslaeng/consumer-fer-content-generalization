# StyleGAN2 Setup Guide

## Prerequisites
- CUDA 11.0+
- NVIDIA GPU (8GB+ VRAM)
- Python 3.8+

## Installation Steps

### 1. Install CUDA Dependencies
1. Install CUDA Toolkit:
```bash
# For Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
sudo sh cuda_11.0.3_450.51.06_linux.run
```

2. Add to PATH:
```bash
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
```

### 2. Install StyleGAN2 Dependencies
```bash
pip install ninja tensorflow-gan
```

### 3. Clone and Install StyleGAN2-ADA
```bash
git clone https://github.com/NVlabs/stylegan2-ada
cd stylegan2-ada
pip install -e .
```

### 4. Download Pretrained Models
```bash
mkdir -p models/stylegan2/pretrained
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -O models/stylegan2/pretrained/stylegan2-ffhq-256x256.pkl
```

### Troubleshooting
1. **CUDA Not Found**
   - Verify CUDA installation: `nvidia-smi`
   - Check PATH variables

2. **Memory Issues**
   - Reduce batch size
   - Use smaller image resolution

3. **Compilation Errors**
   - Install build tools: `apt install build-essential`
   - Update gcc/g++ 