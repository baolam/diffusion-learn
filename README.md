# Simple Diffusion 🎨

A minimalist, from-scratch implementation of Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) using PyTorch. 

This repository serves as a clear, educational, and easy-to-understand implementation of diffusion models, primarily focusing on the UNet architecture and noise scheduling processes.

## 🌟 Features

- **Custom UNet Architecture**: An implementation of the UNet model specifically tailored for image generation and noise prediction, featuring DownBlocks, MidBlocks, and UpBlocks with Self-Attention.
- **Linear Noise Scheduler**: Standard DDPM forward (noise addition) and backward (denoising) processes based on linear beta schedules.
- **DDIM Noise Scheduler**: An efficient sampling method that allows for faster image generation with fewer steps compared to traditional DDPM.
- **Unit Tested**: Core scheduling algorithms and the UNet architecture are backed by comprehensive unit tests to ensure shape matching and logical correctness.

## 📂 Project Structure

```text
Simple-Diffusion/
├── src/
│   ├── unet_base.py               # Core UNet implementation (Down/Mid/Up Blocks)
│   ├── linear_noise_scheduler.py  # DDPM Linear Noise Scheduling
│   └── ddim_noise_scheduler.py    # DDIM Noise Scheduling
├── tests/
│   ├── test_noise_schedulers.py   # Unit tests for the Noise Schedulers
│   └── test_unet_base.py          # Unit tests for the UNet model
├── venv/                          # Python Virtual Environment
└── README.md                      # Project Documentation
```

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with PyTorch. 

If you are using the included virtual environment, you can activate it and ensure dependencies are installed.

```bash
# Create and activate a virtual environment (if starting fresh)
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install torch pytest
```

### Running Tests
To verify that the code logic and tensor shapes are working correctly, you can run the test suite using Python's built-in `unittest` module:

```bash
# Using the python executable inside the venv (Windows)
.\venv\python.exe -m unittest discover tests/

# Or if the environment is activated
python -m unittest discover tests/
```

## 📚 Acknowledgments & References

This implementation draws inspiration and structural ideas from the following excellent repositories:
- [milesial/pytorch-unet](https://github.com/milesial/pytorch-unet)
- [explainingai-code/DDPM-Pytorch](https://github.com/explainingai-code/DDPM-Pytorch/blob/main)
