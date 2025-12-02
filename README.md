# Real-Time Video Style Transfer (Edge-Optimized)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

> A lightweight, PyTorch-based implementation of Fast Neural Style Transfer, engineered to perform real-time artistic stylization on video streams and webcam feeds using resource-constrained hardware.


---

## Project Overview

This project implements a feed-forward Convolutional Neural Network (CNN) capable of applying artistic styles to images and video with low latency. Based on Johnson et al. (2016), the model separates training and inference so stylization runs in real time (suitable for video/webcam).

Key ideas:
- Feed-forward TransformerNet (encoder → residual blocks → decoder)
- Instance Normalization for improved stylization
- Optimized for resource-constrained hardware (CPU / MPS / CUDA)

---

## Key Features

- ⚡ Real-time inference (30+ FPS on capable hardware)
- 🏗 Encoder → Bottleneck (residual blocks) → Decoder architecture
- 🎨 Instance Normalization for better style transfer convergence
- 💻 Hardware agnostic: CUDA (NVIDIA), MPS (Apple Silicon), CPU

---

## Engineering Constraints & Optimization Strategy

This implementation is a proof-of-concept tailored for machines with limited VRAM and memory. Engineering trade-offs:

| Constraint | Strategy |
|---|---|
| VRAM limits | Batch size restricted to 2 to avoid OOM during training |
| Memory bandwidth | 256×256 training resolution to keep tensor footprint small |
| Storage / Data | Curated dataset subset instead of large COCO dataset |
| Compute time | 2 training epochs for quick validation (extend for quality) |

Result: The model generalizes style textures and preserves content structure under these constraints.

---

## Technical Architecture

1. Generator (TransformerNet)
   - Downsampling via strided conv layers
   - Bottleneck: 5 Residual Blocks
   - Upsampling with `Upsample` + `Conv2d` to avoid checkerboard artifacts
   - Instance Normalization throughout

2. Loss Network (VGG16)
   - Pretrained VGG16 (frozen) used for perceptual loss
   - Content loss computed at `relu2_2`
   - Style loss computed via Gram matrices at `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`

---

## Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/RealTimeTransfer.git
cd RealTimeTransfer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare data
- Content images: put JPG/PNG in `data/content/`
- Style images: put references in `data/style/`

---

## Usage

Training:
```bash
python train.py \
  --content-dir ./data/content \
  --style-image ./data/style/style.jpg \
  --batch-size 2 \
  --epochs 2 \
  --save-model-dir ./checkpoints
```
Note: Script auto-detects MPS on Apple Silicon.

Webcam inference:
```bash
python inference.py --model checkpoints/model.pth --webcam --image-size 480
```

Process a video:
```bash
python inference.py \
  --model checkpoints/model.pth \
  --input my_video.mp4 \
  --output stylized_video.mp4
```

---

## Dataset Credits

Content images sourced from Unsplash collections (Kaggle). Verify licenses before redistribution.

---

## Roadmap

- [ ] Increase batch size for stable training
- [ ] Train on full COCO for robust generalization
- [ ] Add temporal consistency (optical flow) to reduce flicker
- [ ] Export to ONNX for mobile/web deployment

---

If you want, I can:
- add the demo GIF to `assets/` and update the path,
- or open a PR with this README replacement.

