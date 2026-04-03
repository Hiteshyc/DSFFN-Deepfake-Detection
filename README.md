<div align="center">

<br/>

```
██████╗ ███████╗███████╗███████╗███╗   ██╗
██╔══██╗██╔════╝██╔════╝██╔════╝████╗  ██║
██║  ██║███████╗█████╗  █████╗  ██╔██╗ ██║
██║  ██║╚════██║██╔══╝  ██╔══╝  ██║╚██╗██║
██████╔╝███████║██║     ██║     ██║ ╚████║
╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚═╝  ╚═══╝
```

# Dual-Stream Forgery Fusion Network

**Robust Cross-Domain Deepfake Detection**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

> *"A deepfake detector is only as good as the forgeries it has never seen."*

<br/>

**Hitesh Chaudhari · Aayush Raja**
Vellore Institute of Technology, Vellore, India

</div>

---

## The Problem

Modern deepfake detectors are brittle. Train them on one dataset, and they collapse the moment they encounter forgeries from a different source — a phenomenon known as **domain shift**.

The root cause: most detectors learn *shallow, dataset-specific shortcuts* (like a particular blending artifact or compression pattern) rather than *fundamental forgery signatures*. When those shortcuts don't appear in unseen data, the model fails.

**DSFFN** directly attacks this problem.

---

## The Solution

<div align="center">

```
┌──────────────────────────────────────────────────┐
│               Input Image (RGB)                  │
└────────────────┬─────────────────┬───────────────┘
                 │                 │
                 ▼                 ▼
     ┌───────────────┐   ┌──────────────────────┐
     │ Spatial Stream│   │  Frequency Transform │
     │  (RGB as-is)  │   │  (Phase via 2D DFT)  │
     └───────┬───────┘   └──────────┬───────────┘
             │                      │
             ▼                      ▼
   ┌──────────────────┐   ┌──────────────────────┐
   │  EfficientNet-B0 │   │   EfficientNet-B0    │
   │ Spatial Backbone │   │  Frequency Backbone  │
   └────────┬─────────┘   └──────────┬───────────┘
            │                        │
            └───────────┬────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   Feature Fusion Layer │
           │  cat(F_spatial,F_freq) │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    Classifier (MLP)    │
           │ Linear→ReLU→Drop→Linear│
           └────────────┬───────────┘
                        │
                        ▼
              Real  /  Fake  (BCE)
```

</div>

The key insight: **spatial features** catch what's visually wrong. **Phase-spectrum features** reveal what the generative process left behind — structural traces that survive domain shifts. Together, they're fundamentally harder to fool than either alone.

---

## Results

### Test 1 — Intra-Dataset (same distribution)

| Model | Accuracy | AUC |
|:------|:--------:|:---:|
| Frequency-Only | 72.46% | 82.27% |
| Spatial-Only | 88.79% | 97.51% |
| **DSFFN (Ours)** | **90.12%** | **98.01%** |

### Test 2 — Cross-Dataset (unseen forgeries — the real test)

| Model | Home Acc | Unseen Acc | Gap |
|:------|:--------:|:----------:|:---:|
| Spatial-Only | 88.79% | 68.14% | −20.65% |
| Frequency-Only | 72.46% | 61.30% | −11.16% |
| **DSFFN (Ours)** | **90.12%** | **74.52%** | **−15.60%** |

> **DSFFN closes the generalization gap by 5.05%** compared to the spatial baseline — a direct, measurable improvement in cross-domain robustness.

---

## Setup

### 1 · Clone & install dependencies

```bash
git clone https://github.com/your-username/DSFFN.git
cd DSFFN
pip install torch torchvision opencv-python scikit-learn tqdm pillow kaggle numpy
```

### 2 · Configure Kaggle API

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3 · Download datasets

```bash
# Training + intra-dataset evaluation
kaggle datasets download -d manjilkarki/deepfake-and-real-images
unzip -o -q deepfake-and-real-images.zip -d ./dataset

# Cross-dataset generalization test (unseen)
kaggle datasets download -d ciplab/real-and-fake-face-detection
unzip -o -q real-and-fake-face-detection.zip -d ./ciplab_dataset
```

Expected structure after unzipping:

```
dataset/
└── Dataset/
    ├── Train/
    │   ├── Real/    ← 70,000 images
    │   └── Fake/    ← 70,000 images
    └── Test/
        ├── Real/    ← 10,000 images
        └── Fake/    ← 10,000 images
```

---

## Running Experiments

The full pipeline runs sequentially in a single script. Steps are clearly labeled.

```bash
python dsffn_experiments.py
```

Or open in Google Colab (recommended — free GPU access):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Results are printed to console and saved automatically to `output.txt`.

---

## How It Works

### Dual-Stream Input Preprocessing

Every image generates **two inputs** before entering the network:

**Spatial input** — standard RGB pipeline:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Frequency input** — phase spectrum extraction:

```python
img_gray  = np.array(img.convert('L'))

# 2D Discrete Fourier Transform
dft       = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Extract phase angle
_, phase  = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

# Normalize to [0,255] and convert to 3-channel for EfficientNet
phase_img = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
freq_input = transform(Image.fromarray(phase_img).convert('RGB'))
```

### Model Architecture

```python
class DSFFN(nn.Module):
    def __init__(self):
        super().__init__()
        # Two EfficientNet-B0 backbones, classifier heads removed
        self.spatial_stream = models.efficientnet_b0(pretrained=True)
        self.spatial_stream.classifier = nn.Identity()

        self.freq_stream = models.efficientnet_b0(pretrained=True)
        self.freq_stream.classifier = nn.Identity()

        # Fusion head: concatenated features → binary classification
        self.classifier_head = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x_spatial, x_freq):
        f_spatial = self.spatial_stream(x_spatial)
        f_freq    = self.freq_stream(x_freq)
        f_fused   = torch.cat((f_spatial, f_freq), dim=1)
        return self.classifier_head(f_fused)
```

### Training Configuration

| Hyperparameter | Value |
|:---------------|:-----:|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 6 |
| Loss | BCEWithLogitsLoss |
| Image size | 224 × 224 |

> Reduce `BATCH_SIZE` to 32 if you hit CUDA OOM errors on smaller GPUs.

---

## Datasets

| Dataset | Kaggle ID | Role | Train | Test |
|:--------|:----------|:-----|------:|-----:|
| Deepfake and Real Images | `manjilkarki/deepfake-and-real-images` | Training + intra-test | 140,000 | 20,000 |
| Real and Fake Face Detection | `ciplab/real-and-fake-face-detection` | Cross-domain eval only | 0 (unseen) | ~2,041 |

---

## Hardware

Experiments were run on **Google Colab** with an **NVIDIA T4 (16GB VRAM)**.

| Component | Spec |
|:----------|:-----|
| GPU | NVIDIA T4 16GB |
| Python | 3.10 |
| PyTorch | 2.1 |
| CUDA | 12.0 |

---

## Why DSFFN Generalizes Better

Two scenarios illustrate the cross-domain improvement:

**Case 1 — Unseen blending artifact**
A fake from the new dataset uses a technique that leaves no visible spatial boundary. The spatial-only model sees nothing wrong and marks it real. DSFFN's frequency stream detects the phase distortion left by the generative process — a signal that survives across domains — and correctly flags it as fake.

**Case 2 — Real image with compression noise**
Heavy compression on a genuine image looks like forgery artifacts to a spatial-only model → false positive. The phase spectrum of an authentic image is naturally coherent, so DSFFN's frequency stream pushes back against the false alarm and reduces misclassification.

---

## Limitations

- **Static frames only** — no temporal modeling; video-level inconsistencies are not exploited
- **Single cross-domain benchmark** — tested against one unseen dataset; broader evaluation needed
- **Concatenation fusion** — learned attention weighting between streams was not explored

---

## Future Work

1. **Temporal extension** — 3D-CNNs or recurrent layers for video-level deepfake detection
2. **Domain-adversarial training** — combine DSFFN's feature extractor with DANN-style alignment to force fused features toward domain invariance
3. **Attention-based fusion** — replace concatenation with cross-attention to adaptively weight each stream per input
4. **Broader benchmarks** — evaluate on FaceForensics++, CelebDF v2, and DFDC

---

## Related Work

| Method | Strategy | Benchmark | Score |
|:-------|:---------|:----------|:-----:|
| Li et al. (2025) | Domain Generalization | Cross-Dataset Avg | 86.43% AUC |
| Yang et al. | Feature Disentanglement | Cross-Dataset | 77.90% AUC |
| Gao et al. (2024) | Texture/Artifact Streams | Custom | 81.44% Acc |
| Prashnani et al. (2025) | Phase-Based Motion | CelebDFv2 | 92.40% AUC |
| **DSFFN (Ours)** | **Spatial + Frequency Fusion** | **Cross-Dataset** | **−5.05% gap** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{chaudhari2024dsffn,
  title       = {DSFFN: Dual-Stream Forgery Fusion Network for Robust Cross-Domain Deepfake Detection},
  author      = {Chaudhari, Hitesh and Raja, Aayush},
  institution = {Vellore Institute of Technology, Vellore, India},
  year        = {2024}
}
```

---

<div align="center">

Made at **Vellore Institute of Technology** &nbsp;·&nbsp; For academic use

</div>
