# DSFFN: Dual-Stream Forgery Fusion Network
### Robust Cross-Domain Deepfake Detection

> **Authors:** Hitesh Chaudhari, Aayush Raja — Vellore Institute of Technology, Vellore, India

---

## Overview

The **Dual-Stream Forgery Fusion Network (DSFFN)** is a deepfake detection architecture designed to address the core challenge of *domain shift* — the tendency of detectors to fail when encountering forgeries from sources they weren't trained on.

DSFFN learns complementary forgery cues from two parallel input streams:
- **Spatial (RGB) Stream** — captures visual/textural artifacts and blending inconsistencies
- **Frequency (Phase) Stream** — reveals structural and generative-process anomalies invisible to the naked eye

By fusing both representations, DSFFN achieves significantly better cross-domain generalization than single-stream baselines.

---

## Key Results

### Test 1 — Intra-Dataset Performance (manjilkarki test set)

| Model | Accuracy | AUC |
|---|---|---|
| Spatial-Only | 88.79% | 97.51% |
| Frequency-Only | 72.46% | 82.27% |
| **DSFFN (Fused)** | **90.12%** | **98.01%** |

### Test 2 — Cross-Dataset Generalization (unseen ciplab dataset)

| Model | Intra-Dataset Acc | Cross-Dataset Acc | Generalization Gap |
|---|---|---|---|
| Spatial-Only | 88.79% | 68.14% | −20.65% |
| Frequency-Only | 72.46% | 61.30% | −11.16% |
| **DSFFN (Fused)** | **90.12%** | **74.52%** | **−15.60%** |

**DSFFN reduces the generalization gap by 5.05%** compared to the spatial-only baseline — a concrete demonstration of improved cross-domain robustness.

---

## Architecture

```
Input Image (RGB)
      │
      ├────────────────────────────────────┐
      │                                    │
      ▼                                    ▼
Spatial Stream                    Frequency Transform
(EfficientNet-B0)                 (Phase Spectrum via DFT)
      │                                    │
      ▼                                    ▼
 F_spatial                        Frequency Stream
 (feature vector)                 (EfficientNet-B0)
      │                                    │
      └──────────────┬─────────────────────┘
                     │
                     ▼
            Feature Fusion Layer
            (Concatenate F_spatial + F_freq)
                     │
                     ▼
            Classifier Head (MLP)
            Linear(N*2 → 512) → ReLU → Dropout(0.5) → Linear(512 → 1)
                     │
                     ▼
            Classification Loss (BCE)
```

---

## Datasets

| Dataset | Role | Kaggle ID | Train Images | Test Images |
|---|---|---|---|---|
| Deepfake and Real Images | Training + Intra-Dataset Test | `manjilkarki/deepfake-and-real-images` | 140,000 (70k each class) | 20,000 (10k each class) |
| Real and Fake Face Detection | Cross-Dataset Generalization Test | `ciplab/real-and-fake-face-detection` | 0 (unseen) | ~2,041 (~1k each class) |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (experiments used NVIDIA T4 16GB)
- Kaggle API credentials (`kaggle.json`)

### Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install tqdm
pip install kaggle
pip install numpy pillow
```

### Configure Kaggle API

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Running the Experiments

This project is structured as a single end-to-end Colab notebook. Steps are clearly demarcated and can be run sequentially.

### Step 1 — Download Training Dataset

```bash
kaggle datasets download -d manjilkarki/deepfake-and-real-images
unzip -o -q deepfake-and-real-images.zip -d /content/dataset
```

Expected structure:
```
/content/dataset/Dataset/
├── Train/
│   ├── Real/      ← real training images (.jpg)
│   └── Fake/      ← fake training images (.jpg)
└── Test/
    ├── Real/
    └── Fake/
```

### Step 2 — Train All Models

Three models are trained in sequence for 6 epochs each:

```python
# Spatial-Only baseline
model_spatial = SpatialOnlyModel().to(device)

# Frequency-Only baseline
model_freq = FreqOnlyModel().to(device)

# DSFFN (proposed)
model_fused = DSFFN().to(device)
```

All models use:
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Optimizer:** Adam, lr = 1e-4
- **Loss:** BCEWithLogitsLoss
- **Batch size:** 64
- **Epochs:** 6
- **Image size:** 224×224

### Step 3 — Cross-Dataset Evaluation

Download the unseen generalization dataset:

```bash
kaggle datasets download -d ciplab/real-and-fake-face-detection
unzip -o -q real-and-fake-face-detection.zip -d /content/ciplab_dataset
```

The trained models (no retraining) are evaluated directly on this dataset.

---

## Data Preprocessing

Each image generates **two inputs** for the dual-stream network:

**Spatial Input (RGB)**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Frequency Input (Phase Spectrum)**
```python
# Convert to grayscale
img_gray = np.array(img_pil.convert('L'))

# Compute 2D DFT and shift zero-frequency to center
dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Extract phase (angle)
_, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

# Normalize to [0, 255] and convert to 3-channel image
phase_img = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
phase_pil = Image.fromarray(phase_img).convert('RGB')

# Apply same ImageNet normalization
freq_input = transform(phase_pil)
```

---

## Model Definitions

### SpatialOnlyModel / FreqOnlyModel (Baselines)

```python
class SpatialOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.base_model(x)
```

### DSFFN (Proposed)

```python
class DSFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_stream = models.efficientnet_b0(pretrained=True)
        self.spatial_stream.classifier = nn.Identity()

        self.freq_stream = models.efficientnet_b0(pretrained=True)
        self.freq_stream.classifier = nn.Identity()

        num_ftrs = 1280  # EfficientNet-B0 output features
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x_spatial, x_freq):
        f_spatial = self.spatial_stream(x_spatial)
        f_freq = self.freq_stream(x_freq)
        f_fused = torch.cat((f_spatial, f_freq), dim=1)
        return self.classifier_head(f_fused)
```

---

## Training Algorithm

```
Algorithm: DSFFN Training Procedure
Require: Source domains DS, Learning rate η

1. Initialize θ_streams, θ_classifier
2. for each training iteration:
   a. Sample batch (X, Y_cls) from DS
   b. for each image x in X:
      - Compute phase spectrum x_phase
      - F_spatial ← SpatialStream(x; θ_streams)
      - F_freq    ← FrequencyStream(x_phase; θ_streams)
      - F_fused   ← Concatenate(F_spatial, F_freq)
      - P_cls     ← ClassifierHead(F_fused; θ_classifier)
      - L_cls     ← BCE(P_cls, Y_cls)
   c. L_total ← L_cls
   d. Update (θ_streams, θ_classifier) via gradient descent
3. return trained parameters
```

---

## Output

Results are automatically printed and saved to `output.txt`:

```
--- Test 1: Intra-Dataset Performance ---
  Model             |  Accuracy  |  AUC
  Spatial Only      |   88.79%   |  97.51%
  Frequency Only    |   72.46%   |  82.27%
  DSFFN (Fused)     |   90.12%   |  98.01%

--- Test 2: Cross-Dataset Generalization ---
  Model             | Intra-Dataset | Cross-Dataset | Performance Drop
  Spatial Only      |    88.79%     |    68.14%     |     -20.65%
  Frequency Only    |    72.46%     |    61.30%     |     -11.16%
  DSFFN (Fused)     |    90.12%     |    74.52%     |     -15.60%
```

---

## Hardware Requirements

| Component | Specification |
|---|---|
| GPU | NVIDIA T4 (16GB VRAM) or equivalent |
| Platform | Google Colab (recommended) |
| Python | 3.10 |
| PyTorch | 2.1 |
| CUDA | 12.0 |

> **Note:** Reduce `BATCH_SIZE` from 64 if you encounter CUDA out-of-memory errors.

---

## Limitations

- **Static frames only** — no temporal modeling for video deepfakes
- **Single cross-domain test** — generalization tested against one unseen dataset
- **Simple feature fusion** — concatenation-based; no learned attention weighting between streams

---

## Future Work

1. **Temporal Extension** — incorporate 3D-CNNs or recurrent layers for video-level detection
2. **Domain-Adversarial Training** — combine DSFFN's feature extractor with explicit domain alignment (DANN) to further reduce domain shift
3. **Broader Benchmark Testing** — evaluate against FaceForensics++, CelebDF v2, DFDC, and other standard benchmarks
4. **Attention-Based Fusion** — replace concatenation with cross-attention or learned weighting to adaptively prioritize the more informative stream per input

---

## Citation

If you use this work, please cite:

```bibtex
@article{chaudhari2024dsffn,
  title     = {DSFFN: Dual-Stream Forgery Fusion Network for Robust Cross-Domain Deepfake Detection},
  author    = {Chaudhari, Hitesh and Raja, Aayush},
  institution = {Vellore Institute of Technology},
  year      = {2024}
}
```

---

## License

This project is released for academic and research purposes.
