# 🧠 Physics-Informed Hybrid Synthetic Brain Tumor MRI Generation Pipeline

> A multi-stage deep learning framework that synthesizes clinically realistic brain tumor MRI volumes by combining **Physics-Informed Neural Networks (PINNs)**, **StyleGAN-3D**, and **Diffusion Models** — grounded in biomechanical tumor growth principles.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Medical AI models for brain tumor analysis suffer from **limited, privacy-restricted, and non-diverse MRI datasets**. This project proposes a 6-stage hybrid pipeline that:

1. Starts from a **real healthy brain MRI** (MNI152 template via `nilearn`)
2. Synthesizes a **physics-based tumor** using biomechanical reaction-diffusion modeling
3. Trains a **PINN backbone** to learn spatial tumor physics features
4. Refines tumor structure using a **StyleGAN-inspired 3D GAN**
5. Enhances fine-grained texture using a **DDPM Diffusion U-Net**
6. Blends the generated residual back into healthy anatomy

The result is **anatomically consistent, biologically plausible, multi-modal synthetic brain MRI data** (T1, T1c, T2, FLAIR).

---

## 🏗️ Pipeline Architecture

```
Real MRI (NIfTI)
       │
       ▼
┌──────────────────────┐
│  Stage 1             │  Normalize → Segment → Multi-Modal Synthesis
│  MRI Preprocessor    │  Output: (4, 128³) tensor
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Stage 2             │  Physics-based tumor field:
│  Tumor Synthesizer   │  necrotic core + enhancing ring + edema + mass effect
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Stage 3             │  Coord (x,y,z,t) → 5D Physics Features
│  TumorPINN           │  [concentration, displacement_xyz, pressure]
└──────────┬───────────┘
           │
      ┌────┴────┐
      ▼         ▼
┌──────────┐ ┌───────────────┐
│ Stage 4  │ │   Stage 5     │
│ GAN-3D   │ │ DDPM U-Net    │
│ (64³)    │ │ (32³)         │
└─────┬────┘ └──────┬────────┘
      └──────┬──────┘
             ▼
┌────────────────────────┐
│  Stage 6               │  Upsample → Additive Blend
│  Residual Blending     │  I_final = I_healthy + α·R_tumor
└────────────────────────┘
```

---

## ✨ Key Features

- 🔬 **Physics-informed tumor modeling** using reaction-diffusion PDE: `∂c/∂t = D∇²c + ρc(1−c)`
- 🧬 **Multi-zone tumor morphology**: necrotic core, enhancing ring, peritumoral edema with noisy non-spherical boundaries
- 💡 **StyleGAN-3D with AdaIN** — physics features injected as style vectors via Adaptive Instance Normalization
- 🌊 **DDPM Diffusion with FiLM conditioning** — timestep-aware 3D U-Net with physics bottleneck broadcast
- 🤖 **Agent-based orchestration** — modular PhysicsAgent, TextureAgent, SynthesisAgent, QualityAgent
- 📊 **Dual evaluation** — Objective clinical metrics + Radiological Realism Scoring

---

## 📂 Repository Structure

```
📦 brain-tumor-synthesis/
├── 📓 brain_tumor_pipeline.ipynb   # Main all-in-one pipeline notebook
├── 📄 requirements.txt                       # Python dependencies
├── 📄 CONTRIBUTING.md                        # Contribution guidelines
└── 📄 README.md
```

---

## ⚙️ Requirements

### System
- Python 3.9+
- CUDA-capable GPU (Tesla T4 or equivalent recommended)
- Google Colab (recommended) or local Jupyter environment

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision nibabel nilearn requests scipy scikit-image
```

> ⚠️ GPU is strongly recommended. CPU execution is ~50× slower for 3D convolutions.

---

## 🚀 Quickstart

### ▶️ Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `brain_tumor_pipeline.ipynb`
3. Set runtime: **Runtime → Change runtime type → GPU (T4)**
4. Click **Runtime → Run all**

The notebook auto-downloads the MNI152 brain template, runs all 6 pipeline stages, and outputs evaluation results.

### 💻 Local Jupyter

```bash
git clone https://github.com/your-username/brain-tumor-synthesis.git
cd brain-tumor-synthesis
pip install -r requirements.txt
jupyter notebook brain_tumor_pipeline.ipynb
```

---

## 🧪 Experimental Results

Evaluated on **200 synthetic images** across 4 model variants, assessed on a central tumor slice (z=74).

### Objective Clinical Quality Metrics

| Rank | Method | PSNR (dB) | SSIM | MAE | Lesion MAE | Edge Corr | Boundary MAE | Contrast Delta | Composite |
|:----:|:----------:|:---------:|:------:|:------:|:----------:|:---------:|:------------:|:--------------:|:---------:|
| 🥇 1 | **GAN + PINN** | 32.67 | 0.9679 | 0.0099 | 0.0491 | 0.9823 | 0.0216 | 0.0056 | **0.9710** |
| 🥈 2 | GAN only | 32.62 | 0.9581 | 0.0100 | 0.0463 | 0.9821 | 0.0201 | 0.0145 | 0.9329 |
| 🥉 3 | Diffusion + PINN | 31.90 | 0.8926 | 0.0160 | 0.0433 | 0.9821 | 0.0245 | 0.0182 | 0.6776 |
| 4 | Diffusion only | 29.65 | 0.8853 | 0.0202 | 0.0835 | 0.9794 | 0.0455 | 0.0548 | 0.0000 |

### Radiological Realism Scoring (Weighted)

| Rank | Method | Enh. Ring (30%) | Cross-Modal (25%) | Contrast (20%) | Boundary (15%) | Lesion SSIM (10%) | Weighted Score |
|:----:|:----------:|:---------------:|:-----------------:|:--------------:|:--------------:|:-----------------:|:--------------:|
| 🥇 1 | **GAN only** | 0.7651 | 0.875 | 0.8525 | 0.9610 | 0.9581 | **0.8587** |
| 🥈 2 | GAN + PINN | 0.5864 | 0.875 | 0.9216 | 0.9622 | 0.9679 | 0.8201 |
| 🥉 3 | Diffusion + PINN | 0.5467 | 0.875 | 0.7804 | 0.9680 | 0.8926 | 0.7733 |
| 4 | Diffusion only | 0.3748 | 0.375 | 0.5068 | 0.9648 | 0.8853 | 0.5408 |

> 💡 **GAN + PINN** leads in objective image quality. **GAN only** leads in radiological realism, particularly in Enhancement Ring quality (30% weight).

---

## 🏋️ Training Configuration

| Component | Optimizer | LR | Epochs | Output Shape |
|:---------:|:---------:|:---:|:------:|:------------:|
| TumorPINN | Adam + CosineAnnealingLR | 1e-3 | 800 | (B, 5) point-wise |
| StyleGAN-3D | Adam (β=(0.0, 0.99)) | 2e-4 | 1200 | (1, 4, 64³) |
| DDPM U-Net | Adam | 1e-4 | 1200 | (1, 4, 32³) |

**Key training details:**
- Mixed-precision (AMP) training enabled throughout
- Tumor-weighted loss mask: `W = 0.2 × brain_mask + 0.8 × tumor_mask`
- GAN loss: `L = Σ|G(z,φ) − R_target| · W / ΣW`
- Diffusion schedule: T=50 linear steps, β from `1e-4 → 0.02`
- PINN loss weights: concentration=1.0, displacement=0.5, pressure=0.3

---

## 📐 Model Architecture Details

<details>
<summary><b>Stage 3 — TumorPINN</b></summary>

- **Input** (90-dim): 4D spatiotemporal coordinates (x,y,z,t) with sinusoidal positional encoding (6 freq bands, 52-dim) + tissue embedding (32-dim) + DTI features (6-dim)
- **Backbone**: 4 Residual blocks (Linear → LayerNorm → GELU, 128-dim hidden)
- **Output heads**:
  - Concentration head → scalar ∈ [0,1] via Sigmoid
  - Displacement head → 3D vector ∈ [−0.1, 0.1] via scaled Tanh
  - Pressure head → unbounded scalar
- **Physics loss**: MSE + PDE residual of `∂c/∂t = D∇²c + ρc(1−c)` + boundary conditions

</details>

<details>
<summary><b>Stage 4 — StyleGAN-3D Generator</b></summary>

- **Progressive upsampling**: 4³ → 8³ → 16³ → 32³ → 64³ (channels: 128→64→32→16→16)
- **Style injection**: 64-dim latent `z` concatenated with encoded physics vector → style network → style vector `w` injected via **AdaIN** at every block
- **Each block**: trilinear 2× upsample → 2× Conv3D (3³) + AdaIN + LeakyReLU(0.2) + skip connection
- **Output**: 4 independent heads (T1, T1c, T2, FLAIR) with Tanh activation → shape (1, 4, 64³)
- **Discriminator**: PatchGAN-style 3D, 3 strided conv layers + InstanceNorm

</details>

<details>
<summary><b>Stage 5 — DDPM 3D U-Net</b></summary>

- **Architecture**: 3-level encoder-decoder (channels 4→16→32→64, resolution 32³→16³→8³)
- **Conditioning**: FiLM (Feature-wise Linear Modulation) from sinusoidal timestep embeddings at every U-Net block: `h = h·(1+γ_t) + β_t`
- **Physics conditioning**: 5D PINN features encoded via MLP (5→64→64) and broadcast-added at bottleneck (8³, 64 channels)
- **Skip connections**: channel-wise concatenation from encoder levels
- **Inference**: 50-step DDPM reverse denoising from pure Gaussian noise

</details>

---

## 📊 Evaluation Framework

### Objective Metrics (per modality, central tumor slice)

| Metric | Description | Direction |
|--------|-------------|-----------|
| PSNR | Peak Signal-to-Noise Ratio | ↑ Higher is better |
| SSIM | Structural Similarity Index | ↑ Higher is better |
| MAE | Mean Absolute Error | ↓ Lower is better |
| Lesion MAE | MAE within tumor mask | ↓ Lower is better |
| Edge Correlation | Sobel-based edge map correlation | ↑ Higher is better |
| Boundary MAE | MAE within tumor boundary band | ↓ Lower is better |
| Contrast Delta | Lesion-to-brain contrast difference | ↓ Lower is better |

### Radiological Realism Score

A weighted scoring framework mimicking radiologist assessment:

```
Weighted Score = 0.30 × Enhancement_Ring
               + 0.25 × Cross_Modal_Consistency
               + 0.20 × Contrast_Fidelity
               + 0.15 × Boundary_Sharpness
               + 0.10 × Lesion_SSIM
```

---

## 🔭 Future Work

- [ ] Longitudinal tumor progression synthesis
- [ ] Multi-center scanner adaptation
- [ ] Full 3D volumetric diffusion refinement (128³)
- [ ] Downstream clinical task benchmarking (segmentation, detection)
- [ ] Ablation: GAN only → GAN+Diffusion → PINN+GAN → Full Hybrid

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{deepakr2026tumorgan,
  title     = {Physics-Informed Hybrid Synthetic MRI Generation for Brain Tumor Analysis
               Using GAN and Diffusion Models},
  author    = {Deepak R and Abhijith S D and Malthesh Kathare and
               Gurumurthy Kalyanpur Vishanathaiah and T R Raghu Prasad and Arun K. Desai},
  booktitle = {Conference on Medical Image Analysis},
  year      = {2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [nilearn](https://nilearn.github.io/) for the MNI152 brain template
- [BraTS Challenge](https://www.synapse.org/brats) for benchmark evaluation references
- PyTorch team for the deep learning framework

---

<p align="center">Made with ❤️ for advancing medical imaging AI</p>
