# 🤝 Contributing to Brain Tumor Synthesis Pipeline

Thank you for considering contributing to this project! Here's how to get started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Coding Standards](#coding-standards)

---

## 📜 Code of Conduct

This project follows a standard open-source Code of Conduct. Be respectful, inclusive,
and constructive in all interactions.

---

## 🛠️ How to Contribute

There are several ways to contribute:

- 🐛 **Bug fixes** — open an issue first describing the bug
- ✨ **New features** — discuss in an issue before implementing
- 📚 **Documentation** — improve README, docstrings, or inline comments
- 🧪 **Tests** — add unit tests for pipeline stages
- 📊 **Evaluation** — add new metrics or improve radiological scoring

---

## 💻 Development Setup

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:

```bash
git clone https://github.com/your-username/brain-tumor-synthesis.git
cd brain-tumor-synthesis
```

3. **Create a branch** for your changes:

```bash
git checkout -b feature/your-feature-name
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

5. **Make your changes** and test them on Google Colab or locally with GPU.

---

## 📥 Pull Request Process

1. Ensure your code runs end-to-end on a GPU environment
2. Update `README.md` if your change affects usage or architecture
3. Write clear commit messages (e.g., `feat: add 3D volumetric diffusion refinement`)
4. Open a Pull Request against the `main` branch
5. Fill out the PR template:
   - What does this PR do?
   - How was it tested?
   - Are there any breaking changes?

---

## 🐛 Reporting Issues

When reporting a bug, please include:

- Python and PyTorch versions (`python --version`, `python -c "import torch; print(torch.__version__)"`)
- GPU type and CUDA version
- Full error traceback
- Minimal reproducible example (ideally a single notebook cell)

---

## 🧹 Coding Standards

- Follow **PEP 8** for Python style
- Use **type hints** for all function signatures
- Add **docstrings** to all classes and public methods
- Comment tensor shapes inline, e.g.: `# (B, 4, 64, 64, 64) float32 [−1, 1]`
- Use descriptive variable names; avoid single-letter names except for loop indices

---

## 🏗️ Project Structure for Contributors

| Module | Responsibility |
|--------|---------------|
| `MRIPreprocessor` | Stage 1: NIfTI loading, normalization, multi-modal synthesis |
| `TumorSynthesizer` | Stage 2: Physics-based tumor field generation |
| `TumorPINN` | Stage 3: Physics-informed neural network backbone |
| `StyleGAN3D` | Stage 4: GAN-based structural synthesis |
| `TumorDiffusion` | Stage 5: DDPM texture refinement |
| `PipelineOrchestrator` | Agent-based coordinator for all stages |
| `ClinicalEvaluator` | Objective metrics + radiological realism scoring |

---

*Thank you for helping improve synthetic medical imaging AI!*
