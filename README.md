<p align="center">
  <a href="https://crowdgpt.net">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo-light.png">
      <img src="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo-light.png" alt="CrowdGPT" width="100%" style="max-width: 800px; border-radius: 12px;">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://crowdgpt.net">Website</a> ·
  <a href="./docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="./docs/DATA.md">Training Data</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Beta-orange?style=for-the-badge">
  <img src="https://img.shields.io/endpoint?style=for-the-badge&url=https%3A%2F%2Fraw.githubusercontent.com%2FVxtzq%2FCrowdGPT%2Fmain%2Fres%2Ftokens.json" alt="Tokens Processed">
</p>

<p align="center">
  <strong>Train AI together. No datacenter required.</strong>
</p>

---

## What is CrowdGPT?

CrowdGPT is a **distributed framework for training LLMs** using volunteered compute.

Instead of one datacenter doing all the work, CrowdGPT distributes training across participating machines.

**No accounts. No API keys. No centralized GPU cluster.**

### How it works

See [ARCHITECTURE.md](https://github.com/Vxtzq/CrowdGPT/blob/main/docs/ARCHITECTURE.md)

## ⚡ Quick Start

```bash
git clone https://github.com/Vxtzq/CrowdGPT.git
cd CrowdGPT
```

Install the dependencies for your hardware:

```bash
# NVIDIA
pip install -r requirements_cuda.txt

# AMD
pip install -r requirements_rocm.txt

# Windows / DirectML
pip install -r requirements_directml.txt

# CPU / Apple Silicon
pip install -r requirements.txt
```

Then:

```bash
python crowdgpt.py
```

For the automatic installer, see [`docs/easy_install.md`](./docs/easy_install.md).

---

## ⚙️ Configuration

```bash
python crowdgpt.py \
  --server http://your-coordinator:3000 \
  --mode deep \
  --batch-size 8 \
  --precision bf16
```

| Option         | Values                                                |
| -------------- | ----------------------------------------------------- |
| `--mode`       | `quick` · `balanced` · `deep` · `ultra`               |
| `--batch-size` | `1` → `512`                                           |
| `--seq-len`    | `8` → `64`                                            |
| `--precision`  | `fp32` · `bf16` · `fp16`                              |
| `--backend`    | `auto` · `cuda` · `rocm` · `directml` · `mps` · `cpu` |
| `--single`     | Run one cycle                                         |

---

## 🧠 Contribute

Training data is curated through pull requests.

See [`docs/DATA.md`](./docs/DATA.md) to contribute.

---

<p align="center">
  <a href="https://crowdgpt.net"><strong>crowdgpt.net</strong></a>
  ·
  <a href="./docs/ARCHITECTURE.md">Architecture</a>
  ·
  <a href="./docs/DATA.md">Data</a>
</p>
