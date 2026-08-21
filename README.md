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

```text
        ┌──────────────┐
        │   Coordinator │
        └───────┬──────┘
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   GPU #1    GPU #2    GPU #3
      │         │         │
      └─────────┼─────────┘
                ▼
          Shared model
```

---

## 📊 Network Status

<div align="center">
  <div style="
    display: inline-block;
    padding: 24px 40px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    color: #f0f6fc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    min-width: 280px;
  ">
    <div style="font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px;">
      CrowdGPT Network
    </div>
    <div id="token-counter" style="font-size: 24px; font-weight: bold; color: #58a6ff;">
      Tokens processed: Loading...
    </div>
  </div>
</div>

<script>
  (function fetchTokens() {
    fetch('res/tokens.txt')
      .then(response => {
        if (!response.ok) throw new Error('Network error');
        return response.text();
      })
      .then(data => {
        const count = data.trim();
        document.getElementById('token-counter').textContent = `Tokens processed: ${count}`;
      })
      .catch(err => {
        // Fallback value if local fetch fails or is restricted
        document.getElementById('token-counter').textContent = 'Tokens processed: 1000';
      });
  })();
</script>

---

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
