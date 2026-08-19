<p align="center">
  <a href="https://crowdgpt.ai">
    <img src="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo.png" alt="CrowdGPT" width="100%" style="max-width: 800px; border-radius: 12px;">
  </a>
</p>

<p align="center">
  <a href="https://crowdgpt.ai"><img src="https://img.shields.io/badge/Website-crowdgpt.ai-00E5FF?style=for-the-badge&logo=vercel&logoColor=black" alt="Website"></a>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Beta-orange?style=for-the-badge" alt="Status">
</p>

<h3 align="center">True Decentralized, Permissionless LLM Training.</h3>
<p align="center">
  <a href="https://crowdgpt.ai"><strong>crowdgpt.ai</strong> (Coming Soon 🚀)</a> · <a href="./docs/ARCHITECTURE.md">Architecture</a> · <a href="./docs/DATA.md">Contribute Data</a>
</p>

---

## 🤔 What is CrowdGPT?

CrowdGPT is a **BitTorrent-style compute swarm** for training Large Language Models. Instead of relying on centralized mega-corporations, CrowdGPT turns idle GPUs into a single, decentralized supercomputer.

- **Permissionless:** No sign-up required, no API keys.
- **Byzantine Fault Tolerant:** The swarm survives poisoners and bad actors using coordinate-wise median aggregation.
- **Hardware Agnostic:** Auto-detects NVIDIA (CUDA), AMD (ROCm), Intel/AMD (DirectML), and Apple Silicon (MPS).

---

## ⚡ Quick Start

Want the absolute fastest setup? Check out [**`easy_install.md`**](./easy_install.md) for one-line install commands that auto-detect your hardware and launch the client instantly.

### 🛠️ Manual Installation

If you prefer to manage your own Python environments or already have PyTorch installed:

1. **Clone the repository:**

   ~~~bash
   git clone https://github.com/Vxtzq/CrowdGPT/
   cd CrowdGPT
   ~~~

2. **Install the requirements for your hardware:**

   *(Note: `torch` is unpinned in these files to preserve your existing CUDA/ROCm installations and avoid re-downloading multi-GB wheels)*

   - **NVIDIA (CUDA):**
     ~~~bash
     pip install -r requirements_cuda.txt
     ~~~
   - **AMD (ROCm):**
     ~~~bash
     pip install -r requirements_rocm.txt
     ~~~
   - **DirectX / Windows GPUs (DirectML):**
     ~~~bash
     pip install -r requirements_directml.txt
     ~~~
   - **CPU / macOS (MPS auto-detected):**
     ~~~bash
     pip install -r requirements.txt
     ~~~

3. **Launch the client:**

   ~~~bash
   python crowdgpt.py
   ~~~

---

## 🎛️ CLI Configuration

By default, the node runs in continuous swarm mode, pulling shards from the coordinator. You can override the defaults to force specific hardware or training intensities:

~~~bash
python crowdgpt.py --server http://your-coordinator:3000 --mode deep --batch-size 8 --precision bf16
~~~

| Flag | Description | Allowed Values |
|---|---|---|
| `--server` |  | Any valid URL |
| `--mode` | Training intensity | `quick`, `balanced`, `deep`, `ultra` |
| `--batch-size` | Batch size (power of 2) | `1, 2, 4, 8, 16, 32, 64, 128, 256, 512` |
| `--seq-len` | Sequence length (power of 2) | `8, 16, 32, 64` |
| `--precision` | Weight precision | `fp32`, `bf16`, `fp16` |
| `--backend` | Force compute backend | `auto`, `cuda`, `rocm`, `directml`, `mps`, `cpu` |
| `--single` | Run one cycle and exit | Boolean flag |

---

## 📚 Contribute Training Data

Want to shape what the model learns? We curate our training data via Pull Requests to our HuggingFace dataset to ensure high quality and prevent poisoning. 

👉 **[Read the Data Contribution Guide](./docs/DATA.md)**

---

## 🤝 Community & Links

- 🌐 **Website:** [crowdgpt.ai](https://crowdgpt.ai) *(Coming Soon)*
- 💬 **Discord:** [Join CrowdGPT](#) *(Link coming soon)*
- 🐦 **Twitter/X:** [@CrowdGPT_ai](#) *(Link coming soon)*
