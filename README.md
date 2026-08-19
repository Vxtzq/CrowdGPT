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

<h3 align="center">The Decentralized, Permissionless LLM Swarm.</h3>
<p align="center">
  No accounts. No gatekeeping. Just raw compute.<br>
  <a href="https://crowdgpt.ai"><strong>crowdgpt.ai</strong> (Coming Soon 🚀)</a> · <a href="./docs/ARCHITECTURE.md">Architecture</a> · <a href="./docs/DATA.md">Contribute Data</a>
</p>

---

## 🌪️ What is CrowdGPT?

CrowdGPT is an **anarchic, BitTorrent-style compute swarm** for training Large Language Models. Instead of relying on centralized mega-corporations, CrowdGPT turns idle GPUs around the world into a single, decentralized supercomputer.

- **Permissionless:** No sign-ups, no emails, no API keys. Just run the script.
- **Byzantine Fault Tolerant:** The swarm survives poisoners and bad actors using coordinate-wise median aggregation.
- **Hardware Agnostic:** Auto-detects NVIDIA (CUDA), AMD (ROCm), Intel/AMD (DirectML), and Apple Silicon (MPS).
- **Power-of-2 Constraints:** Mathematically locked batch sizes and sequence lengths ensure perfect consensus across the swarm.

---

## ⚡ Quick Start (Zero-Friction)

Just copy and paste the one-liner for your operating system. It will automatically clone the repo, detect your GPU hardware, install the correct PyTorch backend, and launch the swarm node.

**🪟 Windows (PowerShell):**
```powershell
git clone https://github.com/Vxtzq/CrowdGPT/; cd CrowdGPT; $req="requirements.txt"; if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $req="requirements_cuda.txt" } elseif (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon|Intel|Arc" }) { $req="requirements_directml.txt" }; Write-Host "🔧 Installing $req"; pip install -r $req; python crowdgpt.py
