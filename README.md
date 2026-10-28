<p align="center">
  <a href="https://crowdgpt.net">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/logo-light.png">
      <img src="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo-light.png" alt="CrowdGPT" width="100%" style="max-width: 800px; border-radius: 12px;">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://crowdgpt.net">Website</a> ·
  <a href="https://discord.gg/2Shwqj3Xpx">Discord</a> ·
  <a href="./docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="./docs/DATA.md">Training Data</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Version-v1.8-green?style=for-the-badge">
  <a href="https://discord.gg/2Shwqj3Xpx"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <img src="https://img.shields.io/endpoint?style=for-the-badge&url=https%3A%2F%2Fhuggingface.co%2FVxtzq%2FCrowd-v1%2Fraw%2Fmain%2Fbadge_tokens.json" alt="Tokens Processed">
</p>

<p align="center">
  <strong>The best 0$ AI.</strong>
</p>

# 🆕 CrowdGPT v1.9 Changelog 🆕
- Revert back from broken weights (7 steps back on server)
- UI fixes
- client.py cryptographical signing
- Engram training enabled by default
- Fix freeze at startup on certain devices

## What is CrowdGPT?

CrowdGPT is a **project for training arbitrary big LLMs** using volunteered compute.

Instead of one datacenter doing all the work, CrowdGPT distributes training across participating machines.

**All of this working without money, API keys or any centralized GPU clusters.**

### How it works

See [ARCHITECTURE.md](https://github.com/Vxtzq/CrowdGPT/blob/main/docs/ARCHITECTURE.md)

## ⚡ Quick installation (one command)

For Linux/MacOS

```bash
curl -sSL https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/install.sh | bash
```

For Windows

```
cmd /c "curl.exe -fsSL -o install.bat https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/install.bat && install.bat"
```


## ▶️ Running

The installer adds a `crowdgpt` command and a desktop / Start Menu shortcut. The app opens a window by default.

On a server or over SSH, run without a window:

```bash
crowdgpt --headless                      # reuses the login saved by the app
CROWDGPT_PASSWORD=... crowdgpt --headless --user NAME
crowdgpt --headless --anon               # train without an account
crowdgpt --headless --rounds 5           # stop after 5 rounds
```

`crowdgpt --help` lists all options.


## 🔧 Troubleshooting

- **Stuck on "Pool matchmaking 1/3 (waiting for peers)"**: each round is trained by a pool of 3 clients with compatible hardware and precision. When few people are online, waiting is normal.
- **GPU not used**: check the `Backend:` line in the log. GTX 10xx and older need the cu126 PyTorch build (see `requirements_cuda.txt`). `--backend` forces a specific one.
- **Out of memory**: `--vram-gb` limits how much VRAM the client plans to use. Layers that don't fit are streamed from RAM.
- **Files**: login, settings and downloaded weights are in `checkpoints/` next to `client.py` (`~/.local/share/crowdgpt` on Linux, `~/Library/Application Support/CrowdGPT` on macOS, `%LOCALAPPDATA%\CrowdGPT` on Windows).


## ⏳ Legacy installation (deprecated)

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
python client.py
```

For the **automatic installer**, see [`docs/easy_install.md`](./docs/easy_install.md).

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
