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
  <img src="https://img.shields.io/badge/License-GPL--3.0-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Version-v1.9-green?style=for-the-badge">
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

1. The client asks the server for a task and is placed in a pool of 3 volunteers with compatible hardware.
2. It downloads the current weights from Hugging Face and trains on its shard of the dataset. All members of a pool get the same shard, seed and step count.
3. It uploads the weight delta. The server compares the 3 deltas of each pool, drops outliers, and merges the rest into the global model at the end of the round.
4. The new weights are published to [Vxtzq/Crowd-v1](https://huggingface.co/Vxtzq/Crowd-v1) and the next round starts.

The current model, Crowd-v1, is a 658M-parameter decoder (24 layers, 2048-token context, Qwen3 tokenizer) with an engram lookup table. The client runs no code from the server: it only receives weights and data. More in [ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## 💻 Supported hardware

| Hardware | Backend | Notes |
|---|---|---|
| NVIDIA | CUDA | GTX 10xx and older need the cu126 PyTorch build. 8-bit optimizer via `bitsandbytes` |
| AMD | ROCm or DirectML | ROCm on Linux. On Windows, ROCm for supported GPUs and DirectML for the rest |
| Intel Arc | XPU | |
| Apple Silicon | MPS | |
| Anything else | CPU | Works, but slow |

The installer detects the hardware and installs the matching PyTorch build. Small GPUs work too: layers that don't fit in VRAM are kept in RAM and streamed in. Training the engram table adds about 1.4 GB of RAM (`CROWDGPT_ENGRAM=0` turns it off).

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

`crowdgpt --help` lists all options. A few settings can also be set through environment variables:

| Variable | Description |
|---|---|
| `CROWDGPT_TOKEN` / `CROWDGPT_PASSWORD` | Credentials for headless runs |
| `CROWDGPT_VRAM_GB` | Same as `--vram-gb` |
| `CROWDGPT_ENGRAM=0` | Don't train the engram table (same as `--no-engram`) |

### Updates

The client checks for updates on start and between rounds. An update is only installed if its manifest is signed with the maintainer's key, so a compromised repository or account can't push code to clients. `--no-update` skips the check. Details in [SECURITY.md](./SECURITY.md).


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

---


## 🧠 Contribute

- **Training data** is curated through pull requests. See [`docs/DATA.md`](./docs/DATA.md).
- **Code**: pull requests are welcome. CI checks that the Python files compile and that `install.sh` parses.
- **Bugs**: open an issue with the bug report template, and include the client version and the log.
- **Server**: the coordinator lives in [Vxtzq/CrowdGPT-Server](https://github.com/Vxtzq/CrowdGPT-Server).

## 📄 License

GPL-3.0. See [LICENSE](./LICENSE).

---

<p align="center">
  <a href="https://crowdgpt.net"><strong>crowdgpt.net</strong></a>
  ·
  <a href="./docs/ARCHITECTURE.md">Architecture</a>
  ·
  <a href="./docs/DATA.md">Data</a>
</p>
