# Running CrowdGPT in Docker (NVIDIA / AMD ROCm / Intel XPU / CPU)

The Docker setup runs `client.py` in `--headless` mode with an unprivileged user (`crowd`).

- **Application code** lives inside the image at `/app`. Rebuilding the image updates the code immediately.
- **Persistent state** lives in the `crowdgpt-data` volume mounted at `/data` (`/app/checkpoints` is symlinked to `/data/checkpoints`), preserving:
  - `checkpoints/auth_token.json` and `checkpoints/device_id.json`
  - `checkpoints/selected_backend.json` and `checkpoints/selected_vram.json`
  - `checkpoints/hf_weights/` (content-addressed model & engram weights)
  - Hugging Face, Torch, CUDA, MIOpen, and SYCL caches

---

## 1. Prerequisites by Backend

| Backend | Host Requirements | Overlay used by `./docker/run.sh` |
|---|---|---|
| **NVIDIA CUDA** (with toolkit) | NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | `docker-compose.cuda.yml` |
| **NVIDIA CUDA** (no toolkit fallback) | NVIDIA driver only (auto-selected when toolkit is absent, or `CROWDGPT_CUDA_NORUNTIME=1`) | `docker-compose.cuda-noruntime.yml` |
| **AMD ROCm** | `amdgpu` kernel driver exposing `/dev/kfd` and `/dev/dri` | `docker-compose.rocm.yml` |
| **Intel Arc / Data Center (XPU)** | `i915` or `xe` kernel driver exposing `/dev/dri` | `docker-compose.xpu.yml` |
| **CPU** | Docker Engine + Compose v2 | `docker-compose.cpu.yml` |

> ⚠️ **Security note:** The `cuda-noruntime` overlay runs the container in `--privileged` mode so it can create NVIDIA device nodes. Prefer installing [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and using the standard `cuda` overlay if possible.

---

## 2. Quickstart

`./docker/run.sh` auto-detects your host GPU (`cuda`, `rocm`, `xpu`, or `cpu`) and chains the matching Compose overlay:

```bash
./docker/run.sh auto build
./docker/run.sh auto up -d
./docker/run.sh auto logs -f
```

You can also target a backend explicitly (or run `docker compose -f docker-compose.yml -f docker-compose.<backend>.yml ...` directly):

```bash
./docker/run.sh cuda up -d
./docker/run.sh rocm up -d
./docker/run.sh xpu up -d
./docker/run.sh cpu up -d

# NVIDIA host without nvidia-container-toolkit installed:
CROWDGPT_CUDA_NORUNTIME=1 ./docker/run.sh cuda up -d
```

---

## 3. Accounts & Authentication

By default, the container trains **anonymously** when no credentials or saved session are found.

### Option A — Register or log in interactively (saves token to `/data/checkpoints/auth_token.json`)

```bash
./docker/run.sh auto run --rm crowdgpt \
  python client.py --register --user YOUR_USER --email you@example.com
```

### Option B — Configure `.env`

Copy `.env.example` to `.env` and set either:

```dotenv
CROWDGPT_USER=your_username
CROWDGPT_PASSWORD=your_password
```

or:

```dotenv
CROWDGPT_TOKEN=your_token
```

Then restart:

```bash
./docker/run.sh auto up -d --force-recreate
```

---

## 4. Verifying GPU Access & Diagnostics

Run the built-in environment doctor inside the container:

```bash
./docker/run.sh auto run --rm crowdgpt python client.py --doctor
```

Or run backend-specific checks:

```bash
# NVIDIA CUDA
./docker/run.sh cuda run --rm crowdgpt \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# AMD ROCm
./docker/run.sh rocm run --rm crowdgpt \
  python -c "import torch; print('hip:', torch.version.hip, 'avail:', torch.cuda.is_available())"

# Intel XPU
./docker/run.sh xpu run --rm crowdgpt \
  python -c "import torch; print('xpu:', torch.xpu.is_available())"

# CPU
./docker/run.sh cpu run --rm crowdgpt \
  python -c "import torch; print('torch:', torch.__version__)"
```

---

## 5. Updating CrowdGPT

Because code lives in the image and state lives in `/data`, updating takes two commands and preserves all weights and credentials:

```bash
git pull
./docker/run.sh auto build
./docker/run.sh auto up -d --force-recreate
```

### Migrating from the legacy `crowdgpt-app` volume

If you used the previous Docker setup (which stored everything in the `crowdgpt-app` volume), copy your existing `checkpoints/` directory into the new `crowdgpt-data` volume once:

```bash
docker run --rm \
  -v crowdgpt_crowdgpt-app:/old:ro \
  -v crowdgpt_crowdgpt-data:/data \
  alpine sh -c "mkdir -p /data/checkpoints && cp -a /old/checkpoints/. /data/checkpoints/"
```

---

## 6. Environment Variables

Set these in `.env` or in your shell:

| Variable | Default | Description |
|---|---|---|
| `CROWDGPT_BACKEND` | `auto` | Runtime backend override (`auto`, `cuda`, `rocm`, `xpu`, `cpu`) |
| `CROWDGPT_ANON` | `1` | Train anonymously if no token/user/saved session is present |
| `CROWDGPT_USER` / `CROWDGPT_PASSWORD` | empty | Account credentials |
| `CROWDGPT_TOKEN` | empty | Pre-existing bearer token |
| `CROWDGPT_VRAM_GB` | auto | Cap VRAM budget in GB (layers exceeding budget stream from RAM) |
| `CROWDGPT_ROUNDS` | `0` | Stop after N rounds (`0` = unlimited) |
| `CROWDGPT_ENGRAM` | `1` | Set `0` to disable engram training (~1.4 GB less RAM) |
| `CROWDGPT_NO_UPDATE` | `1` | Disable in-place auto-update (rebuild image to update instead) |
| `CROWDGPT_NO_TOKENIZER` | `1` | Skip loading tokenizer (`trust_remote_code=True`) in headless mode |
| `CROWDGPT_NO_TUI` | `0` | Set `1` to force plain log lines even when attached to a TTY |
| `CROWDGPT_DEBUG` | `0` | Verbose debug logging |
| `CROWDGPT_EXTRA_ARGS` | empty | Extra CLI flags appended to `python client.py --headless` |
| `CROWDGPT_CUDA_NORUNTIME` | `auto` | Auto-falls back to `cuda-noruntime` when `nvidia-container-toolkit` is absent (`1` = force fallback, `0` = force toolkit) |
| `HSA_OVERRIDE_GFX_VERSION` | empty | ROCm architecture override for consumer Radeon GPUs (e.g. `10.3.0`, `11.0.0`) |

---

## 7. Troubleshooting

- **NVIDIA Blackwell (RTX 50-series) `ArchIncompatible` (`sm_120`):**
  Build with a CUDA 12.8+ PyTorch image:
  ```bash
  BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime ./docker/run.sh cuda build
  ```
- **NVIDIA host without `nvidia-container-toolkit`:**
  `./docker/run.sh` automatically falls back to `docker-compose.cuda-noruntime.yml` when `nvidia-container-toolkit` is not detected. You can also force it explicitly with `CROWDGPT_CUDA_NORUNTIME=1 ./docker/run.sh cuda up -d`.
- **AMD ROCm reports `ROCm reports no usable GPU` on consumer Radeon cards:**
  Set `HSA_OVERRIDE_GFX_VERSION` in `.env` (for example `10.3.0` for RX 6000 / RDNA2 or `11.0.0` for RX 7000 / RDNA3) and recreate the container.
- **`cuda-noruntime` only exposes the first GPU (`/dev/nvidia0`):**
  The noruntime fallback creates device nodes for a single GPU. Multi-GPU hosts should install `nvidia-container-toolkit` and use the standard `cuda` overlay, or run one container per GPU with explicit `CUDA_VISIBLE_DEVICES` values.
- **ROCm base image override:**
  The default ROCm image is pinned for reproducibility. If the pinned tag is unavailable, override it:
  ```bash
  BASE_IMAGE=your-image:tag ./docker/run.sh rocm build
  ```
