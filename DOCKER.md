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
| **AMD ROCm** | `amdgpu` kernel driver exposing `/dev/kfd` and `/dev/dri` (no ROCm install needed on the host; the image brings its own). See [AMD ROCm](#3-amd-rocm) | `docker-compose.rocm.yml` |
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

`./docker/run.sh` uses `docker compose` when it is installed and falls back to `podman-compose` otherwise, so rootless Podman hosts work with the same commands.

You can also target a backend explicitly (or run `docker compose -f docker-compose.yml -f docker-compose.<backend>.yml ...` directly; for ROCm, prefer `./docker/run.sh`, which picks the image for your GPU and works around a `podman-compose` limitation, see [AMD ROCm](#3-amd-rocm)):

```bash
./docker/run.sh cuda up -d
./docker/run.sh rocm up -d
./docker/run.sh xpu up -d
./docker/run.sh cpu up -d

# NVIDIA host without nvidia-container-toolkit installed:
CROWDGPT_CUDA_NORUNTIME=1 ./docker/run.sh cuda up -d
```

---

## 3. AMD ROCm

### Image selection

For the `rocm` backend, `./docker/run.sh` reads the GPU target from the kernel
(`/sys/class/kfd/kfd/topology/nodes/*/properties`, so no ROCm tools are needed on the host)
and picks the base image:

| Detected GPU | Base image | PyTorch |
|---|---|---|
| A target the ROCm 10.2 image ships: RDNA1–RDNA4 (`gfx1010`–`gfx1036`, `gfx1100`–`gfx1103`, `gfx1150`–`gfx1153`, `gfx1200`, `gfx1201`), `gfx1250`, and `gfx908` / `gfx90a` / `gfx942` / `gfx950` | `rocm/pytorch:rocm10.2.0_ubuntu24.04_py3.12_pytorch_release_2.15.0` | 2.15, Python 3.12 |
| Anything else, e.g. Vega (`gfx900`) or Radeon VII / MI50 (`gfx906`) | `rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.6.0` | 2.6, Python 3.10 |
| No GPU detected | ROCm 10.2 | |

With several GPUs, ROCm 10.2 is used only if every one of them is supported. The choice is
printed on each run, for example:

```text
[run.sh] ROCm GPU: gfx1201 -> supported by ROCm 10.2
[run.sh]   base image: rocm/pytorch:rocm10.2.0_ubuntu24.04_py3.12_pytorch_release_2.15.0
[run.sh]   bitsandbytes arch: gfx1201
```

If `HSA_OVERRIDE_GFX_VERSION` is set, its target is used instead of the detected one (e.g.
`10.3.0` → `gfx1030`), because that is what the ROCm runtime will treat the GPU as.

Values you set yourself always win: `BASE_IMAGE`, `TORCH_INDEX` and `BNB_ROCM_ARCH` from
your shell or `.env` are left untouched. Rebuild after changing GPU or image:
`./docker/run.sh rocm build`.

### 8-bit AdamW (bitsandbytes)

The PyPI `bitsandbytes` package has no build for ROCm 10.x, so the ROCm image compiles
bitsandbytes 0.50.2 from source for the detected GPU (`BNB_ROCM_ARCH`). This makes the
first build take noticeably longer. With an empty `BNB_ROCM_ARCH` (for example when no GPU
was detected), bitsandbytes is built for its own default list of GPUs instead: slower, but
it runs on any card in that list.

At startup the client runs a one-step `AdamW8bit` self-test on the GPU. A passing run logs
`bnb8=True` and `optim=adamw8bit`; if the test fails, the client falls back to fp32 AdamW,
which needs four times the optimizer memory. Set `CROWDGPT_DISABLE_BNB=1` to force fp32.

### Host IPC namespace

The ROCm container must share the host IPC namespace (`ipc: host` in
`docker-compose.rocm.yml`). Without it, the GPU is detected but the first allocation aborts
with `Memory critical error by agent node-0 ... Reason: Memory in use` (exit code 139).

`podman-compose` (tested with 1.6.0) silently ignores the `ipc:` key, so `./docker/run.sh`
passes `--ipc=host` to `podman run` itself. If you call `podman-compose` directly, add it
yourself:

```bash
podman-compose --podman-run-args="--ipc=host" \
  -f docker-compose.yml -f docker-compose.rocm.yml up -d
```

Host IPC uses the host's `/dev/shm`, so the ROCm overlay removes the base `shm_size`
setting (Podman refuses both together).

### Tested hardware

| GPU | Image | Status |
|---|---|---|
| RX 9070 XT (`gfx1201`, RDNA4) | ROCm 10.2, bitsandbytes built for `gfx1201` | Run end to end (login, cohort, 500 training steps with `adamw8bit`, upload) with rootless Podman and `podman-compose` 1.6.0 |
| RX 7700 XT (`gfx1101`, RDNA3) | ROCm 6.4 (previous default, PyPI bitsandbytes) | Reported working by a user; not yet run on ROCm 10.2 |

Other GPUs in the table above are supported by the image but have not been run, and the
ROCm 6.4 fallback with a source-built bitsandbytes has not been tested on hardware.

To stay on the previous ROCm 6.4 image on a GPU that would now get 10.2, pin it in `.env`
and rebuild:

```dotenv
BASE_IMAGE=rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4
``` See
[docs/ROCM_TUNING.md](./docs/ROCM_TUNING.md) for measurements and ROCm tuning variables.

---

## 4. Accounts & Authentication

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

## 5. Verifying GPU Access & Diagnostics

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

## 6. Updating CrowdGPT

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

## 7. Environment Variables

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
| `HSA_OVERRIDE_GFX_VERSION` | empty | ROCm architecture override for consumer Radeon GPUs (e.g. `10.3.0`, `11.0.0`); `./docker/run.sh` also builds bitsandbytes for this target |
| `BASE_IMAGE` | per GPU (ROCm) | Base image override; for ROCm, `./docker/run.sh` picks 10.2 or 6.4 from the detected GPU unless this is set |
| `TORCH_INDEX` | per image | PyTorch wheel index, used only if the base image's PyTorch does not match the backend |
| `BNB_ROCM_ARCH` | detected GPU | ROCm GPU target(s) to build bitsandbytes for, `;`-separated (e.g. `gfx1201`, `gfx1100;gfx1030`); empty builds bitsandbytes' default list |
| `CROWDGPT_DISABLE_BNB` | unset | Set `1` to skip the bitsandbytes self-test and train with fp32 AdamW |

---

## 8. Troubleshooting

- **NVIDIA Blackwell (RTX 50-series) `ArchIncompatible` (`sm_120`):**
  Build with a CUDA 12.8+ PyTorch image:
  ```bash
  BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime ./docker/run.sh cuda build
  ```
- **NVIDIA host without `nvidia-container-toolkit`:**
  `./docker/run.sh` automatically falls back to `docker-compose.cuda-noruntime.yml` when `nvidia-container-toolkit` is not detected. You can also force it explicitly with `CROWDGPT_CUDA_NORUNTIME=1 ./docker/run.sh cuda up -d`.
- **AMD ROCm reports `ROCm reports no usable GPU` on consumer Radeon cards:**
  Check the GPU line `./docker/run.sh rocm build` prints. RDNA1–RDNA4 cards are supported natively by the ROCm 10.2 image and normally need no override. For a card outside that list, set `HSA_OVERRIDE_GFX_VERSION` in `.env` to the nearest supported target (for example `10.3.0` for RDNA2 or `11.0.0` for RDNA3), then rebuild and recreate the container.
- **AMD ROCm crashes with `Memory critical error ... Reason: Memory in use` (exit 139):**
  The container is not in the host IPC namespace. Start it with `./docker/run.sh rocm up -d`, or pass `--ipc=host` yourself when using `podman-compose` directly (see [Host IPC namespace](#host-ipc-namespace)).
- **ROCm startup log shows `bnb8=False`:**
  bitsandbytes failed its self-test and the client is using fp32 AdamW. Make sure the image was built for your GPU (the `bitsandbytes arch:` line from `./docker/run.sh rocm build`), or set `BNB_ROCM_ARCH` explicitly and rebuild.
- **`cuda-noruntime` only exposes the first GPU (`/dev/nvidia0`):**
  The noruntime fallback creates device nodes for a single GPU. Multi-GPU hosts should install `nvidia-container-toolkit` and use the standard `cuda` overlay, or run one container per GPU with explicit `CUDA_VISIBLE_DEVICES` values.
- **ROCm base image override:**
  `./docker/run.sh` picks a pinned ROCm image from your GPU (see [Image selection](#image-selection)). To use a different one, set `BASE_IMAGE` in `.env` or your shell:
  ```bash
  BASE_IMAGE=your-image:tag ./docker/run.sh rocm build
  ```
