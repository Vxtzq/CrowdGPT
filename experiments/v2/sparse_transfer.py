"""How much does a sparse delta save, and does error feedback keep it honest?

Two questions, two measurements:

1. **Bytes.** A delta is one bf16 value per server parameter. At 657.8 M that is
   1.31 GB per volunteer per round. Top-k keeps a fraction of the coordinates, so
   the payload is `6 bytes x frac x params` (4-byte index + 2-byte value) plus a
   12-byte header. Numbers here are measured on a 16 M-coordinate vector and
   scaled linearly, which is exact because the encoding is per-entry.

2. **Bias.** Compressing a delta and forgetting what you dropped is a *biased*
   operator: coordinates that are consistently small are never transmitted, so the
   model accumulates a permanent lag along exactly those directions. Error
   feedback (SparseLoCo, arXiv 2407.07852) carries the dropped mass forward. The
   test is whether the never-sent mass stays bounded as rounds accumulate.

Run: `/app/crowdgpt/.venv/bin/python experiments/v2/sparse_transfer.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import crowd_sparse as S

PARAMS = 16_000_000          # measured size; the table also reports the real budget
REAL_PARAMS = 657_787_392    # the server's flat vector at the 1.01 B client model
FRACS = (0.002, 0.005, 0.01, 0.02, 0.05)
ROUNDS = 8


def heavy_tailed(n, seed):
    """A plausible round delta: most coordinates tiny, a few large directions.

    A dense Gaussian is the easy case for top-k (its mass is spread), so the
    synthetic delta gives 1% of coordinates a 20x scale — closer to what a real
    update looks like once a few directions dominate.
    """
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, generator=g)
    k = max(1, n // 100)
    idx = torch.randperm(n, generator=g)[:k]
    d[idx] *= 20.0
    return d * 3e-4          # AdamW-sized displacement


def one_round(delta, cfg, feedback=None):
    """Encode one delta the way the client does, return (bytes, sent, residual)."""
    work = feedback.correct("w", delta) if feedback is not None else delta
    idx, val = S.select(work, cfg)
    block = S.encode_block([(idx, val)], total=delta.numel())
    if feedback is not None:
        kept = torch.zeros(delta.numel())
        kept[idx] = val
        feedback.keep("w", work - kept)
    return len(block), float(val.abs().sum()), len(idx)


def main():
    n = PARAMS
    print(f"delta vector: {n:,} coordinates (server-visible parameters)")
    dense = n * 2
    print(f"dense bf16 payload: {dense / 1e6:.1f} MB   "
          f"(1-bit NGD1 would be ~{n / 8 / 1e6:.1f} MB)\n")

    print(f"{'frac':>7} {'entries':>12} {'MB':>9} {'vs dense':>9} {'MB @657.8M':>13} {'L2 err':>8}")
    for frac in FRACS:
        cfg = S.SparseConfig(frac=frac)
        d = heavy_tailed(n, 1)
        nbytes, _, count = one_round(d, cfg)
        # relative L2 of a single-shot send, i.e. what the server would miss this round
        idx, val = S.select(d, cfg)
        sent = torch.zeros(n)
        sent[idx] = val
        err = float((d - sent).norm() / d.norm())
        scaled = nbytes * REAL_PARAMS / n
        print(f"{frac:>7.3%} {count:>12,} {nbytes / 1e6:>9.1f} {dense / nbytes:>8.0f}x "
              f"{scaled / 1e6:>13.0f} {err:>8.2%}")

    print("\nThe bias test: does error feedback bound the lag?")
    print("Consecutive rounds are correlated by rho (d_t = rho*d_{t-1} + noise);")
    print("rho=1 is a displacement the client is still trying to deliver, rho=0 is")
    print("independent noise. Lag is reported in units of one round's delta norm,")
    print("after 8 rounds, so 1.0 means the server is one round behind.\n")
    print(f"{'frac':>7} {'rho':>5} {'lag, no EF':>11} {'lag, EF':>9} {'EF gain':>8}")
    for frac in (0.01, 0.02, 0.05):
        cfg = S.SparseConfig(frac=frac)
        for rho in (0.9, 0.5, 0.0):
            g = torch.Generator().manual_seed(7)
            d = heavy_tailed(n, 5)
            scale = float(d.norm())
            fb = S.ErrorFeedback()
            lost = torch.zeros(n)
            for r in range(1, ROUNDS + 1):
                noise = heavy_tailed(n, 200 + r)
                noise = noise * (scale / float(noise.norm()))
                d = rho * d + (1.0 - rho) * noise
                scale = float(d.norm()) if float(d.norm()) > 0 else scale
                idx, _ = S.select(d, cfg)
                kept = torch.zeros(n); kept[idx] = d[idx]
                lost += d - kept
                one_round(d, cfg, fb)
            actual = fb.correct("w", d)
            lag_noef = float(lost.norm()) / scale
            lag_ef = float(fb.residual["w"].float().norm()) / scale
            print(f"{frac:>7.1%} {rho:>5.1f} {lag_noef:>11.2f} {lag_ef:>9.2f} "
                  f"{(lag_noef / lag_ef if lag_ef > 0 else float('inf')):>7.1f}x")


if __name__ == "__main__":
    main()
