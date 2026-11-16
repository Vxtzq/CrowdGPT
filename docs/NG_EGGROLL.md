# Toy linear-regression spike

`crowdgpt_ng/eggroll.py` is a dependency-free toy prototype of the EGGROLL /
low-rank evolution-strategies idea. It is a mechanism check, not a 658M-model
result and not a production optimizer.

Command:

    .venv/bin/python tools/bench_eggroll.py --steps 100 --population 32 --rank 2 --lr 0.05

Measured on the toy linear-regression task:

| measure | value |
|---|---|
| first loss | 1.05081639 |
| last loss | 0.14507398 |
| loss ratio | 0.138058 |
| toy params | 4 |
| scalar payload | 136 bytes |
| projected dense bf16 delta | 1,315,574,784 bytes |
| projected compression | 9,673,344x |

The projection is arithmetic from 657,787,392 model parameters and a
32-member population; the toy task has only four parameters, so the toy
compression value is not meaningful. The only claims supported here are that
the low-rank perturbation mechanics are deterministic, the toy objective
converges, and a scalar-only upload is tiny in principle.

Open questions for a real spike: convergence on a real checkpoint, the number
of population members needed at 658M scale, how the fitness scalar is made
checkable by the coordinator, and how three-user pools are preserved for
verification.

# Small Transformer benchmark

`tools/bench_eggroll_model.py` extends the spike above from a four-parameter
linear model to a runnable small Transformer LM trained only by EGGROLL-style
low-rank evolution strategies (no gradients, no dense parameter deltas). It is
still a **scaled-down mechanism benchmark**: the 658M production model is not
instantiated, trained, or measured here.

The model is a pre-norm decoder-only Transformer (token embedding tied to the
output head, learned position embeddings, causal multi-head attention, GELU
MLP). Two-dimensional weights are perturbed in low-rank form
`dW = (sigma / sqrt(rank)) A B^T`; LayerNorm vectors are perturbed full-rank.
The perturbation is never materialised in the forward pass: a linear layer
computes `x W^T + (sigma / sqrt(rank)) (x B) A^T`, two thin matmuls, which is
the EGGROLL inference-cost claim. Each population member is scored by a single
scalar next-token loss; the update is the usual ES estimator with z-scored
fitness, `theta <- theta - (lr / (n * sigma)) sum_i z_i dW_i`.

The task is synthetic and local: `token[t+1] = (token[t] + 1) mod vocab` with a
random start per row. That keeps the loss a measure of the optimizer rather
than of data difficulty.

## Command and measured output

Run on one NVIDIA GeForce RTX 4060 Ti, torch 2.6.0+cu124, Python 3.12:

    python3 tools/bench_eggroll_model.py --device cuda --dim 64 --layers 2 \
        --heads 4 --seq 32 --vocab 64 --batch 4 --population 32 --rank 4 \
        --sigma 0.02 --lr 0.05 --steps 60 --seed 1

```json
{
  "dense_bytes": 210176.0,
  "device": "cuda",
  "eggroll": {
    "lr": 0.05,
    "population": 32,
    "rank": 4,
    "sigma": 0.02,
    "seed": 1,
    "steps": 60
  },
  "factor_bytes": 1376256.0,
  "factor_compression": 0.15,
  "first_loss": 4.204011,
  "last_loss": 3.567479,
  "loss_ratio": 0.848589,
  "mean_member_loss_first": 4.206023,
  "mean_member_loss_last": 3.66056,
  "model": {
    "batch": 4,
    "dim": 64,
    "heads": 4,
    "layers": 2,
    "params": 105088,
    "seq": 32,
    "vocab": 64
  },
  "model_params": 105088.0,
  "note": "scaled-down Transformer benchmark; the 658M model is not run and payload figures are arithmetic, not production measurements",
  "peak_gpu_memory_mb": 10.55,
  "scalar_bytes": 384.0,
  "scalar_compression": 547.33,
  "schema": "ng-eggroll-model-bench-1",
  "seconds": 8.7363,
  "tokens_per_second": 28130.89,
  "tokens_processed": 245760
}
```

`first_loss` / `last_loss` are measured on a fixed evaluation batch before and
after training. Tokens processed are `steps * population * batch * seq`
(245,760); `tokens_per_second` therefore counts every member's forward pass.
The same configuration with seeds 2 and 3 gave loss ratios 0.891 and 0.862, so
the direction is consistent but the exact drop is seed-dependent and not
monotonic step to step.

## What the numbers support

- The low-rank perturbation + scalar-fitness ES loop reduces next-token loss on
  a real Transformer forward pass (4.204 -> 3.567, -15.1%, seed 1), with no
  backward pass and no dense weight delta.
- A scalar-only client report is tiny compared with a dense upload: 384 bytes
  for 32 members (one fp32 fitness plus an 8-byte id each) against 210,176
  bytes of bf16 weights, 547x smaller in this arithmetic.
- The fused low-rank forward is correct: `tests/test_ng_eggroll_model.py`
  checks it against materialised `W + dW` to `atol=1e-5`.
- Peak CUDA memory for this configuration is 10.55 MB and one run takes ~8.7 s.

## Caveats

- **Not the production model.** This model has 105,088 parameters, about 0.016%
  of the 657,787,392-parameter model. Nothing here proves convergence at 658M,
  and the loss values are not comparable to language-modeling numbers.
- **The task is synthetic.** A local successor rule is far easier than real
  text; a falling loss measures the optimizer, not model quality.
- **Throughput is not a prediction.** At 105k parameters each step is dominated
  by Python/torch op overhead (32 sequential forwards), so ~28k tokens/s says
  nothing about 658M-scale throughput.
- **Memory is not a prediction.** 10.55 MB means the model fits in cache. The
  benchmark evaluates population members sequentially, so memory stays near one
  model; this structure, not the number, is the transferable observation.
- **Payload arithmetic is arithmetic.** Dense bytes are `params * 2`; scalar
  bytes assume a client sends only its fitness and id. If members' low-rank
  factors had to be shipped, traffic would be 1,376,256 bytes, i.e. 0.15x
  compression versus dense — the upload win comes from both sides regenerating
  perturbations from a shared seed, not from shipping low-rank factors.
- **Hyperparameters are lightly tuned.** A small sweep found `lr=0.1` diverged
  at population 16 while `lr=0.05` was stable at population 32; no adaptive
  optimizer or learning-rate schedule is implemented.
- **No coordinator path.** Scalar fitness is trusted; how a coordinator verifies
  it, and how three-user pools preserve auditability, remain open questions from
  the spike above.
- **Determinism is CPU-tested only.** `run_benchmark` is bit-identical across
  runs for a fixed seed on CPU (covered by tests). CUDA generator/kernel
  behavior is not guaranteed to reproduce CPU results or results across GPUs.

## Tests

    python3 -m unittest tests.test_ng_eggroll_model -v

Tiny CPU configurations cover a finite-metrics run, loss improvement,
same-seed determinism, fused-vs-materialised forward equivalence, payload
projection, per-member seed derivation, draw-order independence, and CLI flag
parsing. The tests skip cleanly when torch is absent or stubbed, so
`python -m unittest discover -s tests` still passes in the stdlib-only CI.

## Coordinator-side seed regeneration

The scalar-upload claim only pays off if the coordinator can rebuild what each
client actually evaluated. A scalar alone says "member 7 scored 3.91"; it does
not say which perturbation member 7 was.

The original driver drew every member from one `torch.Generator` stream, so
member 7 was reproducible only by replaying members 0-6 first, in order, with
the same population size. That is a fragile contract: any change to the ordering
or the population size breaks it silently.

`--per-member-seeds` (and `--verify-regen`, which implies it) derive each member
independently:

```
seed(step, member) = int.from_bytes(blake2b(f"{base}:{step}:{member}", 8), "little") >> 1
```

`sample_member()` then builds that member's `Perturbation` from its own
generator, so a coordinator that knows `(base_seed, step, member_index)` can
reconstruct the exact factors without receiving them.

`--verify-regen` proves this on the run it is about to measure:

```
python3 tools/bench_eggroll_model.py --device cuda --verify-regen
```

`check_seed_regen()` rebuilds every member twice - once in population order and
once in reverse - and requires that

- each member rebuilds identically regardless of draw order (`reproducible`),
- the seeds are all distinct (`distinct`),
- the seed changes with the step and with the member index.

The run aborts if any of that fails. The result JSON records the outcome:

```json
"seeds": {
  "perMember": true,
  "scheme": "blake2b(base_seed:step:member)[0:8] >> 1",
  "regeneration": {
    "reproducible": true,
    "distinct": true,
    "changesWithStep": true,
    "changesWithMember": true,
    "passes": true
  }
}
```

What this does **not** do yet: it does not shrink any upload by itself. This
benchmark ships nothing, and a production loop would still need a coordinator
that accepts `(member, fitness)` pairs plus the shared seed and applies the
update itself. This change makes that possible and checkable; it does not
implement the coordinator side. The single-generator stream also remains the
default, so the earlier measured runs stay reproducible.

## Measured at 11M parameters

Run on a Radeon 890M (gfx1150, 30.48 GB shared RAM, ROCm 7.2, torch
2.14.1+rocm7.2) with per-member seeds and the regeneration check enabled:

```
python3 tools/bench_eggroll_model.py --device cuda \
  --dim 320 --layers 8 --heads 8 --vocab 4096 --seq 128 \
  --batch 2 --population 32 --rank 8 --sigma 0.02 --lr 0.05 --steps 20 \
  --verify-regen
```

| measure | value |
|---|---|
| parameters | 11,192,960 |
| steps x population x batch x seq | 163,840 tokens |
| wall clock | 10.72 s |
| throughput | 15,283 tok/s |
| peak GPU memory | 173.75 MB |
| scalar payload (32 members) | 384 B (58,297x under dense bf16) |
| low-rank factors, if shipped | 53,559,296 B (0.42x dense) |

The regeneration block reported `reproducible`, `distinct`, `changesWithStep`,
`changesWithMember` and `passes` all true at step 0 with population 32, which is
the whole point of the flag: the population this run trained with can be
rebuilt from `(seed, step, member)` alone.

Two honest readings:

- **The loss went up** (8.364213 -> 8.676544 over 20 steps). That is a property
  of the benchmark, not a convergence result: `make_batch()` draws uniform
  random token ids, so there is no learnable signal in the data and the ES
  update is fitting noise. The tiny CPU test that asserts loss improvement
  passes at its own fixed seed, which is luck rather than evidence -- treat any
  loss number from this tool as a smoke signal only.
- **Shipping the factors would defeat the purpose.** 53.6 MB of fp32 factors is
  2.4x the 22.4 MB dense bf16 upload for the same model, so seed regeneration is
  not an optimisation here; it is what makes a scalar-only upload honest.

The 658M rung is not attempted. 20 steps at 11.2 M parameters already take
10.7 s on 8 compute units; the production model would not finish a useful run
on this hardware, and extrapolating from 11 M would be a guess dressed as a
measurement.
