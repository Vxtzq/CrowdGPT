# EGGROLL mechanism spike

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
