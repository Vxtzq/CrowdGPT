# Are ReLoRA, SOLO and MOSAIC compatible?

Author's ask: *"Implement ReLoRA, SOLO, and Mosaic if you want.... After verifying these 3 are
compatible of course"*.

Three different questions hide inside "compatible", so they are answered separately.

## 1. Do they compose on one model?

They compose **in one order and not the other**, and the asymmetry is real rather than a bug:

| attach order | what happens |
|---|---|
| `mosaic` then `relora` | **works.** MOSAIC's slices are name prefixes (`blocks.<i>.`), and ReLoRA renames `blocks.0.lin` → `blocks.0.lin.base` while preserving the prefix, so `slice_of()` still classifies every tensor. Only adapter positions move. |
| `relora` then `mosaic` | **works, and costs you the adapters.** `LoRALinear` exposes `.base.weight`, which MOSAIC treats as an ordinary slice parameter and trains directly — so you get slice-local full-rank training underneath a frozen-base wrapper. Legitimate, but it is not "LoRA inside MOSAIC". |
| `mosaic` + `solo` | **redundant, not broken.** `mosaic_attach` builds its own SOLO heads for the non-terminal slices, so SOLO is already inside MOSAIC. Attaching SOLO first only changes head initialisation (see the RNG note). |
| `solo` + `relora` | **works.** SOLO needs `solo_splits` over `model.blocks`; ReLoRA preserves block identity, so the splits still resolve. |

The selector enforces one paradigm at a time (`CP.select("relora+mosaic")` raises), so
composition is an explicit harness decision, not something the client will do by accident.

## 2. Do they still produce the server's vector?

Yes, for every combination. The wire is one flat bf16 vector over
`_is_server_param()` tensors, and the delta writer walks `named_parameters()` in the same
order `load_base_weights` reads it. Measured on the 1632-scalar fake model (one step,
`|d|` = norm of what the server would merge):

```
backprop         |d|=0.04032  nz=1632/1632
solo             |d|=0.04027  nz=1632/1632
relora           |d|=0.00245  nz=1023/1632
mosaic           |d|=0.13465  nz=1088/1632
solo+mosaic      |d|=0.13370  nz=1088/1632
mosaic+solo      |d|=0.04035  nz=1632/1632
relora+mosaic    |d|=0.00989  nz=1024/1632
mosaic+relora    |d|=0.00239  nz=1024/1632
relora+solo      |d|=0.02323  nz=1565/1632
solo+relora      |d|=0.00239  nz=1024/1632
```

`solo_heads.*` and `engram.*` never appear in the count, at any nesting.

## 3. Do they agree about *when* a step happens?

**No — and this is the trap.** `Paradigm.step()` has two different contracts:

- `backprop`, `solo`, `relora`, `dust` **accumulate gradients**. `step()` ends with
  grads populated and weights untouched; the client's optimizer applies them afterwards.
- `mosaic` **applies its own outer update inside `step()`** — it runs `local_steps` per
  slice, aggregates with the median clip and Nesterov outer step, and copies the result
  back. For MOSAIC the client's optimizer must *not* also step, or the round is applied
  twice.

Any harness that measures the weight delta immediately after `step()` therefore sees zero
for the first four and a real delta only for MOSAIC. That is not a failure; it is the
contract. It is also why the client's `_paradigm_step_safe` hands MOSAIC its own path.

### RNG note

`mosaic_attach` always rebuilds `solo_heads`, so attaching SOLO first consumes RNG draws
and the heads start from different values. In one round the *support* (which scalars move)
is identical, the *magnitudes* are not. Compare support, or fix the seed after attach —
do not assert element-wise equality across attach orders.

## How this is tested

`tests/test_paradigm_compat.py`, 15 tests:

```
/app/crowdgpt/.venv/bin/python -m pytest tests/test_paradigm_compat.py -q
# 15 passed
```

- `test_every_combination_still_uploads_the_server_vector` — 9 attach orders, each must
  emit exactly the reference vector length.
- `test_mosaic_trains_the_same_slices_with_or_without_solo_attached` — support equality.
- `test_relora_inside_mosaic_still_only_moves_adapter_positions` — the adapter confinement
  survives the slice machinery.
- `test_solo_and_mosaic_head_tensors_never_reach_the_wire`.
- `test_unknown_paradigm_is_rejected`, `test_selector_names_the_three_the_author_asked_for`.
- `test_mosaic_slices_are_name_based_so_relora_wrapping_cannot_break_them` — `slice_of` on
  base and adapter names.
