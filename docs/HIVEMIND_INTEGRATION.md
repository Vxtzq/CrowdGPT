# HiveMind integration notes

The three HiveMind modules are retained. The maintainer expects them to be
useful soon, so they are no longer treated as dead code.

## What each module is for

- `hivemind_bridge.py`: exact CrowdGPT delta pack/unpack, chunked Multi-Krum,
  and the loss-ratchet gate.
- `hivemind_loss_check.py`: the `CrowdV1` evaluator, signed `LossCheckReceipt`,
  spot-check selection, and receipt verification.
- `hivemind_loss_check_service.py`: the FastAPI-side `LossCheckCoordinator`
  with persisted jobs, verification slots, and receipt handling.

## Where NG should meet them

- The signed `StepReceipt` and held-out eval work in NG should be compared with
  `LossCheckReceipt` before adding another receipt format.
- The evaluator sidecar can wrap `LossCheckCoordinator` instead of maintaining a
  separate `eval_service` implementation forever.
- `hivemind_bridge` is a candidate for delta/Multi-Krum compatibility tests, not
  a second client path.
- Do not delete or bypass the modules; wire them behind an explicit adapter and
  keep the existing three-user verification pool.

## Evaluation adapter

`tools/hivemind_eval_service.py` is the additive adapter between deployment
configuration and the existing coordinator. It does not reimplement job state,
receipt verification, or decisions: those stay in `LossCheckCoordinator`.

It provides:

- `ServiceConfig.from_env()` — parse `CROWDGPT_*` configuration.
- `FileJobLoader` — read `base_url`, `candidate_url`, and `heldout_url` from local
  paths or `file://` URLs, with hash checks before any forward pass.
- `build_coordinator(config, loader, **overrides)` — create the unchanged
  coordinator with config defaults; overrides are for verification tuning.
- `create_router(...)` / `create_app(...)` — FastAPI router and app mounting the
  coordinator's `GET /job` and `POST /receipt` handlers, plus the coordinator's
  own `POST /eval` contract.
- `main()` — `--print-config` for a secret-free configuration dump, and `--serve`
  for an optional uvicorn sidecar.

### Configuration

| Variable | Required | Meaning |
| --- | --- | --- |
| `CROWDGPT_HIVEMIND_STATE_PATH` | No | JSON state file; default `.crowdgpt_hivemind/eval_state.json`. |
| `CROWDGPT_HIVEMIND_EVALUATOR_KEYS` | Yes | `{"eval-a": "<hex>", ...}` or `eval-a:<hex>,eval-b:<hex>`. |
| `CROWDGPT_HIVEMIND_SERVER_SECRET` | Yes | `hex:<hex>` or a literal string; at least 16 bytes. |
| `CROWDGPT_HIVEMIND_MODEL_FACTORY` | No | `module:attr` factory for CPU recomputation; defaults to `hivemind_loss_check.CrowdV1`. |
| `CROWDGPT_HIVEMIND_EVAL_FN` | No | `module:attr` evaluator answering `POST /eval`; absent means the route returns 503. |
| `CROWDGPT_HIVEMIND_ROUTE_PREFIX` | No | Route prefix; default `/fl/loss_check`. |

A bare hex string is decoded as bytes for the server secret; any other string is
UTF-8 encoded. Evaluator keys are always hex. `safe_summary()` and
`--print-config` expose evaluator ids and secret length, never key material.

### Loader contract

The coordinator already defines the contract:

```python
loader(job) -> (base, candidate, heldout)
```

`FileJobLoader` resolves each URL as a plain path or `file://` and accepts:

- weights via `hivemind_loss_check.load_flat_safetensors` (key `weights`/`flat`,
  or a single-tensor file);
- held-out tokens via `.npy`, `.json` (list of ids), or raw little-endian
  `uint32` bytes. `heldout_range` (`start-end`, inclusive) applies only to raw
  files.

`http(s)://` URLs are rejected with a clear error rather than an implicit
network client. Deployments that fetch over HTTP inject their own callable, which
is also how tests avoid a 658M checkpoint.

### Testing without a checkpoint

`tests/test_hivemind_eval_service.py` runs under the stdlib-only CI command
`python -m unittest discover -s tests`. It covers:

- environment parsing, key/secret formats, secret-free summaries, and factory
  loading;
- local path / `file://` resolution and token-file formats;
- a fake loader round-trip through `open_job`, `handle_get_job`, and
  `handle_post_receipt`, using `k=0` so no torch or model is needed;
- bad signatures, unknown jobs, tampered receipts, and replay-free validation;
- the coordinator `/eval` contract: validation, 503 when no evaluator is
  configured, response normalisation, 502 on bad results, and the HTTP route;
- FastAPI route creation and HTTP calls, skipped cleanly when FastAPI or httpx is
  absent.

No test needs a real checkpoint, a GPU, or a network. The loader is injectable
precisely so the fake path exercises the same handler code as production.

### Coordinator `/eval` contract

`server.py::_round_eval` does not speak the job/receipt protocol. It posts:

```
POST {CROWDGPT_EVAL_URL}/eval
{"round": 372, "globalStep": 371, "checkpoint": "/path/model_latest.bin"}
```

and reads back `loss`, `ciLow`, `ciHigh`, `windows`, `manifestSha256`, and
`modelSha256`. Before this change the adapter answered that URL with 404, so
`CROWDGPT_EVAL_URL` could only point at a bespoke sidecar.

The adapter now serves the same path (default prefix `/fl/loss_check`), so one
process answers both protocols and the coordinator needs no code change:

```python
from tools import hivemind_eval_service as hes

def evaluate(payload):                     # round, globalStep, checkpoint
    return {"loss": 2.913, "ciLow": 2.88, "ciHigh": 2.95, "windows": 8,
            "manifestSha256": "...", "modelSha256": "..."}

app = hes.create_app(config, eval_fn=evaluate)
# CROWDGPT_EVAL_URL=http://127.0.0.1:8099/fl/loss_check
```

For `--serve`, set `CROWDGPT_HIVEMIND_EVAL_FN=module:attr` instead of passing a
callable; `create_app(config)` then wires it automatically. Without either, the
route answers 503 rather than inventing a loss.

Behaviour is deliberately opinionated, because a wrong number is worse than no
number:

| Case | Response |
| --- | --- |
| no `eval_fn` supplied | `503 {"error": "evaluator not configured"}` — never a fabricated loss |
| malformed body, missing or wrong-typed fields | `400` |
| body over 64 KiB | `413` |
| evaluator raises, or returns a non-object / non-finite / missing `loss` / inverted CI | `502`, exception type only (no traceback text) |
| success | `200` with `{round, globalStep, loss, ciLow, ciHigh, windows, manifestSha256, modelSha256, source}` |

`evaluate_round(payload, eval_fn)` is a pure function, so the contract is tested
with the standard library alone; the FastAPI route is a thin wrapper.

### Router caveat

The adapter wires the two handlers itself instead of calling
`hivemind_loss_check_service.fastapi_router`. That factory imports `Request`
inside its own body, and the module uses `from __future__ import annotations`;
current FastAPI resolves the string annotation against module globals, does not
find `Request`, and answers `POST /receipt` with 422. Moving `Request`/`Response`
imports to module scope in the service module would remove the need for this
adapter-side wiring, but that is left as a separate, minimal upstream cleanup.

## Decision: sidecar vs direct import

**Recommended for now: direct import / in-process router. Pending maintainer
confirmation.** Mount `create_router()` in the coordinator's existing FastAPI
app and keep one owner of the JSON state file.

Why:

- `LossCheckCoordinator` persists state in one JSON file and serializes access
  with in-process locks. Two processes sharing that path would race. Direct
  import keeps a single state owner without inventing a distributed lock.
- The receipt HMAC already authenticates evaluators, so a second network hop does
  not add authentication value.
- The adapter is already a router; mounting it adds no service to deploy.

Keep the sidecar path (`main --serve`) as the seam for later. Move to it only if
CPU recomputation must be isolated from the coordinator or scaled separately. In
that case the sidecar must be the sole owner of `CROWDGPT_HIVEMIND_STATE_PATH`,
and the coordinator should talk to it over the narrow HTTP contract rather than
opening the state file itself.

## Remaining open decisions

1. Should NG's `StepReceipt` be replaced by, or translated to,
   `LossCheckReceipt`?
2. Which module owns the loss-ratchet decision: the coordinator or the
   evaluator service?
3. When, if ever, does the sidecar split become worth its second deployment and
   state-ownership constraint?

## Current status

This branch adds the adapter, its stdlib-compatible tests, and this document. It
does not change the client, the coordinator, the existing modules, training, or
the three-user verification pool.

## Loss gate (`/gate`)

The coordinator has a second evaluator contract besides `/eval`: before it
accepts a round it stages the candidate weights and asks for a held-out loss.
That is the input to `CROWDGPT_LOSS_CHECK_MODE=enforce`; `/eval` only publishes
`heldoutLoss`.

`create_app(config, gate_scorer=...)` wires it. `gate_scorer` is built by
`build_gate_scorer(GateConfig.from_env())`, which returns `None` when the
manifest is missing — the route then answers `503` instead of measuring
something else. Environment: `CROWDGPT_EVAL_MANIFEST`,
`CROWDGPT_EVAL_DATA_ROOT`, `CROWDGPT_EVAL_COUNT` (default 64),
`CROWDGPT_EVAL_SEED`, `CROWDGPT_EVAL_DEVICE`.

The scorer downloads the URLs in `candidate`, verifies `modelSha256` /
`engramSha256` when present, reads the flat bf16 payloads and scores the fixed
`ng-eval-v1` windows with `client.py`'s own model. Refusals: `400` invalid
request, `503` no scorer, `502` download/verify/scoring failure. It never
returns a loss it did not measure. Operating it: `docs/LOSS_CHECK_RUNBOOK.md`.
