# Loss-ratchet runbook

The coordinator can measure the candidate weights of a round and refuse to
accept a model that got worse. That check is what makes the published number
mean something: without it, a round is accepted because the clients said it
converged.

Nothing here is automatic. The ratchet is **off by default** and stays off
until an evaluator answers.

## Two contracts, one service

| Path | Dir | Payload | Answer | Purpose |
|---|---|---|---|---|
| `POST {prefix}/eval` | coordinator → evaluator | `{round, globalStep, checkpoint}` | `{loss, ciLow, ciHigh, …}` | publish `heldoutLoss` in `status.json` |
| `POST {prefix}/gate` | coordinator → evaluator | `{step, round, previousAcceptedLoss, candidate{modelUrl, engramUrl, modelSha256, …}}` | `{loss, ciLow, ciHigh, windows, …}` | decide accept / reject, and roll back in `enforce` |

`tools/hivemind_eval_service.py` serves both from the same process, plus the
HiveMind job/receipt routes. `/gate` downloads the staged candidate, verifies
the declared SHA-256, and scores it on `ng-eval-v1` with `client.py`'s own
model, so the gate measures the same forward pass the swarm trains. If it
cannot measure, it answers 503 or 502 and **never invents a number**.

## Modes

| `CROWDGPT_LOSS_CHECK_MODE` | Behaviour |
|---|---|
| `off` (default) | the evaluator is never called, the ratchet history is not updated, and `/eval` publishing falls back to `lastAcceptedLoss` being unset |
| `log` | the round is measured and the verdict is logged (`🟡 would reject`), nothing is rolled back |
| `enforce` | a rejected round triggers `load_model_weights()` + `load_engram_weights()` and the step counter resets to the pre-round value |

Related knobs: `CROWDGPT_LOSS_CHECK_URL`, `CROWDGPT_LOSS_CHECK_TOKEN`,
`CROWDGPT_LOSS_TOLERANCE` (default `0.0`), `CROWDGPT_LOSS_FAIL_OPEN`
(default `1`), `CROWDGPT_LOSS_TIMEOUT_S` (default 1800).

## Wiring

Evaluator host:

```bash
export CROWDGPT_HIVEMIND_EVALUATOR_KEYS=<64 hex chars>
export CROWDGPT_HIVEMIND_SERVER_SECRET=<32+ bytes>
export CROWDGPT_EVAL_MANIFEST=eval/ng-eval-v1/manifest.json
export CROWDGPT_EVAL_COUNT=64
export CROWDGPT_GATE_SIGNING_KEY=<64 hex chars = 32-byte Ed25519 seed>
python tools/hivemind_eval_service.py --print-config   # gate block must not say "missing"
python tools/hivemind_eval_service.py --serve --host 0.0.0.0 --port 8099
```

Coordinator:

```bash
export CROWDGPT_LOSS_CHECK_URL=http://evaluator.internal:8099/fl/loss_check/gate
export CROWDGPT_LOSS_CHECK_TOKEN=<bearer token, optional but recommended>
export CROWDGPT_LOSS_CHECK_MODE=log
export CROWDGPT_LOSS_FAIL_OPEN=0
export CROWDGPT_LOSS_RECEIPT_REQUIRED=1
export CROWDGPT_LOSS_EVALUATOR_KEYS=primary:<the seed's 64-hex public key>
```

`CROWDGPT_GATE_SIGNING_KEY` on the evaluator and `CROWDGPT_LOSS_EVALUATOR_KEYS`
on the coordinator are two halves of one keypair: `--print-config` prints the
public half as `receiptSigner`, and that is the value the coordinator trusts.

Smoke test before any mode flip — a `503` here means the manifest or the
scorer is missing, and flipping to `enforce` with `FAIL_OPEN=0` would then
block every round:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -X POST "$CROWDGPT_LOSS_CHECK_URL" -H 'Content-Type: application/json' \
  -d '{"step":1,"round":1,"requestId":"00","candidate":{"modelUrl":"https://example.invalid/model"}}'
# 400 = the route is alive and validates (a fake URL is rejected before scoring)
# 503 = the evaluator has no manifest; fix that first
```

## Signed receipts

A receipt binds the loss the evaluator measured to the question the coordinator
asked, so the coordinator no longer has to trust a bare number. On a `200` the
evaluator adds three fields to the gate body:

```json
{
  "loss": 5.125,
  "step": 41,
  "round": 12,
  "receipt": {
    "step": 41, "round": 12, "requestId": "00112233...",
    "modelSha256": "...", "engramSha256": "...",
    "loss": 5.125, "issuedAt": 1763...
  },
  "pubkey": "<64 hex>",
  "signature": "<hex>"
}
```

The signature is Ed25519 over the canonical receipt — `json.dumps(receipt,
sort_keys=True, separators=(",", ":"), ensure_ascii=False)` encoded as UTF-8 —
and the receipt echoes the request's `requestId` and candidate digests verbatim.
The coordinator rejects the round when the signature does not verify, when
`step`/`round`/`requestId`/either digest disagree with what it sent, when the
loss is not a finite number, when `issuedAt` is older than
`CROWDGPT_LOSS_RECEIPT_MAX_AGE_S` (default 3600), or when the same receipt was
already used. A bad receipt is a rejection in **every** mode: `LOSS_FAIL_OPEN`
only covers an evaluator that cannot be reached at all.

Without `CROWDGPT_GATE_SIGNING_KEY` the evaluator answers exactly as it did
before — unsigned. That only pairs with `CROWDGPT_LOSS_RECEIPT_REQUIRED=0`, and
that combination is for a staged rollout, not a steady state.

## Week one: `log`

Run `log` for at least 7 days. Watch `GET /api/admin/loss_ratchet?secret=…`:

- `lastAcceptedLoss` / `lastAcceptedStep` advance every round.
- `recent` should show small negative deltas — the model improving.
- Any `evaluator HTTP 5xx` in the coordinator log is a measurement failure,
  not a model failure. With `FAIL_OPEN=1` the round still proceeds.

## Flip to `enforce`

1. Confirm the evaluator answered every round for 7 days (no 5xx, no
   timeouts, no `evaluator unavailable` in the log).
2. Confirm `enforce` has its rollback point: `checkpoints/` must contain the
   pre-round model and engram at the expected sizes (`MODEL_SIZE*4` and
   `ENG_PARAMS*4` bytes); `_ensure_rollback_point()` writes them if missing.
3. Decide `LOSS_FAIL_OPEN`: keep `1` (a dead evaluator does not stop training)
   until a second, independent evaluator exists; set `0` only then, because
   `0` plus a broken evaluator rejects every round forever.
4. Flip at a round boundary:
   `POST /api/admin/loss_ratchet?secret=…&mode=enforce`
   (or set `CROWDGPT_LOSS_CHECK_MODE=enforce` and restart — the admin call is
   reversible without a restart).

## Rollback

`POST /api/admin/loss_ratchet?secret=…&mode=log` — the next round is measured
and logged again, no rollback fires. `mode=off` stops measuring entirely.
Both are one request and need no deploy.

## What can go wrong

- **503 on every round** — manifest missing on the evaluator, or the scorer
  refused. With `FAIL_OPEN=1` rounds still pass, so this is silent unless you
  read the log.
- **502 on every round** — candidate download failed (expired token, URL not
  reachable from the evaluator host) or a digest mismatch. The coordinator
  treats it as "evaluator unavailable".
- **Rejections on a healthy model** — compare `loss` for the same round with
  `python tools/reproduce_eval.py --model client --count 64`. If the two
  disagree, the evaluator is scoring a different checkpoint than the one that
  was published; stop and fix that before trusting any rejection.
- **A rejected round that should have passed** — check `LOSS_TOLERANCE`; a
  tolerance of `0.0` rejects any non-improvement, including noise smaller than
  the ±0.01 bootstrap interval. Raise it only with a measurement to justify the
  number.

## Decision D11 (recorded 2026-11-15)

Flip to `log` as soon as the gate is wired, run it for one week, then
`enforce` with `LOSS_FAIL_OPEN=1`. Do **not** set `FAIL_OPEN=0` until a second
independent evaluator (different operator, different hardware) agrees with the
first within 0.02 nats on 20 consecutive rounds. Rationale: a single evaluator
is a single point of both failure and capture; while there is one, fail-open
keeps the swarm training and the published number honest, and the ratchet's
value is catching regressions, not punishing outages.
