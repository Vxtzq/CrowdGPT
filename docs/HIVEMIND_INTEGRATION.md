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

## Open decisions

1. Does the coordinator import the service module directly, or run it as a
   sidecar with a narrow HTTP contract?
2. Should NG's `StepReceipt` be replaced by, or translated to,
   `LossCheckReceipt`?
3. Which module owns the loss-ratchet decision: the coordinator or the
   evaluator service?
4. How do we test the adapter without a real 658M checkpoint or a second GPU?

## Current status

This branch only adds documentation and an integration smoke test. It does not
change the client, the coordinator, or pool sizing.
