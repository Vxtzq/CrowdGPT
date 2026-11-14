# Signed records, fingerprints, and audit (P2)

## Node identity and round records

- The server keeps an Ed25519 key at `orchestrator_node_key.pem` (0600).
- It registers a `NodeRecord` (`orchestrator.nodes`) and publishes
  `/api/node` + `/api/nodes`.
- Every round close builds an `ng-1` `RoundRecord`: open/close timestamps,
  model digest, leader, ticket/submission/delta Merkle roots, previous record
  hash, byte counters, code versions.
- The leader signs the canonical payload; the certificate is appended after
  signing. `verify_certificate()` rejects a tampered field or unknown key.
- Key rotation writes `valid_until = now + overlap_s` on the old key; the
  registry resolves both keys during the overlap and only the new one after.
- Clients keep an Ed25519 key at `checkpoints/client_key.pem` and send
  `X-Client-Pubkey`; the server registers it and verifies the signed
  `StepReceipt` (`ticket_id`, round, base hash, tokens, steps, loss samples,
  fingerprint root, timestamps, nonce).

## Activation fingerprints

`orchestrator/fingerprints.py` implements the validator side:

- a fixed 1536x64 Gaussian projection;
- one 64-bit SimHash per token;
- 32-token block = 2-byte count + 32*8 bytes = **258 bytes**;
- roots are SHA-256 over blocks.

`orchestrator/validation.py` samples steps deterministically and compares
claimed blocks against a recompute callback. The client computes the same
fingerprint when `CROWDGPT_FINGERPRINT=1` and puts the root in its receipt.

## Audit log

`/api/audit/rounds`, `/api/audit/checkpoint`, `/api/audit/proof`, and
`/api/audit/chain` expose the append-only publications. A client can verify a
published record with `crowdgpt_ng.protocol.verify_round_certificate` and a
leaf with `verify_merkle_proof`; neither requires the server package. `/audit` is a small
HTML page over those endpoints. `public_rounds.jsonl` is copied into the HF
upload folder as `rounds.jsonl`; `CROWDGPT_AUDIT_MIRROR` copies it to an
independent path.

## Adversarial fixtures

`tests/test_orchestrator_adversarial.py` covers implausible loss, a sybil
origin pair, replay of a consumed ticket, duplicate delta content, and a
tampered signed record. The HTTP harness additionally rejects a submitted
loss of `0.001` with `{"reason":"implausible_loss"}`.

## Evidence

- `CrowdGPT-Server`: 32 passed.
- `Crowdgpt-ng` harness: 12 passed.
