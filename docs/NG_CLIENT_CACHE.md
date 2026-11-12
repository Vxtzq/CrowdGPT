# Client cache, resume, and prefetch (P1.4)

## Weight cache

Weights live under `checkpoints/hf_weights/blobs/` and are named by the HF
content id (`X-Linked-Etag`, the LFS SHA-256), size, and role:

`<role>__<content-id>__<size>.safetensors`

A new commit that leaves the file unchanged hits the same path. The client logs
`Weight cache: N/M hit (P%)` every round before loading.

## Download resume

`_download_file_with_progress` keeps `<name>.part` and issues a Range request
from the current size. A user stop or network fault leaves the fragment in
place; the next attempt continues rather than restarting.

## Unit checkpoint

`CROWDGPT_UNIT_CHECKPOINT_EVERY=<steps>` (default `0`, off) makes the trainer
write `checkpoints/unit_resume/<task>.pt` every N optimizer steps. The file
contains the model state, both optimizer states, and the counters needed to
continue (micro-step, tokens, running loss sums). On a later process start with
the same task id, the trainer loads it and logs `Resumed unit ... at step N`.
The checkpoint is deleted when the delta is handed to the uploader. It is
opt-in because a full state dict is about 1.4 GB.

## Prefetch

After a delta is handed to the uploader, `prefetch_latest_hf_weights` reads the
model repo's latest revision from the HF API and warms the content-addressed
cache if that revision is not already present. It never asks the coordinator
for a task, so it cannot reserve a pool slot the client will not train.

## Tests

`pytest tests/test_ng_timing.py -q` → 6 passed: unit-checkpoint round-trip
(model parameters and counters), cache path stability across revisions, plus
the timing-log cases.
