# Unused HiveMind modules — review required

The following modules were not imported by the client, the installer, or any
production path:

- `hivemind_bridge.py` (316 lines)
- `hivemind_loss_check.py` (405 lines)
- `hivemind_loss_check_service.py` (264 lines)

The only references were documentation and tests. The documentation also points
at paths that do not exist in this repository (`src/hivemind/...`,
`scripts/...`). Together these are 985 lines of unexercised audit logic.

## Why this is a draft

This removal is a proposal, not a judgement that the ideas are unwanted. The
author may intend to wire these modules into the coordinator. Do not merge
without an explicit decision from the maintainer. If there is a plan, close
this PR or restore the files and document the intended integration point.

## If removed

The security and integrity claims in `docs/HIVEMIND_SYNERGY.md`,
`docs/CROWDGPT_GPU_LOSS_CHECK.md`, and
`docs/CROWDGPT_COLLABORATION_AGREEMENT.md` describe reference designs, not
enforced behavior in the current client. That should be made explicit in those
documents as a follow-up.
