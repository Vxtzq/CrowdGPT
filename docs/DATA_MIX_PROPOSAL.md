# Data-mix proposal for Crowd-v1: where the model is weak (measured), and what data would help

Status: **proposal**. Nothing is submitted to the dataset; adding data stays the maintainer's decision.

## Measurement (HiveMind T4, 2026-11-14)

Setup:
- Crowd-v1 revision `5ac3916` (step 366) with its engram table.
- 8 windows of 2,048 tokens per domain; documents packed with `<|endoftext|>`; same Qwen3 tokenizer for both models.
- Reference: Qwen3-0.6B.
- Raw data: xaviercallens/HiveMind `artifacts/experiments` (`p0_domains`). Tool to repeat it on any checkpoint:
  `tools/domain_eval.py` (this PR).

| Domain (source) | Crowd-v1 | Qwen3-0.6B | Gap |
|---|---|---|---|
| Python code (codeparrot-clean-valid) | 5.19 | 1.35 | **+3.84** |
| Math (NuminaMath-CoT test) | 3.33 | 0.89 | **+2.44** |
| Encyclopedic (wikitext-2 test) | 4.71 | 2.91 | **+1.80** |
| Science (SciQ test) | 3.82 | 2.32 | +1.50 |
| Educational web (fineweb-edu sample) | 3.99 | 2.95 | +1.04 |
| CrowdGPT's own chunks (chunk 1026) | 3.12 | 2.76 | +0.36 |

**How to read it.**
- On its own distribution Crowd-v1 is close to Qwen3-0.6B (+0.36 nat). The large gaps are on **code, math and
  encyclopedic text**, which the current chunks barely contain.
- Qwen3 trained on far more code and math, so part of each gap is expected. But +3.8 nat on Python means Crowd-v1 has
  seen almost no code.
- Compare gaps, not absolute numbers; a domain the swarm has trained on is no longer held out.
- No French source was measured: none was available offline on the measuring machine.

## Proposal

1. **Add code, math and reference text to the mix**, in that order of measured gap.

   Candidate open sources; **check each license and its terms on the dataset card before use**:

   | Domain | Candidates |
   |---|---|
   | Code | permissively licensed source files, e.g. a filtered subset of The Stack (bigcode; has terms of use) |
   | Math | OpenWebMath, FineMath (both published under ODC-By, per their cards) |
   | Reference | Wikipedia dumps (CC BY-SA), FineWeb-Edu (ODC-By) for educational web text |

   The current chunks come from Ultra-FineWeb / FineWeb, so FineWeb-Edu overlaps them most. Code and math add the most
   new signal.
2. **Keep the format.** Pre-tokenized `uint32` Qwen3 chunks with continuous packing; no client change is needed.
3. **Track it per checkpoint** with `tools/domain_eval.py`, keeping the same windows (fixed seed) so numbers are
   comparable. A mix change is working when the code / math / wiki gaps shrink without the own-data loss rising.
4. **Mixing ratio** (a starting point, not measured): something like 10–20 % code and 5–10 % math, the rest web text,
   then adjust from the per-domain curves.

## Relation to LLM-JEPA

LLM-JEPA brought no gain on Crowd-v1 in the T4 experiments (`docs/LLM_JEPA_PROPOSAL.md` §5). The data mix is the
cheaper lever and does not depend on it.

JEPA could later use code data directly: docstring ↔ function body is a natural pair of views, once the model has seen
enough code.
