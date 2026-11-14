"""CrowdGPT-ng protocol and evaluation library.

The package is intentionally dependency-light so the orchestrator, the client,
and an offline auditor can share one definition of a round.
"""

from .protocol import (
    SCHEMA_VERSION,
    CheckpointRef,
    EvalResult,
    NodeRecord,
    ProtocolError,
    RoundRecord,
    StepReceipt,
    Ticket,
    canonical_bytes,
    decimal_str,
    digest_of,
    generate_keypair,
    merkle_proof,
    merkle_root,
    new_nonce,
    new_ticket_id,
    sign_payload,
    utc_now,
    verify_merkle_proof,
    verify_payload,
)

__all__ = [
    "SCHEMA_VERSION",
    "CheckpointRef",
    "EvalResult",
    "NodeRecord",
    "ProtocolError",
    "RoundRecord",
    "StepReceipt",
    "Ticket",
    "canonical_bytes",
    "decimal_str",
    "digest_of",
    "generate_keypair",
    "merkle_proof",
    "merkle_root",
    "new_nonce",
    "new_ticket_id",
    "sign_payload",
    "utc_now",
    "verify_merkle_proof",
    "verify_payload",
]
