"""Backend coordinator and wire format for the CrowdGPT GPU loss check (see `crowdgpt_loss_check`).

The CPU-only CrowdGPT backend opens one job per round: the base and candidate weights (by URL, identified by the SHA-256
of their bf16 bytes), the held-out tokens (URL, optional byte range, SHA-256) and the window plan. GPU evaluators fetch
the job, compute the losses and post a signed receipt; the backend verifies it with a few CPU spot checks and records a
decision. Framework-agnostic: `handle_get_job` / `handle_post_receipt` return (HTTP status, JSON body) so any web stack
can expose them, and `fastapi_router` wires them into FastAPI when it is installed.

  GET  /fl/loss_check/job?evaluator_id=ID   -> 200 job JSON | 204 no job
  POST /fl/loss_check/receipt   body = receipt JSON   -> 200 verdict JSON | 400 | 404 | 409 | 410

State is one JSON file (atomic rename), so a restart keeps open jobs and decisions. Standard library plus the core module.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # inside HiveMind
    from hivemind.swarm import crowdgpt_loss_check as lc
except ImportError:  # copied next to it in the CrowdGPT repository
    import hivemind_loss_check as lc  # type: ignore[no-redef]

JOB_SCHEMA = "hivemind.crowdgpt.loss_check_job.v1"


@dataclass(frozen=True)
class LossCheckJob:
    job_id: str
    base_url: str
    base_sha256: str
    candidate_url: str
    candidate_sha256: str
    heldout_url: str
    heldout_sha256: str
    window: int
    n: int
    seed: int
    created_at: float
    expires_at: float
    heldout_range: str = ""  # "start-end" byte range of heldout_url, inclusive, empty for the whole file
    schema: str = JOB_SCHEMA

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> "LossCheckJob":
        if d.get("schema") != JOB_SCHEMA:
            raise ValueError(f"unknown job schema {d.get('schema')!r}")
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class _State:
    jobs: dict[str, dict] = field(default_factory=dict)
    receipts: dict[str, list[dict]] = field(default_factory=dict)  # job_id -> verified receipts (wire form + verdict)
    decisions: dict[str, dict] = field(default_factory=dict)


# (base, candidate, heldout) for a job; the backend has these in memory or on disk, since it built the candidate itself
WeightsLoader = Callable[[LossCheckJob], tuple[Any, Any, Any]]


class LossCheckCoordinator:
    def __init__(self, *, evaluator_keys: Mapping[str, bytes], server_secret: bytes, state_path: Path | str, loader: WeightsLoader,
                 model_factory: Callable[[], Any] | None = None, k: int = 2, tol: float = 2e-3, max_regression_tolerance: float = 0.0,
                 min_windows: int = 8, required_receipts: int = 1, agreement_tol: float = 2e-3, job_ttl_s: float = 3600.0,
                 clock: Callable[[], float] = time.time) -> None:
        if required_receipts < 1 or not evaluator_keys:
            raise ValueError("need at least one evaluator key and required_receipts >= 1")
        if required_receipts > len(evaluator_keys):
            raise ValueError("required_receipts cannot exceed the number of known evaluators")
        self.keys, self.secret, self.loader = dict(evaluator_keys), server_secret, loader
        self.model_factory = model_factory or lc.CrowdV1
        self.k, self.tol, self.max_regression, self.min_windows = k, tol, max_regression_tolerance, min_windows
        self.required, self.agreement_tol, self.ttl, self.clock = required_receipts, agreement_tol, job_ttl_s, clock
        self.path = Path(state_path)
        self._lock = threading.Lock()
        self.state = self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> _State:
        if not self.path.exists():
            return _State()
        d = json.loads(self.path.read_text())
        return _State(d.get("jobs", {}), d.get("receipts", {}), d.get("decisions", {}))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".loss_check_state.")
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(self.state), fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- backend API
    def open_job(self, *, base_url: str, base_sha256: str, candidate_url: str, candidate_sha256: str, heldout_url: str,
                 heldout_sha256: str, window: int = 2048, n: int = 32, heldout_range: str = "", job_id: str | None = None) -> LossCheckJob:
        """One job per candidate. The seed is drawn here, after the candidate is fixed."""
        if n < self.min_windows:
            raise ValueError(f"n={n} is below min_windows={self.min_windows}: every receipt would be rejected")
        now = self.clock()
        job = LossCheckJob(job_id=job_id or secrets.token_hex(8), base_url=base_url, base_sha256=base_sha256, candidate_url=candidate_url,
                           candidate_sha256=candidate_sha256, heldout_url=heldout_url, heldout_sha256=heldout_sha256, window=window, n=n,
                           seed=secrets.randbelow(2**31), created_at=now, expires_at=now + self.ttl, heldout_range=heldout_range)
        with self._lock:
            if job.job_id in self.state.jobs:
                raise ValueError(f"job {job.job_id} already exists")
            self.state.jobs[job.job_id] = job.to_wire()
            self._save()
        return job

    def decision(self, job_id: str) -> dict | None:
        return self.state.decisions.get(job_id)

    def next_job(self, evaluator_id: str) -> LossCheckJob | None:
        """The oldest open, unexpired, undecided job this evaluator has not answered yet."""
        if evaluator_id not in self.keys:
            return None
        now = self.clock()
        for jid, d in sorted(self.state.jobs.items(), key=lambda kv: kv[1]["created_at"]):
            if jid in self.state.decisions or d["expires_at"] < now:
                continue
            if any(r["receipt"]["evaluator_id"] == evaluator_id for r in self.state.receipts.get(jid, [])):
                continue
            return LossCheckJob.from_wire(d)
        return None

    def submit(self, receipt_json: str) -> tuple[int, dict]:
        """Verify one receipt; record a decision once enough verified, mutually agreeing receipts are in."""
        try:
            r = lc.LossCheckReceipt.from_json(receipt_json)
        except (ValueError, TypeError, KeyError) as e:
            return 400, {"error": f"malformed receipt: {e}"}
        with self._lock:
            jd = self.state.jobs.get(r.round_id)
            if jd is None:
                return 404, {"error": "unknown job (receipt.round_id must be the job_id)"}
            job = LossCheckJob.from_wire(jd)
            if r.round_id in self.state.decisions:
                return 409, {"error": "job already decided", "decision": self.state.decisions[r.round_id]}
            if job.expires_at < self.clock():
                return 410, {"error": "job expired"}
            if any(x["receipt"]["evaluator_id"] == r.evaluator_id for x in self.state.receipts.get(r.round_id, [])):
                return 409, {"error": "this evaluator already answered this job"}
            if (r.window, len(r.starts), r.seed) != (job.window, job.n, job.seed):
                return 400, {"error": "receipt does not follow the job's window plan (window, n, seed)"}
            base, cand, held = self.loader(job)
            v = lc.verify_receipt(r, evaluator_keys=self.keys, server_secret=self.secret, expected_base_sha256=job.base_sha256,
                                  expected_candidate_sha256=job.candidate_sha256, heldout=held,
                                  recompute=lc.cpu_recompute_fn(self.model_factory, base, cand, held, job.window), k=self.k, tol=self.tol,
                                  max_regression_tolerance=self.max_regression, min_windows=self.min_windows)
            verdict = {"accepted": v.accepted, "reason": v.reason, "loss_before": v.loss_before, "loss_after": v.loss_after,
                       "delta_loss": v.delta_loss, "checked": list(v.checked)}
            if v.reason.startswith("REJECTED"):
                # a receipt that fails verification never counts; a held candidate (honest numbers, loss went up) does count
                return 200, {"recorded": False, "verdict": verdict}
            self.state.receipts.setdefault(r.round_id, []).append({"receipt": json.loads(r.to_json()), "verdict": verdict})
            rs = self.state.receipts[r.round_id]
            out: dict[str, Any] = {"recorded": True, "verdict": verdict, "receipts": len(rs), "required": self.required}
            if len(rs) >= self.required:
                parsed = [lc.LossCheckReceipt.from_json(json.dumps(x["receipt"])) for x in rs]
                disagree = [lc.receipts_agree(a, b, self.agreement_tol)[1] for i, a in enumerate(parsed) for b in parsed[i + 1:]
                            if not lc.receipts_agree(a, b, self.agreement_tol)[0]]
                if disagree:
                    dec = {"accepted": False, "reason": f"EVALUATORS_DISAGREE ({disagree[0]})", "evaluators": [p.evaluator_id for p in parsed]}
                else:
                    dec = {"accepted": all(x["verdict"]["accepted"] for x in rs), "reason": rs[-1]["verdict"]["reason"],
                           "loss_before": verdict["loss_before"], "loss_after": verdict["loss_after"], "delta_loss": verdict["delta_loss"],
                           "evaluators": [p.evaluator_id for p in parsed]}
                dec["decided_at"] = self.clock()
                self.state.decisions[r.round_id] = dec
                out["decision"] = dec
            self._save()
            return 200, out

    # ---------------------------------------------------------------- HTTP-shaped handlers
    def handle_get_job(self, evaluator_id: str) -> tuple[int, dict | None]:
        job = self.next_job(evaluator_id)
        return (204, None) if job is None else (200, job.to_wire())

    def handle_post_receipt(self, body: bytes | str) -> tuple[int, dict]:
        return self.submit(body.decode() if isinstance(body, bytes) else body)


def fastapi_router(coord: LossCheckCoordinator, prefix: str = "/fl/loss_check") -> Any:
    """Optional FastAPI wiring of the two handlers (the receipt's HMAC is the authentication)."""
    from fastapi import APIRouter, Request, Response
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix=prefix)

    @router.get("/job")
    def get_job(evaluator_id: str) -> Response:
        status, body = coord.handle_get_job(evaluator_id)
        return Response(status_code=204) if body is None else JSONResponse(body, status_code=status)

    @router.post("/receipt")
    async def post_receipt(request: Request) -> Response:
        status, body = coord.handle_post_receipt(await request.body())
        return JSONResponse(body, status_code=status)

    return router
