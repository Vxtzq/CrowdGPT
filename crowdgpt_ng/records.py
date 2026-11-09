"""Normalise the coordinator's published artifacts into round records.

The current server publishes three snapshots (``status.json``,
``leaderboard.json``, ``loss_history.json``) that overlap but do not agree:
the leaderboard total and the status total differ, and a round's loss appears
both in ``status.lastLoss`` and in ``loss_history.rounds[]`` with different
values. This module does not reconcile them. It labels each value with where
it came from and keeps reported and checked numbers in separate fields.

Output is deterministic: sorted rounds, canonical JSON per line, one trailing
newline. The same inputs always produce byte-identical output, so the file can
be committed and diffed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import ProtocolError, canonical_bytes, decimal_str

LEGACY_SCHEMA = "ng-legacy-1"


class RecordError(ValueError):
    """An input artifact is not the shape this exporter understands."""


@dataclass(frozen=True)
class LegacyRound:
    """One round as published by the legacy coordinator.

    ``verification`` is always ``unverified`` here: these values are the
    operator's own report. ``loss_checked`` stays ``None`` until a validator
    recomputes it.
    """

    round: int
    step: int | None
    phase: str
    observed_at: str
    tokens: int | None
    deltas: int | None
    loss_reported: str | None
    loss_min_reported: str | None
    loss_max_reported: str | None
    loss_checked: str | None
    verification: str
    source: str
    schema: str = LEGACY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "round": self.round,
            "step": self.step,
            "phase": self.phase,
            "observed_at": self.observed_at,
            "tokens": self.tokens,
            "deltas": self.deltas,
            "loss_reported": self.loss_reported,
            "loss_min_reported": self.loss_min_reported,
            "loss_max_reported": self.loss_max_reported,
            "loss_checked": self.loss_checked,
            "verification": self.verification,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegacyRound":
        if not isinstance(data, Mapping):
            raise RecordError("round record must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise RecordError(f"round record has unknown fields: {sorted(unknown)}")
        missing = allowed - set(data)
        if missing:
            raise RecordError(f"round record is missing fields: {sorted(missing)}")
        return cls(
            schema=data["schema"],
            round=data["round"],
            step=data["step"],
            phase=data["phase"],
            observed_at=data["observed_at"],
            tokens=data["tokens"],
            deltas=data["deltas"],
            loss_reported=data["loss_reported"],
            loss_min_reported=data["loss_min_reported"],
            loss_max_reported=data["loss_max_reported"],
            loss_checked=data["loss_checked"],
            verification=data["verification"],
            source=data["source"],
        )


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise RecordError(f"missing artifact: {source}") from exc
    except json.JSONDecodeError as exc:
        raise RecordError(f"invalid JSON in {source}: {exc}") from exc


def _loss(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{what} must be a number or null")
    try:
        return decimal_str(value)
    except ProtocolError as exc:
        raise RecordError(f"{what}: {exc}") from exc


def _optional_int(value: Any, what: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecordError(f"{what} must be an integer >= {minimum} or null")
    return value


def _round_entries(history: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    entries = history.get("rounds") if isinstance(history, Mapping) else history
    if not isinstance(entries, list):
        raise RecordError("loss history must contain a 'rounds' list")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RecordError("each loss-history round must be an object")
    return entries


def build_legacy_rounds(
    status: Mapping[str, Any],
    history: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[LegacyRound]:
    """Merge the closed rounds with the open round from ``status.json``."""

    if not isinstance(status, Mapping):
        raise RecordError("status must be an object")

    by_round: dict[int, LegacyRound] = {}
    for entry in _round_entries(history):
        round_number = entry.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
            raise RecordError("loss-history round needs a non-negative integer 'round'")
        observed = entry.get("t")
        if not isinstance(observed, str) or not observed:
            raise RecordError(f"round {round_number} is missing its timestamp")
        record = LegacyRound(
            round=round_number,
            step=_optional_int(entry.get("step"), "step"),
            phase="closed",
            observed_at=observed,
            tokens=_optional_int(entry.get("tokens"), "tokens"),
            deltas=_optional_int(entry.get("deltas"), "deltas"),
            loss_reported=_loss(entry.get("loss"), "loss"),
            loss_min_reported=_loss(entry.get("lossMin"), "lossMin"),
            loss_max_reported=_loss(entry.get("lossMax"), "lossMax"),
            loss_checked=None,
            verification="unverified",
            source="loss_history.json",
        )
        previous = by_round.get(round_number)
        if previous is not None and previous.to_dict() != record.to_dict():
            raise RecordError(f"conflicting duplicate round {round_number}")
        by_round[round_number] = record

    current = status.get("currentRound")
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise RecordError("status needs a non-negative integer 'currentRound'")
    if current not in by_round:
        updated = status.get("updatedAt")
        if not isinstance(updated, str) or not updated:
            raise RecordError("status needs an 'updatedAt' timestamp")
        by_round[current] = LegacyRound(
            round=current,
            step=_optional_int(status.get("globalStep"), "globalStep"),
            phase="open",
            observed_at=updated,
            tokens=None,
            deltas=None,
            loss_reported=_loss(status.get("lastLoss"), "lastLoss"),
            loss_min_reported=None,
            loss_max_reported=None,
            loss_checked=None,
            verification="unverified",
            source="status.json",
        )

    return [by_round[number] for number in sorted(by_round)]


def write_rounds_jsonl(records: Iterable[LegacyRound], path: str | Path) -> int:
    """Write records deterministically; return the number written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: item.round)
    payload = b"".join(canonical_bytes(item.to_dict()) + b"\n" for item in ordered)
    target.write_bytes(payload)
    return len(ordered)


def read_rounds_jsonl(path: str | Path) -> list[LegacyRound]:
    target = Path(path)
    records: list[LegacyRound] = []
    with target.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(f"{target}:{number}: invalid JSON: {exc}") from exc
            records.append(LegacyRound.from_dict(data))
    return records


def status_snapshot(status: Mapping[str, Any]) -> dict[str, Any]:
    """A labelled snapshot of the live counters (never mixed with checks)."""

    if not isinstance(status, Mapping):
        raise RecordError("status must be an object")

    def integer(key: str) -> int:
        value = status.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordError(f"status.{key} must be a non-negative integer")
        return value

    return {
        "schema": LEGACY_SCHEMA,
        "source": "status.json",
        "observed_at": status.get("updatedAt"),
        "participants": integer("totalParticipants"),
        "contributions": integer("totalContributions"),
        "active_gpus": integer("activeGpus"),
        "total_tokens": integer("totalTokens"),
        "global_step": integer("globalStep"),
        "current_round": integer("currentRound"),
        "last_loss_reported": _loss(status.get("lastLoss"), "lastLoss"),
        "verification": "unverified",
    }


def leaderboard_stats(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Distribution facts that the current site does not show."""

    if not isinstance(entries, (list, tuple)):
        raise RecordError("leaderboard must be a list")

    tokens: list[int] = []
    zero = 0
    flagged = 0
    low_trust = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RecordError("each leaderboard entry must be an object")
        value = entry.get("totalTokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordError("leaderboard entry needs a non-negative integer totalTokens")
        tokens.append(value)
        if value == 0:
            zero += 1
        if entry.get("byzantineFlags", 0):
            flagged += 1
        trust = entry.get("trustScore")
        if isinstance(trust, (int, float)) and not isinstance(trust, bool) and trust < 0.5:
            low_trust += 1

    total = sum(tokens)
    ordered = sorted(tokens, reverse=True)

    def share(count: int) -> str:
        if total == 0:
            return "0.000000"
        return decimal_str(Decimal(sum(ordered[:count])) / Decimal(total))

    hhi = "0.000000" if total == 0 else decimal_str(sum((Decimal(value) / Decimal(total)) ** 2 for value in tokens))

    return {
        "schema": LEGACY_SCHEMA,
        "source": "leaderboard.json",
        "entries": len(tokens),
        "tokens_attributed": total,
        "zero_token_entries": zero,
        "flagged_entries": flagged,
        "trust_below_0_5": low_trust,
        "top1_share": share(1),
        "top3_share": share(3),
        "top10_share": share(10),
        "hhi": hhi,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export coordinator artifacts to rounds.jsonl")
    parser.add_argument("--status", required=True, help="path to status.json")
    parser.add_argument("--loss-history", required=True, help="path to loss_history.json")
    parser.add_argument("--leaderboard", help="optional path to leaderboard.json")
    parser.add_argument("--out", help="write rounds.jsonl here instead of stdout")
    parser.add_argument("--summary", help="write a JSON summary (status + leaderboard)")
    args = parser.parse_args(argv)

    status = _load_json(args.status)
    history = _load_json(args.loss_history)
    records = build_legacy_rounds(status, history)

    if args.out:
        count = write_rounds_jsonl(records, args.out)
        print(f"wrote {count} rounds to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write("".join(canonical_bytes(item.to_dict()).decode("utf-8") + "\n" for item in records))

    if args.summary:
        summary: dict[str, Any] = {"status": status_snapshot(status)}
        if args.leaderboard:
            summary["leaderboard"] = leaderboard_stats(_load_json(args.leaderboard))
        target = Path(args.summary)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_bytes(summary) + b"\n")
        print(f"wrote summary to {args.summary}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
