"""The public site may only label a number "held-out" when the coordinator says
it was evaluated.

`docs/index.html` renders both the training loss and the held-out loss from the
same `status.json`. Training and held-out losses are not comparable, so the page
must never present one under the other's label. This test pins that contract two
ways: structurally on the markup, and behaviourally by running the page's own
`pickLoss()` in node with representative payloads.

The behavioural half skips cleanly when node is unavailable (the CI job does not
install it); the structural half always runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "docs" / "index.html"


def _read_index() -> str:
    return INDEX.read_text(encoding="utf-8")


def _extract_function(source: str, name: str) -> str:
    """Return the source of a top-level `function name(...) { ... }` block."""
    match = re.search(rf"(?m)^function {re.escape(name)}\(", source)
    if not match:
        raise AssertionError(f"function {name}() not found in docs/index.html")
    start = match.start()
    depth = 0
    i = source.index("{", match.end() - 1)
    while i < len(source):
        char = source[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in {name}()")


class LossLabelMarkupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = _read_index()

    def test_home_loss_label_has_an_id_so_it_can_be_retargeted(self):
        self.assertIn('id="home-stat-loss-lbl"', self.html)

    def test_status_loss_sub_line_has_an_id(self):
        self.assertIn('id="status-loss-sub"', self.html)

    def test_labels_do_not_default_to_a_bare_model_loss(self):
        self.assertNotIn(">Current loss<", self.html)
        self.assertNotIn("'status.stat.loss': 'Model Loss'", self.html)

    def test_translation_keys_for_both_kinds_exist(self):
        for key in ("loss.heldout", "loss.training", "loss.sub.heldout",
                    "loss.sub.heldout.ci", "loss.sub.training", "loss.sub.unevaluated"):
            self.assertIn(f"'{key}':", self.html, f"missing i18n key {key}")

    def test_chart_is_labelled_as_training_loss(self):
        self.assertIn("Training loss history", self.html)


class LossSelectionTests(unittest.TestCase):
    """Run the page's `pickLoss()` unchanged against payload fixtures."""

    CASES = [
        (
            "evaluated round",
            {
                "evalStatus": "evaluated",
                "heldoutLoss": 2.913456,
                "heldoutCi": {"low": 2.88, "high": 2.95},
                "evalRound": 372,
                "trainLoss": 3.0812,
                "lastLoss": 3.0812,
            },
            "heldout",
            2.913456,
        ),
        (
            "unevaluated round keeps the training number",
            {"evalStatus": "unevaluated", "heldoutLoss": None, "trainLoss": 3.0812, "lastLoss": 3.0812},
            "training",
            3.0812,
        ),
        (
            "legacy payload without trainLoss",
            {"lastLoss": 3.05},
            "training",
            3.05,
        ),
        (
            "garbage held-out value never becomes a label",
            {"evalStatus": "evaluated", "heldoutLoss": "oops", "trainLoss": 3.0},
            "training",
            3.0,
        ),
        (
            "held-out present but not evaluated",
            {"evalStatus": "pending", "heldoutLoss": 2.5, "trainLoss": 3.0},
            "training",
            3.0,
        ),
        (
            "empty payload",
            {},
            "training",
            None,
        ),
    ]

    def test_cases(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")

        html = _read_index()
        pick = _extract_function(html, "pickLoss")
        script = (
            pick
            + "\nconst CASES = "
            + json.dumps([[c[1], c[2], c[3]] for c in self.CASES])
            + ";\n"
            + "console.log(JSON.stringify(CASES.map(([d]) => {\n"
            + "  const p = pickLoss(d);\n"
            + "  return { kind: p.kind, value: p.kind === 'heldout' ? p.held : p.train };\n"
            + "})));\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script)
            path = handle.name
        try:
            out = subprocess.run([node, path], capture_output=True, text=True, timeout=30)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        result = json.loads(out.stdout)
        for (name, _payload, kind, value), got in zip(self.CASES, result):
            with self.subTest(name):
                self.assertEqual(got["kind"], kind)
                if value is None:
                    self.assertIsNone(got["value"])
                else:
                    self.assertAlmostEqual(got["value"], value, places=6)


if __name__ == "__main__":
    unittest.main()
