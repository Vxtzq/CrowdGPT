"""Stdlib-only tests for tools/domain_eval.py helpers (the model path needs real torch and weights)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import domain_eval as de  # noqa: E402


class TestDomainEvalHelpers(unittest.TestCase):
    def test_parse_domains(self) -> None:
        d = de.parse_domains(["code=a.txt", " math = b.txt "])
        self.assertEqual(d, {"code": Path("a.txt"), "math": Path("b.txt")})
        for bad in (["code"], ["=a.txt"], ["code="], ["x=a", "x=b"]):
            with self.assertRaises(ValueError):
                de.parse_domains(bad)

    def test_split_and_pack(self) -> None:
        self.assertEqual(de.split_documents("a b\n\n\n\nc\n\n  \n"), ["a b", "c"])
        self.assertEqual(de.pack_documents([[1, 2], [3]], eos=9), [1, 2, 9, 3, 9])

    def test_gap_report_sorts_by_gap(self) -> None:
        r = {"own": {"crowd": 3.1, "ref": 2.8}, "code": {"crowd": 5.2, "ref": 1.3}}
        lines = de.gap_report(r, "crowd", "ref").splitlines()
        self.assertTrue(lines[1].startswith("code"))
        self.assertIn("+3.900", lines[1])
        self.assertEqual(len(de.gap_report(r).splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
