"""A pre-degraded (CPU-head) client must embed exactly like a GPU-head peer.

In a bf16 cohort `_fwl` hands `embed` a bf16 cast of the tied weight. Embedding
from that copy rounds the ids and pushes the embedding gradient back through a
bf16 cast, while a GPU-head peer embeds from the fp32 `wte.weight` — different
enough that the coordinator's cross-check read the pre-degraded client as an
outlier. The guard is that the shortcut only applies when the head copy carries
the weight's own dtype.
"""

import contextlib
import io
import types
import unittest

try:  # the repo CI job is standard-library only
    import pytest  # noqa: F401
    import torch
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")


def _import_client():
    # client.py closes stdio on import under pytest, so give it throwaway streams
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import client
    return client


C = _import_client()


class EmbedHeadDtypeTest(unittest.TestCase):
    def setUp(self):
        self._engram = C.ENGRAM_ACTIVE
        C.ENGRAM_ACTIVE = False
        torch.manual_seed(0)
        self.model = types.SimpleNamespace(
            main_device=torch.device("cpu"),
            cpu_device=torch.device("cpu"),
            head_on_gpu=False,
            wte=torch.nn.Embedding(32, 8),
        )
        self.idx = torch.randint(0, 32, (4, 6))

    def tearDown(self):
        C.ENGRAM_ACTIVE = self._engram

    def test_a_bf16_head_copy_does_not_change_the_embedding(self):
        want = C.SotaGPT.embed(self.model, self.idx)
        got = C.SotaGPT.embed(self.model, self.idx,
                            head_w=self.model.wte.weight.to(torch.bfloat16))
        self.assertTrue(
            torch.equal(got, want),
            "a CPU-head client embedded from the bf16 head cast, so it diverges "
            "from its pool mates in a bf16 cohort",
        )

    def test_an_fp32_head_copy_is_still_reused(self):
        want = C.SotaGPT.embed(self.model, self.idx)
        got = C.SotaGPT.embed(self.model, self.idx, head_w=self.model.wte.weight)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
