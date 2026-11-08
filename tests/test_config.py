"""Unit tests of the configuration logic (no database, no dataset files).

    PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import unittest
import uuid
from unittest import mock

from wvbench import dataset as ds
from wvbench import indexing, sizing
from wvbench.resolve import OversizeError, resolve
from wvbench.settings import ConfigError, ProfileConfig, parse_params


def _from_registry(name: str) -> ds.DatasetInfo:
    reg = ds.registry()["datasets"][name]
    return ds.DatasetInfo(name, "", "registry", reg["rows"], reg["dim"], reg["distance"],
                          ds.METRICS[reg["distance"]], "float", None, None)


EXPECTED = {
    "glove-200-cosine": "small",
    "celeba-resnet-2048-cosine": "medium",
    "landmark-nomic-768-normalized": "medium",
    "arxiv-nomic-768-normalized": "large",
    "inaturalist-resnet-2048-cosine": "large",
}
HOTPOTQA = "hotpotqa-harrier-640-normalized"


@mock.patch.dict(os.environ, {"WVB_PARAMS": "", "WVB_PROFILE": ""})
class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.pc = ProfileConfig()

    def r(self, name, **kw):
        return resolve(name, pc=self.pc, info=_from_registry(name), **kw)

    def test_auto_profiles(self):
        for name, profile in EXPECTED.items():
            with self.subTest(name=name):
                r = self.r(name)
                self.assertEqual(r.settings.profile, profile)
                self.assertTrue(r.estimate.fits)
                self.assertEqual(r.info.metric, "cosine")

    def test_baseline_rule_simple_mode(self):
        # simple mode with full cache and no quantizer is the baseline rule raw x 1.5 <= 8 GB
        for name in EXPECTED:
            with self.subTest(name=name):
                self.assertTrue(self.r(name, params=["estimator.mode=simple"]).estimate.fits)
        with self.assertRaises(OversizeError):
            self.r(HOTPOTQA, params=["estimator.mode=simple"])

    def test_hotpotqa_rejected_by_default(self):
        with self.assertRaises(OversizeError):
            self.r(HOTPOTQA)

    def test_hotpotqa_fallbacks(self):
        r = self.r(HOTPOTQA, params=["oversize.policy=fallback"])
        self.assertEqual(r.settings.profile, "hnsw_rq8")
        self.assertTrue(r.estimate.fits)
        r = self.r(HOTPOTQA, params=["oversize.policy=fallback", "oversize.fallback=hfresh"])
        self.assertEqual(r.settings.profile, "hfresh")
        self.assertTrue(r.estimate.fits)

    def test_manual_profile_and_precedence(self):
        r = self.r("glove-200-cosine", profile="large", params=["max_connections=40"])
        self.assertEqual(r.settings.profile, "large")
        self.assertEqual(r.index["max_connections"], 40)          # CLI beats profile
        self.assertEqual(r.index["ef_construction"], 128)        # from profile "large"
        with mock.patch.dict(os.environ, {"WVB_PARAMS": "max_connections=20,ingest.batch_size=500"}):
            r = self.r("glove-200-cosine", params=["max_connections=40"])
            self.assertEqual(r.index["max_connections"], 40)      # CLI beats env
            self.assertEqual(r.section("ingest")["batch_size"], 500)

    def test_manual_oversize_needs_flag(self):
        with self.assertRaises(OversizeError):
            self.r(HOTPOTQA, profile="large")
        r = self.r(HOTPOTQA, profile="large", allow_oversize=True)
        self.assertFalse(r.estimate.fits)

    def test_unknown_keys_rejected(self):
        for p in ("nope=1", "ingest.nope=1", "nosection.x=1"):
            with self.subTest(p=p), self.assertRaises(ConfigError):
                self.r("glove-200-cosine", params=[p])
        with self.assertRaises(ConfigError):
            self.r("glove-200-cosine", profile="does-not-exist")

    def test_vector_cache_limits_memory(self):
        full = self.r("arxiv-nomic-768-normalized").estimate
        part = self.r("arxiv-nomic-768-normalized", params=["vector_cache_max_objects=300000"]).estimate
        self.assertLess(part.vector_bytes, full.vector_bytes)


class SizingTest(unittest.TestCase):
    def test_compression(self):
        base = {"index": "hnsw", "quantizer": "none"}
        self.assertEqual(sizing.compression_factor(base, 768), 1)
        self.assertEqual(sizing.compression_factor({**base, "quantizer": "rq", "rq_bits": 8}, 768), 4)
        self.assertEqual(sizing.compression_factor({**base, "quantizer": "rq", "rq_bits": 1}, 768), 32)
        self.assertEqual(sizing.compression_factor({**base, "quantizer": "pq", "pq_segments": 0}, 768), 32)
        self.assertEqual(sizing.compression_factor({"index": "hfresh"}, 768), 32)

    def test_pq_auto_segments_divide_dim(self):
        for dim in (200, 640, 768, 2048):
            self.assertEqual(dim % sizing.pq_segments({"pq_segments": 0}, dim), 0)

    def test_rules(self):
        rules = [{"profile": "a", "max_raw_gb": 1, "min_dim": 500}, {"profile": "b", "max_raw_gb": 1}]
        self.assertEqual(sizing.select_profile(rules, 1000, 1000)[0], "a")
        self.assertEqual(sizing.select_profile(rules, 1000, 100)[0], "b")
        self.assertIsNone(sizing.select_profile(rules, 10**9, 100)[0])


class MiscTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(ds.table_name("glove-200-cosine"), "glove_200_cosine")
        self.assertEqual(ds.collection_name("glove-200-cosine"), "Glove_200_cosine")

    def test_uuid_roundtrip(self):
        for row in (0, 1, 443321, 5_233_328):
            self.assertEqual(ds.uuid_row(ds.row_uuid(row)), row)
            self.assertEqual(ds.uuid_row(str(ds.row_uuid(row))), row)
        self.assertEqual(ds.row_uuid(5), uuid.UUID("00000000-0000-0000-0000-000000000005"))

    def test_parse_params(self):
        self.assertEqual(parse_params(["a=1", "b=0.5", "c=true", "d=x", "e=[1;2]"]),
                         {"a": 1, "b": 0.5, "c": True, "d": "x", "e": [1, 2]})
        with self.assertRaises(ConfigError):
            parse_params(["novalue"])

    def test_timing_mode(self):
        with mock.patch.dict(os.environ, {"ASYNC_INDEXING": ""}):
            self.assertEqual(indexing.timing_mode({"mode": "auto"}, {"ASYNC_INDEXING": "true"}), "async")
            self.assertEqual(indexing.timing_mode({"mode": "auto"}, {"ASYNC_INDEXING": "false"}), "sync")
            with self.assertRaises(ConfigError):
                indexing.timing_mode({"mode": "sync"}, {"ASYNC_INDEXING": "true"})
            with self.assertRaises(ConfigError):
                indexing.timing_mode({"mode": "auto"}, {})
        with mock.patch.dict(os.environ, {"ASYNC_INDEXING": "false"}):
            self.assertEqual(indexing.timing_mode({"mode": "auto"}, {}), "sync")


if __name__ == "__main__":
    unittest.main()
