import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cfx import config, stats
from cfx.analysis import pair_category
from cfx.parse import explanation_text, normalize_label, parse
from cfx.prompts import SCHEMAS, messages, render
from cfx.revise import finalize
from cfx.spans import check_prediction, check_span, coverage, token_f1


class TestParse(unittest.TestCase):
    def test_good_json(self):
        r = parse("c1", '{"label": "entailment", "explanation": "A dog is an animal."}')
        self.assertEqual(r["parse_status"], "ok")
        self.assertEqual(r["label"], "entailment")
        self.assertTrue(r["key_order_ok"])

    def test_code_fence_and_prose(self):
        r = parse("c0", 'Sure!\n```json\n{"label": "Contradiction"}\n```')
        self.assertEqual((r["parse_status"], r["label"]), ("ok", "contradiction"))

    def test_reordered_keys_still_parse(self):
        r = parse("c1", '{"explanation": "x", "label": "neutral"}')
        self.assertEqual(r["parse_status"], "ok")
        self.assertFalse(r["key_order_ok"])

    def test_broken(self):
        self.assertEqual(parse("c0", "I think it is entailment")["parse_status"], "no_json")
        self.assertEqual(parse("c0", '{"label": "maybe"}')["parse_status"], "bad_label")
        self.assertEqual(parse("c1", '{"label": "neutral"}')["parse_status"], "missing_keys")

    def test_braces_inside_strings(self):
        r = parse("c1", '{"label": "neutral", "explanation": "the {hat} is not mentioned"}')
        self.assertEqual(r["parse_status"], "ok")

    def test_aliases(self):
        self.assertEqual(normalize_label("Entails."), "entailment")
        self.assertIsNone(normalize_label("yes"))

    def test_structured_explanation_text(self):
        r = parse("c3", json.dumps({"label": "neutral", "premise_spans": [], "hypothesis_spans": ["sad"],
                                    "justification": "Mood is not stated.", "missing_information": "her mood"}))
        self.assertIn("her mood", explanation_text(r))


class TestSpans(unittest.TestCase):
    P, H = "A man in a red shirt is riding a bike.", "A person rides a bicycle."

    def test_levels(self):
        self.assertEqual(check_span("red shirt", self.P, self.H), "exact")
        self.assertEqual(check_span("Red shirt!", self.P, self.H), "normalized")
        self.assertEqual(check_span("riding a bikes", self.P, self.H), "fuzzy")
        self.assertEqual(check_span("rides a bicycle", self.P, self.H), "wrong_side")
        self.assertEqual(check_span("blue hat", self.P, self.H), "absent")

    def test_prediction_without_spans(self):
        self.assertIsNone(check_prediction({"label": "neutral"}, self.P, self.H))
        checks = check_prediction({"premise_spans": ["a man"], "hypothesis_spans": []}, self.P, self.H)
        self.assertEqual(checks[0]["match"], "normalized")

    def test_overlap_measures(self):
        self.assertEqual(token_f1(["dog"], ["dog"]), 1.0)
        self.assertEqual(token_f1([], ["dog"]), 0.0)
        self.assertAlmostEqual(coverage("red shirt", "the shirt"), 0.5)


class TestStats(unittest.TestCase):
    def test_bootstrap_constant(self):
        est, lo, hi, n = stats.bootstrap_mean([1, 1, 1, 1], [1, 1, 2, 2], reps=200)
        self.assertEqual((est, lo, hi, n), (1.0, 1.0, 1.0, 4))

    def test_bootstrap_keeps_clusters_together(self):
        v, ids = [1, 1, 1, 0, 0, 0], ["a", "a", "a", "b", "b", "b"]
        est, lo, hi, _ = stats.bootstrap_mean(v, ids, reps=500)
        self.assertEqual(est, 0.5)
        self.assertEqual({lo, hi} <= {0.0, 0.5, 1.0}, True)

    def test_bootstrap_ignores_nan(self):
        est, *_ , n = stats.bootstrap_mean([1, np.nan, 0], [1, 2, 3], reps=100)
        self.assertEqual((est, n), (0.5, 2))

    def test_mcnemar(self):
        n01, n10, p = stats.mcnemar_exact([True] * 10, [False] * 10)
        self.assertEqual((n01, n10), (0, 10))
        self.assertAlmostEqual(p, 2 / 1024)
        self.assertEqual(stats.mcnemar_exact([True], [True])[2], 1.0)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_kappa(self):
        self.assertEqual(stats.cohen_kappa(list("abc"), list("abc")), 1.0)
        self.assertAlmostEqual(stats.cohen_kappa(list("aabb"), list("abab")), 0.0)
        self.assertEqual(stats.weighted_kappa(["no", "yes"], ["no", "yes"], ["no", "partly", "yes"]), 1.0)

    def test_macro_f1(self):
        g = ["entailment", "neutral", "contradiction", "neutral"]
        self.assertEqual(stats.macro_f1(g, g, config.LABELS), 1.0)
        p = ["entailment", "entailment", "contradiction", "neutral"]
        self.assertAlmostEqual(stats.macro_f1(g, p, config.LABELS), (2 / 3 + 2 / 3 + 1) / 3)

    def test_paired_diff(self):
        d = stats.bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0], [1, 2, 3, 4], reps=200)
        self.assertEqual((d["diff"], d["lo"], d["hi"]), (1.0, 1.0, 1.0))
        self.assertLess(d["p"], 0.01)


class TestLeakage(unittest.TestCase):
    def test_prompts_only_show_premise_and_hypothesis(self):
        items, _ = config.load_items("eval")
        for it in items[:60]:
            for kind in ("c0", "c1", "c2", "c3", "c3n"):
                text = "\n".join(m["content"] for m in messages(kind, {"premise": it["premise"],
                                                                       "hypothesis": it["hypothesis"]}))
                for ref in it.get("ref_explanations", []):
                    self.assertNotIn(ref, text)
                for field in ("label_reason", "changed_span_original"):
                    if it.get(field) and len(it[field]) > 25:
                        self.assertNotIn(it[field], text)
                user = messages(kind, {"premise": it["premise"], "hypothesis": it["hypothesis"]})[1]["content"]
                self.assertNotIn(it["gold_label"].upper(), user)

    def test_unfilled_placeholder_raises(self):
        with self.assertRaises(Exception):
            render("c1", {"premise": "x"})

    def test_schemas_cover_all_kinds(self):
        self.assertTrue({"c0", "c1", "c2", "c3", "c3n", "verify", "generic", "probe"} <= set(SCHEMAS))

    def test_item_files(self):
        dev, _ = config.load_items("dev")
        ev, _ = config.load_items("eval")
        self.assertEqual(len(ev), 600)
        self.assertFalse({i["orig_id"] for i in dev} & {i["orig_id"] for i in ev})
        self.assertEqual(len({i["item_id"] for i in ev}), 600)


class TestLogic(unittest.TestCase):
    def test_pair_categories(self):
        self.assertEqual(pair_category("label_change", True, True, False), "correct_update")
        self.assertEqual(pair_category("label_change", True, False, True), "insensitive")
        self.assertEqual(pair_category("label_change", True, False, False), "wrong_flip")
        self.assertEqual(pair_category("label_change", False, True, False), "variant_only")
        self.assertEqual(pair_category("paraphrase", True, True, True), "stable_correct")
        self.assertEqual(pair_category("paraphrase", False, False, True), "stable_wrong")
        self.assertEqual(pair_category("irrelevant_addition", True, False, False), "unstable")

    def test_revision_merge(self):
        item = {"premise": "A dog runs.", "hypothesis": "An animal moves."}
        base = {"label": "neutral", "explanation_text": "old", "premise_spans": ["dog"], "hypothesis_spans": [],
                "parse_status": "ok", "span_checks": []}
        keep = finalize({"parse_status": "ok", "decision": "keep", "label": "entailment"}, dict(base), item)
        self.assertEqual((keep["label"], keep["applied_revision"]), ("neutral", False))
        rev = finalize({"parse_status": "ok", "decision": "revise", "label": "entailment",
                        "premise_spans": ["A dog runs"], "hypothesis_spans": [], "explanation_text": "new"},
                       dict(base), item)
        self.assertEqual((rev["label"], rev["base_label"], rev["applied_revision"]), ("entailment", "neutral", True))
        self.assertEqual(rev["span_checks"][0]["match"], "exact")
        broken = finalize({"parse_status": "no_json", "decision": None, "label": None}, dict(base), item)
        self.assertEqual(broken["label"], "neutral")


class TestHostedEndpoint(unittest.TestCase):
    def test_base_url_key_and_model_check(self):
        from cfx.run import check_model_available, endpoint_for
        port = 18000 + os.getpid() % 1000
        server = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_server.py"), "--port", str(port)])
        try:
            cfg = {"model": "fake-model", "base_url": f"http://127.0.0.1:{port}/v1", "api_key_env": "CFX_TEST_KEY"}
            env_backup = {k: os.environ.pop(k, None) for k in ("CFX_TEST_KEY", "CFX_API_KEY")}
            with self.assertRaises(SystemExit):
                endpoint_for(cfg)
            os.environ["CFX_TEST_KEY"] = "test-key"
            ep = endpoint_for(cfg)
            self.assertTrue(ep.wait_ready(timeout=10, interval=0.2))
            self.assertEqual(ep.api_key, "test-key")
            self.assertNotIn("test-key", ep.describe())
            check_model_available(ep, "fake-model")
            with self.assertRaises(SystemExit) as cm:
                check_model_available(ep, "Qwen/Qwen2.5-72B-Instruct")
            self.assertIn("fake-model", str(cm.exception))
            out = ep.chat("fake-model", messages("c0", {"premise": "A b.", "hypothesis": "C d."}), 20,
                          {"temperature": 0})
            self.assertIn("label", out["text"])
        finally:
            os.environ.pop("CFX_TEST_KEY", None)
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v
            server.terminate()
            server.wait()


class TestEndToEnd(unittest.TestCase):
    def test_all_phases(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            tmp = Path(tmp)
            sock = tmp / "s.sock"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CFX_UDS": str(sock),
                   "CFX_OUTPUTS": str(tmp / "out"), "CFX_REPORTS": str(tmp / "reports"), "CFX_WAIT_SECONDS": "20"}
            server = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_server.py"), "--uds", str(sock)])
            try:
                for _ in range(50):
                    if sock.exists():
                        break
                    time.sleep(0.1)

                def run(*args):
                    res = subprocess.run([sys.executable, "-m", *args], env=env, cwd=ROOT,
                                         capture_output=True, text=True, timeout=600)
                    self.assertEqual(res.returncode, 0, res.stdout[-2000:] + res.stderr[-2000:])
                    return res.stdout

                run("cfx.phase", "--model", "fake", "--phases", "smoke,main,noise,revise,probe")
                runs = tmp / "out" / "runs"
                for rid in ("smoke_fake_c3", "main_fake_c0", "main_fake_c3n", "noise_fake_c3",
                            "revise_main_fake_c3_verify", "probe_main_fake_c2"):
                    self.assertTrue((runs / rid / "manifest.json").exists(), rid)
                preds = config.read_jsonl(runs / "main_fake_c3" / "predictions.jsonl")
                self.assertEqual(len(preds), 600)
                self.assertTrue(all(p["parse_status"] == "ok" for p in preds))
                self.assertTrue(any(p["format_retried"] for p in preds))
                again = run("cfx.run", "--model", "fake", "--condition", "c3", "--items", "eval",
                            "--run-id", "main_fake_c3")
                self.assertIn("done: {'ok': 600}", again)
                manifest = config.load_json(runs / "main_fake_c3" / "manifest.json")
                self.assertEqual(manifest["cache_hits"], manifest["calls"])

                out = run("cfx.analysis", "--reps", "200")
                self.assertIn("loaded", out)
                for t in ("main_results", "pairs_by_modtype", "span_validity", "comparisons", "revision", "probe",
                          "noise_floor", "efficiency"):
                    self.assertTrue((tmp / "reports" / "tables" / f"{t}.csv").exists(), t)
                self.assertTrue((tmp / "reports" / "summary.md").exists())
            finally:
                server.terminate()
                server.wait()


if __name__ == "__main__":
    unittest.main()
