# ruff: noqa: E402  (torch imports are skipped first when torch is missing)
"""Meaningful offline tests: gradients, checkpoint parity, chronology and isolation."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# CI installs requirements-ci.txt (no torch); these tests run where torch is installed
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BertConfig, BertModel, BertTokenizerFast, DebertaV2Config, DebertaV2Model

import prototype_models as pm
import training_recovery as recovery
import run_prototypes as runner
from run_prototypes import apply_pipeline, select_pipeline


def frame(n, start):
    rng = np.random.default_rng(25)
    out = pd.DataFrame(rng.normal(size=(n, len(pm.META))), columns=pm.META)
    out["id"] = [f"{start}_{i}" for i in range(n)]
    out["created_at"] = pd.date_range(start, periods=n, freq="D")
    out["title"] = ["why did the chicken" if i % 2 else "the dad said hello" for i in range(n)]
    out["body"] = "because the joke was funny"
    out["label"] = np.arange(n) % 4 == 0
    out["metric"] = rng.normal(size=n)
    out["metric_pct"] = rng.random(n)
    return out


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / "tiny"
        cls.source.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "why", "did", "the", "chicken", "dad", "said", "hello",
                 "because", "joke", "was", "funny"]
        (cls.source / "vocab.txt").write_text("\n".join(vocab))
        cls.tokenizer = BertTokenizerFast(vocab_file=str(cls.source / "vocab.txt"))
        cls.tokenizer.save_pretrained(cls.source)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)).save_pretrained(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def spec(self, **kwargs):
        return pm.Spec(name="tiny", source=str(self.source), epochs=1, batch_size=3, accumulation=2, **kwargs)

    def test_interaction_reads_both_branches_and_backpropagates(self):
        spec = self.spec(architecture="interaction", max_length=8)
        net = pm.Network(spec)
        d = frame(3, "2020-01-01")
        first, second = pm.encode(d, self.tokenizer, spec)
        a = pm.collate(first, [0, 1, 2], self.tokenizer, "cpu")
        b = pm.collate(second, [0, 1, 2], self.tokenizer, "cpu")
        net.eval()
        p = net(a, torch.zeros(3, len(pm.META)), b)
        changed = self.tokenizer(["dad"] * 3, padding=True, return_tensors="pt")
        self.assertFalse(torch.allclose(p, net(a, torch.zeros(3, len(pm.META)), changed)))
        p.sum().backward()
        self.assertGreater(net.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.head[0].weight.grad.abs().sum().item(), 0)

    def test_punchline_tail_and_missing_body(self):
        d = frame(2, "2020-01-01")
        d.loc[0, "body"] = "the " * 100 + "funny"
        d.loc[1, "body"] = None
        _, body = pm.encode(d, self.tokenizer, self.spec(architecture="interaction", max_length=8, body_chars=None))
        self.assertEqual(len(body["input_ids"][0]), 8)
        self.assertIn(self.tokenizer.convert_tokens_to_ids("funny"), body["input_ids"][0])
        self.assertEqual(self.tokenizer.truncation_side, "right")
        self.assertGreater(len(body["input_ids"][1]), 0)

    def test_deberta_checkpointing_accumulates_gradients(self):
        source = self.root / "tiny_deberta"
        model = DebertaV2Model(DebertaV2Config(vocab_size=16, hidden_size=16,
                 num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                 relative_attention=True, pos_att_type=["p2c", "c2p"], share_att_key=True))
        model.save_pretrained(source)
        net = pm.Network(replace(self.spec(), source=str(source), checkpointing=True))
        net.train()
        tokens = {"input_ids": torch.tensor([[2, 5, 6, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        for _ in range(2):
            pred = net(tokens, torch.zeros(1, len(pm.META)))
            pred.square().mean().backward()
        self.assertGreater(net.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)

    def test_remainder_batches_cover_all_rows_once(self):
        lengths = np.arange(307) % 10
        for rng in (None, np.random.default_rng(42)):
            batches = pm.batch_order(lengths, 7, rng)
            self.assertTrue(all(0 < len(b) <= 7 for b in batches))
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(307))

    def test_train_save_reload_and_cache_invalidation(self):
        tr, val = frame(13, "2020-01-01"), frame(8, "2024-01-01")
        path = self.root / "run"
        pm.train(self.spec(), tr, val, path, device="cpu")
        pred = pm.predict_checkpoint(path, val, device="cpu")
        np.testing.assert_allclose(pred, np.load(path / "val_predictions.npy"), rtol=1e-5, atol=1e-5)
        changed = val.copy()
        changed["metric"] = 10000
        changed["label"] = ~changed.label
        np.testing.assert_allclose(pred, pm.predict_checkpoint(path, changed, device="cpu"))
        posting_only = val[["title", "body", *pm.META]]
        np.testing.assert_allclose(pred, pm.predict_checkpoint(path, posting_only, device="cpu"))
        self.assertEqual(pm.train(self.spec(), tr, val, path, device="cpu"), path)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            pm.train(self.spec(), tr, changed, path, device="cpu")
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            pm.train(self.spec(), val, tr, self.root / "bad", device="cpu")

    def test_rank_transform_is_batch_invariant(self):
        reference = np.array([0, 1, 1, 3, 4])
        whole = pm.empirical_ranks(reference, [2, 100, -1])
        single = pm.empirical_ranks(reference, [2])
        self.assertEqual(whole[0], single[0])
        np.testing.assert_allclose(whole, [.6, 1, 0])

    def test_interrupted_resume_matches_uninterrupted_training(self):
        tr, val = frame(19, "2020-01-01"), frame(8, "2024-01-01")
        spec = replace(self.spec(), epochs=2)
        whole, partial = self.root / "whole", self.root / "partial"
        pm.train(spec, tr, val, whole, device="cpu", checkpoint_seconds=0)
        real_save = recovery.atomic_save
        calls = 0

        def interrupt_after_update(value, path):
            nonlocal calls
            real_save(value, path)
            if Path(path).name == "resume.pt":
                calls += 1
                if calls == 3:  # midway through epoch one, after two complete updates
                    raise RuntimeError("simulated GPU interruption")

        with patch.object(recovery, "atomic_save", side_effect=interrupt_after_update):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                pm.train(spec, tr, val, partial, device="cpu", checkpoint_seconds=0)
        with self.assertRaisesRegex(ValueError, "use --resume"):
            pm.train(spec, tr, val, partial, device="cpu")
        pm.train(spec, tr, val, partial, device="cpu", resume=True, checkpoint_seconds=0)
        np.testing.assert_array_equal(np.load(whole / "val_predictions.npy"), np.load(partial / "val_predictions.npy"))
        a = torch.load(whole / "resume.pt", weights_only=False)
        b = torch.load(partial / "resume.pt", weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual(b["epoch"], 2)
        for key, state in a["optimizer"]["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(value, b["optimizer"]["state"][key][field], rtol=0, atol=0)

    def test_atomic_snapshot_write_failure_keeps_previous(self):
        path = self.root / "atomic.pt"
        recovery.atomic_save({"step": 12}, path)
        with patch.object(recovery.torch, "save", side_effect=OSError("disk interrupted")):
            with self.assertRaises(OSError):
                recovery.atomic_save({"step": 13}, path)
        self.assertEqual(torch.load(path, weights_only=True), {"step": 12})
        self.assertFalse(path.with_name("atomic.pt.tmp").exists())

    def test_legacy_epoch_warm_start_preserves_previous_results(self):
        tr, val = frame(13, "2020-01-01"), frame(8, "2024-01-01")
        path = self.root / "legacy"
        pm.train(self.spec(), tr, val, path, device="cpu")
        spec = replace(self.spec(), epochs=3)
        original = json.loads((path / "identity.json").read_text())
        original["spec"]["epochs"] = 3
        original["code_sha256"] = next(iter(pm.LEGACY_TRAINING_HASHES))
        original.pop("recovery_sha256")
        pm.write_json(path / "identity.json", original)
        for name in ("complete.json", "resume.pt", "best_checkpoint.pt"):
            (path / name).unlink()
        interrupted = self.root / "legacy_interrupted"
        shutil.copytree(path, interrupted)
        pm.train(spec, tr, val, path, device="cpu", resume=True)
        result = json.loads((path / "recovery.json").read_text())
        self.assertEqual(result["from_epoch"], 1)
        self.assertEqual(json.loads((path / "identity.before_recovery.json").read_text()), original)
        self.assertEqual([r["epoch"] for r in json.loads((path / "history.json").read_text())], [1, 2, 3])
        saved = torch.load(path / "resume.pt", weights_only=False)
        self.assertEqual(saved["warm_start_epoch"], 1)
        real_save, count = recovery.atomic_save, 0

        def stop_during_recovered_epoch(value, target):
            nonlocal count
            real_save(value, target)
            if Path(target).name == "resume.pt":
                count += 1
                if count == 2:
                    raise RuntimeError("another interruption")

        with patch.object(recovery, "atomic_save", side_effect=stop_during_recovered_epoch):
            with self.assertRaisesRegex(RuntimeError, "another interruption"):
                pm.train(spec, tr, val, interrupted, device="cpu", resume=True, checkpoint_seconds=0)
        pm.train(spec, tr, val, interrupted, device="cpu", resume=True)
        resumed = torch.load(interrupted / "resume.pt", weights_only=False)
        self.assertEqual(saved["scheduler"], resumed["scheduler"])
        for key in saved["model"]:
            torch.testing.assert_close(saved["model"][key], resumed["model"][key], rtol=0, atol=0)

    def test_completed_legacy_run_can_be_reused_without_retraining(self):
        tr, val = frame(13, "2020-01-01"), frame(8, "2024-01-01")
        path = self.root / "legacy_complete"
        pm.train(self.spec(), tr, val, path, device="cpu")
        identity = json.loads((path / "identity.json").read_text())
        identity["code_sha256"] = next(iter(pm.LEGACY_TRAINING_HASHES))
        identity.pop("recovery_sha256")
        pm.write_json(path / "identity.json", identity)
        with patch.object(pm, "Network", side_effect=AssertionError("must reuse")):
            self.assertEqual(pm.train(self.spec(), tr, val, path, device="cpu", resume=True), path)
        changed = tr.copy()
        changed.loc[0, "metric"] = 1000
        with self.assertRaisesRegex(ValueError, "identity changed"):
            pm.train(self.spec(), changed, val, path, device="cpu", resume=True)

    def test_blend_gate_cannot_rechoose_weights(self):
        val = frame(366, "2024-01-01")
        rng = np.random.default_rng(2)
        preds = [rng.normal(size=len(val)) for _ in range(4)]
        selected, _, refs = select_pipeline(val, *preds[:3], {"interaction": preds[3]})
        changed = val.copy()
        gate = changed.created_at >= pd.Timestamp("2024-07-01")
        changed.loc[gate, "label"] = ~changed.loc[gate, "label"]
        updated, _, _ = select_pipeline(changed, *preds[:3], {"interaction": preds[3]})
        self.assertEqual(selected["blend_specs"], updated["blend_specs"])
        applied = apply_pipeline(selected, refs, *preds[:3], {"interaction": preds[3]})
        self.assertTrue(all(np.isfinite(p).all() for p in applied.values()))
        first = apply_pipeline(selected, refs, *[p[:1] for p in preds[:3]], {"interaction": preds[3][:1]})
        for name in applied:
            self.assertAlmostEqual(first[name][0], applied[name][0])

    def test_identical_paired_bootstrap(self):
        d = frame(100, "2025-01-01")
        scores = np.arange(100)
        result = pm.bootstrap_delta(d, scores, scores, repetitions=10)
        self.assertEqual(result["ap_delta_95ci"], [0, 0])
        self.assertEqual(result["precision_top10_delta_95ci"], [0, 0])

    def test_report_freezes_selection_before_test_inference(self):
        root = self.root / "report_test"
        run = root / "data/full"
        reference = root / "reference"
        run.mkdir(parents=True)
        reference.mkdir()
        val, test = frame(366, "2024-01-01"), frame(100, "2025-01-01")
        rng = np.random.default_rng(4)
        base_val = [rng.random(len(val)) for _ in range(3)]
        original = test[["id", "label"]].copy()
        for name in ("finalist_metadata_blend", "finalist_ensemble", "metadata_hgb"):
            original[name] = rng.random(len(test))
        original.to_parquet(reference / "test_predictions.parquet", index=False)
        pm.write_json(run / "protocol.json", {"synthetic": True})
        paths, predictions = {}, {}
        for name in ("interaction", "deberta"):
            paths[name] = run / name
            paths[name].mkdir()
            for file in ("identity.json", "complete.json", "best.json"):
                pm.write_json(paths[name] / file, {"synthetic": True})
            val[["id", "created_at"]].to_parquet(paths[name] / "val_rows.parquet", index=False)
            np.save(paths[name] / "val_predictions.npy", rng.random(len(val)))
            predictions[name] = rng.random(len(test))

        def infer(path, _):
            self.assertTrue((run / "selection.json").exists())
            return predictions[Path(path).name]

        with patch.object(runner, "ROOT", root), patch.object(runner, "reference_predictions", return_value=base_val), \
                patch.object(pm, "predict_checkpoint", side_effect=infer), \
                patch.object(pm, "bootstrap_delta", return_value={"tested_separately": True}):
            runner.report(run, reference, {"val": val, "test": test}, paths)
        dest = root / "results/prototypes/full"
        metrics = pd.read_csv(dest / "metrics.csv")
        self.assertEqual(len(metrics), 6)
        self.assertTrue((dest / "comparison.png").stat().st_size > 1000)
        self.assertTrue((dest / "SUMMARY.md").exists())
        # Corrupting cached test order must fail before a result is reported.
        original.iloc[::-1].to_parquet(reference / "test_predictions.parquet", index=False)
        with patch.object(runner, "ROOT", root), patch.object(runner, "reference_predictions", return_value=base_val):
            with self.assertRaisesRegex(ValueError, "not aligned"):
                runner.report(run, reference, {"val": val, "test": test}, paths)


if __name__ == "__main__":
    unittest.main()
