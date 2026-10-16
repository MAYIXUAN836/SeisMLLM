import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Inference" / "RAG-similar-cases"))
from split_guard import image_names, load_splits, select_training_records, validate_index


class SplitGuardTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.train = [
            {"images": ["Dataset/new_case_1/train_a.jpg", "Dataset/new_case_1/train_b.jpg"]},
            {"images": ["Dataset/new_case_2/train_c.jpg"]},
        ]
        self.test = [{"images": ["Dataset/new_case_3/test_a.jpg"]}]
        for name, records in [("train", self.train), ("test", self.test)]:
            (self.root / f"{name}.json").write_text(json.dumps(records))
        self.splits = load_splits(self.root)
        self.source = [
            {"case_id": "legacy_99", "dataset_images": ["old/folder/train_a.jpg", "old/folder/train_b.jpg"]},
            {"case_id": "legacy_1", "dataset_images": ["old/folder/train_c.jpg"]},
            {"case_id": "legacy_2", "dataset_images": ["old/folder/test_a.jpg"]},
        ]

    def index(self, records):
        return {
            "metadata": [{"id": r["case_id"], "original_data": r} for r in records],
            "embeddings": [[1.0] for _ in records],
            "bm25_model": SimpleNamespace(corpus_size=len(records)),
        }

    def test_build_selects_training_cases_despite_renumbered_folders(self):
        selected = select_training_records(self.source, self.splits)
        self.assertEqual([r["case_id"] for r in selected], ["legacy_99", "legacy_1"])
        self.assertEqual(image_names(selected[0]), image_names(self.train[0]))

    def test_overlap_between_manifests_is_rejected(self):
        (self.root / "test.json").write_text(json.dumps([{"images": ["elsewhere/train_a.jpg"]}]))
        with self.assertRaises(ValueError):
            load_splits(self.root)

    def test_unknown_or_mixed_source_case_is_rejected(self):
        for paths in [["unknown.jpg"], ["train_a.jpg", "test_a.jpg"]]:
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                select_training_records([{"images": paths}], self.splits)

    def test_training_only_index_loads_for_held_out_inputs(self):
        validate_index(self.index(self.source[:2]), self.splits, self.test)

    def test_contaminated_existing_index_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_index(self.index(self.source), self.splits)

    def test_actual_input_overlap_is_rejected_even_for_partial_views(self):
        with self.assertRaises(ValueError):
            validate_index(self.index(self.source[:2]), self.splits, [{"images": ["renamed_folder/train_b.jpg"]}])

    def test_duplicate_index_case_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_index(self.index([self.source[0], self.source[0]]), self.splits)

    def test_identity_cannot_fall_back_to_an_unverified_case_number(self):
        with self.assertRaises(ValueError):
            image_names({"case_id": "case_0001"})

    def test_misaligned_index_is_rejected(self):
        kb = self.index(self.source[:2])
        kb["embeddings"].pop()
        with self.assertRaises(ValueError):
            validate_index(kb, self.splits)


if __name__ == "__main__":
    unittest.main()
