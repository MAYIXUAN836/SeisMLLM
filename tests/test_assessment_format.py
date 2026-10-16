import copy
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Inference"))
from assessment_format import CONDITIONS, CT_CATEGORIES, PO_CATEGORIES, parse_assessment_output
from utils_qwen_finetune import extract_earthquake_metadata, get_inference_system_prompt
from train import prepare_data


def report(tag="DC", occupation="Offices"):
    conditions = "; ".join(
        f"{name}: Not visible from the provided images (No view of the ground.)" if name == "ground"
        else f"{name}: Minor/none (Column cracking (local); no collapse.)"
        for name in CONDITIONS
    )
    return (
        "### Structured Building Annotation\n[IS]: Exterior only\n"
        f"[PO]: Evidence: Office signage.; Category: {occupation}\n"
        "[CT]: Evidence: Concrete panels.; Category: Tilt-up concrete\n"
        f"[SN]: Over ground: None\n[{tag}]: {conditions}\n"
        "[SID]:\n- first.jpg: Concrete panels with cracking.\n- second.jpg: A closer view."
    )


class AssessmentFormatTests(unittest.TestCase):
    def test_training_report_preserves_condition_text_and_unknown_visibility(self):
        parsed = parse_assessment_output(report())
        self.assertEqual(len(parsed["6_evaluation"]), 6)
        self.assertEqual(parsed["6_evaluation"][CONDITIONS["collapse"]]["description"], "Column cracking (local); no collapse.")
        self.assertEqual(parsed["6_evaluation"][CONDITIONS["ground"]]["status"], "Not visible from the provided images")
        self.assertIsNone(parsed["5_stories"]["over_ground"])
        self.assertEqual(parsed["9_image_descriptions"]["second.jpg"], "A closer view.")
        self.assertNotIn("8_final_status", parsed)
        self.assertNotIn("7_estimated_building_damage", parsed)
        self.assertNotIn("CDD", parsed)

    def test_old_cdd_tag_means_the_same_structured_condition_section(self):
        self.assertEqual(parse_assessment_output(report()), parse_assessment_output(report("CDD")))

    def test_json_predictions_keep_existing_fields_and_posting(self):
        prediction = parse_assessment_output(report())
        prediction["8_final_status"] = {"status": "(2) Restricted Use (Yellow)", "reasoning": "Falling hazard."}
        text = "```json\n" + json.dumps(prediction) + "\n```"
        self.assertEqual(parse_assessment_output(text), prediction)

    def test_incomplete_json_does_not_fall_back_to_a_partial_report(self):
        with self.assertRaises(ValueError):
            parse_assessment_output('{"6_evaluation": ')

    def test_missing_condition_is_not_filled_with_a_low_severity_label(self):
        text = report().replace("others: Minor/none (Column cracking (local); no collapse.)", "")
        with self.assertRaises(ValueError):
            parse_assessment_output(text)

    def test_narrative_cdd_cannot_stand_in_for_six_condition_ratings(self):
        text = report("CDD")
        start = text.index("[CDD]:")
        end = text.index("[SID]:")
        with self.assertRaises(ValueError):
            parse_assessment_output(text[:start] + "[CDD]: Cracks are visible.\n" + text[end:])

    def test_duplicate_condition_is_rejected(self):
        text = report().replace("[SID]:", "; collapse: Severe (Duplicate.)\n[SID]:")
        with self.assertRaises(ValueError):
            parse_assessment_output(text)

    def test_office_alias_uses_existing_released_spelling(self):
        self.assertEqual(parse_assessment_output(report(occupation="Office"))["3_primary_occupation"]["category"], "Offices")

    def test_pending_occupation_label_is_not_reclassified(self):
        self.assertEqual(parse_assessment_output(report(occupation="Bottom-frame masonry"))["3_primary_occupation"]["category"], "Bottom-frame masonry")
        self.assertNotIn("Bottom-frame masonry", PO_CATEGORIES)

    def test_prompt_uses_shared_category_vocabulary(self):
        prompt = get_inference_system_prompt()
        self.assertIn(" | ".join(PO_CATEGORIES), prompt)
        self.assertIn(" | ".join(CT_CATEGORIES), prompt)
        self.assertIn("Not visible from the provided images", CT_CATEGORIES)
        self.assertNotIn("{po_categories}", prompt)

    def test_event_context_is_read_from_user_prompt(self):
        context = extract_earthquake_metadata("- Name: Example Event\n- Date: 2020-01-01\n- Magnitude: Mw 6.0\n- Location: Example city\n")
        self.assertEqual(context["name"], "Example Event")
        self.assertEqual(context["magnitude"], "Mw 6.0")

    def test_training_preparation_checks_adapter_but_preserves_pending_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["first.jpg", "second.jpg"]:
                (root / name).write_bytes(b"image-path fixture")
            rows = [{"images": ["Dataset/first.jpg", "Dataset/second.jpg"], "messages": [
                {"role": "user", "content": "<image><image>Assess the building."},
                {"role": "assistant", "content": report(occupation="Bottom-frame masonry")},
            ]}]
            original = copy.deepcopy(rows)
            source = root / "train.json"
            source.write_text(json.dumps(rows))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                prepared, count = prepare_data(root, root / "output")
            self.assertEqual(count, 1)
            self.assertEqual(json.loads(source.read_text()), original)
            result = json.loads((prepared / "train.json").read_text())[0]
            self.assertEqual(result["messages"], original[0]["messages"])
            self.assertTrue(all(Path(p).is_absolute() for p in result["images"]))
            self.assertIn("1 training cases", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
