import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Inference"))
from cdd import get_case_damage_description


class CDDMappingTests(unittest.TestCase):
    def test_legacy_mapping_uses_existing_text_not_status_labels(self):
        record = {"6_evaluation": {
            "first": {"status": "STATUS_ONLY_SENTINEL", "description": "Column cracking."},
            "second": {"status": "STATUS_ONLY_SENTINEL", "description": "Roof displacement."},
        }}
        self.assertEqual(get_case_damage_description(record), "Column cracking. Roof displacement.")
        self.assertEqual(record["6_evaluation"]["first"]["status"], "STATUS_ONLY_SENTINEL")

    def test_explicit_cdd_takes_precedence_even_with_empty_conditions(self):
        self.assertEqual(get_case_damage_description({"CDD": "Existing case text.", "6_evaluation": {}}), "Existing case text.")

    def test_index_and_query_read_the_same_cdd_field(self):
        record = {"CDD": "Explicit text.", "9_image_descriptions": {"a.jpg": "Visible cracking."}}
        results = []
        for filename in ("index.py", "retrieve.py"):
            p = ROOT / "Inference/RAG-similar-cases" / filename
            functions = [n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {"construct_stage1_text", "construct_full_text"}]
            ns = {"get_case_damage_description": get_case_damage_description}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(p), "exec"), ns)
            results.append([ns[name](record) for name in ("construct_stage1_text", "construct_full_text")])
            self.assertTrue(all("Explicit text." in text for text in results[-1]))
        self.assertEqual(*results)

    def test_pipeline_adapter_keeps_cdd_and_dc_distinct(self):
        p = ROOT / "inference.py"
        function = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "full_json_to_stage1_fields")
        ns = {"get_case_damage_description": get_case_damage_description}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(p), "exec"), ns)
        record = {"CDD": "Wall cracking.", "6_evaluation": {
            "3_racking_damage_to_walls_other_structural_damage": {"status": "Severe", "description": "Wall cracking."},
        }}
        result = ns[function.name](record)
        self.assertEqual(result["CDD"], "Wall cracking.")
        self.assertIn("racking: Severe", result["DC"])
        self.assertEqual(result["6_evaluation"], record["6_evaluation"])


if __name__ == "__main__":
    unittest.main()
