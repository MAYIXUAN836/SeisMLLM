#!/usr/bin/env python3
"""
LoRA-finetuned Qwen2.5-VL-7B inference script.

Note:
- This open-source version does not hard-code any personal absolute paths.
- Please configure the base model path and LoRA path via environment
    variables or command-line arguments, or adjust the defaults below.
"""

import json
import os

from typing import Any, Dict
from assessment_format import PO_MAPPING, CT_MAPPING

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default model paths for open-source users.
# You can override them with environment variables or CLI args if needed.
BASE_MODEL_PATH = os.environ.get(
    "SEISMLLM_QWEN_BASE_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)
LORA_PATH = os.environ.get(
    "SEISMLLM_QWEN_LORA_PATH",
    os.path.join(
        ROOT_DIR,
        "Models",
        "saves(finetuned)",
        "lora",
        "earthquake_stage1_trial2",
    ),
)
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/qwen_7b_finetuned_results.json")


# Normalization maps to clamp model outputs to allowed labels
ALLOWED_VIEW = {
    "exterior only": "Exterior only",
    "interior only": "Interior only",
    "exterior and interior": "Exterior and interior",
}

ALLOWED_PO = PO_MAPPING
ALLOWED_CT = CT_MAPPING

ALLOWED_STATUS = {
    "minor/none": "Minor/none",
    "minor": "Minor/none",
    "none": "Minor/none",
    "moderate": "Moderate",
    "severe": "Severe",
    "not visible": "Not visible from the provided images",
    "not visible from the provided images": "Not visible from the provided images",
    "unknown": "Not visible from the provided images",
}

ALLOWED_DAMAGE = {
    "none": "None",
    "0-1%": "0–1%",
    "0–1%": "0–1%",
    "1-10%": "1–10%",
    "1–10%": "1–10%",
    "10-30%": "10–30%",
    "10–30%": "10–30%",
    "30-60%": "30–60%",
    "30–60%": "30–60%",
    "60-100%": "60–100%",
    "60–100%": "60–100%",
    "100%": "100%",
}

ALLOWED_FINAL = {
    "(1) inspected (green)": "(1) Inspected (Green)",
    "inspected": "(1) Inspected (Green)",
    "green": "(1) Inspected (Green)",
    "(2) restricted use (yellow)": "(2) Restricted Use (Yellow)",
    "restricted": "(2) Restricted Use (Yellow)",
    "yellow": "(2) Restricted Use (Yellow)",
    "(3) unsafe (red)": "(3) Unsafe (Red)",
    "unsafe": "(3) Unsafe (Red)",
    "red": "(3) Unsafe (Red)",
}


def _to_int(val):
    try:
        if val is None or isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            iv = int(val)
            return iv if iv >= 0 else None
        digits = "".join(ch for ch in str(val) if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None


def _norm(value, mapping):
    if not value:
        return None
    key = str(value).strip().lower()
    key_space = key.replace("-", " ")
    return mapping.get(key) or mapping.get(key_space) or value


def normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(pred)

    vt = data.get("2_view_type")
    vt_norm = _norm(vt, ALLOWED_VIEW)
    if vt_norm:
        data["2_view_type"] = vt_norm

    po = data.get("3_primary_occupation", {}) or {}
    po_cat = po.get("category")
    po_norm = _norm(po_cat, ALLOWED_PO)
    if po_norm:
        po["category"] = po_norm
        data["3_primary_occupation"] = po

    ct = data.get("4_type_of_construction", {}) or {}
    ct_cat = ct.get("category")
    ct_norm = _norm(ct_cat, ALLOWED_CT)
    if ct_norm:
        ct["category"] = ct_norm
        data["4_type_of_construction"] = ct

    sn = data.get("5_stories", {}) or {}
    sn["over_ground"] = _to_int(sn.get("over_ground"))
    sn["under_ground"] = _to_int(sn.get("under_ground"))
    data["5_stories"] = sn

    evals = data.get("6_evaluation", {}) or {}
    for k, cond in list(evals.items()):
        if isinstance(cond, dict):
            cond_status = cond.get("status")
            cond_norm = _norm(cond_status, ALLOWED_STATUS)
            if cond_norm:
                cond["status"] = cond_norm
            evals[k] = cond
    data["6_evaluation"] = evals

    dmg = data.get("7_estimated_building_damage")
    dmg_norm = _norm(dmg, ALLOWED_DAMAGE)
    if dmg_norm:
        data["7_estimated_building_damage"] = dmg_norm

    final = data.get("8_final_status", {}) or {}
    fs_status = final.get("status")
    fs_norm = _norm(fs_status, ALLOWED_FINAL)
    if fs_norm:
        final["status"] = fs_norm
        data["8_final_status"] = final

    dc_obj = data.get("DC") or data.get("dc")
    if isinstance(dc_obj, dict):
        for k, v in list(dc_obj.items()):
            st = _norm(v, ALLOWED_STATUS)
            if st:
                dc_obj[k] = st
        data["DC"] = dc_obj

    return data


class QwenLoRAInference:
    def __init__(self, base_model_path, lora_path, device="cuda"):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        print(
            f"Loading base model from {self.base_model_path} with LoRA weights {self.lora_path}..."
        )
        self.processor = AutoProcessor.from_pretrained(self.lora_path, trust_remote_code=True)

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.model = (
            PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                torch_dtype=torch.bfloat16,
            ).eval()
        )

    def generate_json(self, image_paths, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": []},
        ]

        for img_path in image_paths:
            messages[1]["content"].append({"type": "image", "image": img_path})
        messages[1]["content"].append({"type": "text", "text": user_prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = [Image.open(img) for img in image_paths]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Optimized generation params (low temperature sampling for better accuracy)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
