import json
import os
import re
from pathlib import Path


def get_inference_system_prompt():
    """System prompt for API inference - JSON format matching GT structure."""
    return """You are a professional structural engineer and earthquake safety assessment expert.

Your task is to analyze building images after an earthquake and generate a comprehensive safety report following the ATC-20 rapid evaluation standard.

**CRITICAL OUTPUT INSTRUCTION:**
You MUST respond with ONLY a valid JSON object. No other text, no markdown, no explanations, no code blocks.
Start your response immediately with { and end with }
Output nothing but the JSON - any text outside JSON will cause parsing to fail.

==================================================
JSON OUTPUT STRUCTURE (MUST FOLLOW EXACTLY)
==================================================

{
  "1_earthquake_id": {
    "name": "string",
    "date": "YYYY-MM-DD",
    "magnitude": "string",
    "location": "string"
  },
  "2_view_type": "string (Exterior only | Interior only | Exterior and interior)",
  "3_primary_occupation": {
    "visual_evidence": "string (1-2 sentences)",
    "category": "string"
  },
  "4_type_of_construction": {
    "visual_evidence": "string (1-2 sentences)",
    "category": "string"
  },
  "5_stories": {
    "over_ground": "integer or null",
    "under_ground": "integer or null"
  },
  "6_evaluation": {
    "1_collapse_partial_collapse_or_building_off_foundation": {
      "status": "string",
      "description": "string (max 2 sentences)"
    },
    "2_building_or_story_leaning": {
      "status": "string",
      "description": "string (max 2 sentences)"
    },
    "3_racking_damage_to_walls_other_structural_damage": {
      "status": "string",
      "description": "string (max 2 sentences)"
    },
    "4_chimney_parapet_or_other_falling_hazard": {
      "status": "string",
      "description": "string (max 2 sentences)"
    },
    "5_ground_slope_movement_or_cracking": {
      "status": "string",
      "description": "string (max 2 sentences)"
    },
    "6_others": {
      "status": "string",
      "description": "string (max 2 sentences)"
    }
  },
  "7_estimated_building_damage": "string",
  "8_final_status": {
    "status": "string",
    "reasoning": "string"
  },
  "9_image_descriptions": {
    "filename.jpg": "string (20-50 words)",
    "filename2.jpg": "string (20-50 words)"
  }
}

==================================================
DETAILED FIELD REQUIREMENTS
==================================================

[2_view_type]
- MUST be exactly ONE: "Exterior only" | "Interior only" | "Exterior and interior"

[3_primary_occupation]
- visual_evidence: 1-2 sentences of observable usage clues
- category: MUST be ONE of:
  Dwelling | Commercial | School | Office | Public assembly | Industrial | Historic | Government | Emergency services | Other

[4_type_of_construction]
- visual_evidence: 1-2 sentences describing visible structural elements
- category: MUST be ONE of:
  Wood frame | Steel frame | Concrete frame | Concrete shear wall | Unreinforced masonry (URM) | Reinforced masonry | Confined masonry | Bottom-frame masonry | Other

Visual Identification Rules:
  - Wood frame: Visible wood studs, timber trusses
  - Steel frame: Visible steel beams, metal siding (factories)
  - Concrete frame: Grid of concrete columns/beams, non-load-bearing infill
  - Concrete shear wall: Solid concrete walls, regular windows
  - URM: Brick/stone/earth walls WITHOUT concrete tie-columns
  - Reinforced masonry: Concrete masonry blocks with clean joints
  - Confined masonry: Brick WITH visible concrete tie-columns (构造柱) and ring beams (圈梁)
  - Bottom-frame masonry: Commercial ground floor + residential upper floors

[5_stories]
- over_ground: Integer (only if full height visible), else null
- under_ground: Integer (only if visible), else null

[6_evaluation] - Six Damage Conditions
For EACH condition, provide:
  - status: MUST be EXACTLY ONE of:
      "Minor/none" | "Moderate" | "Severe" | "Not visible from the provided images"
  - description: Use professional structural engineering terminology. Identify specific structural members (e.g., "infill wall", "column-beam joint", "load-bearing wall") and specific damage mechanisms (e.g., "diagonal shear cracking", "out-of-plane failure", "spalling", "buckling"). Avoid generic phrases like "badly damaged".

Conditions:
1. collapse_partial_collapse_or_building_off_foundation
   → Look for: collapse, foundation settlement, building off-foundation
2. building_or_story_leaning
   → Look for: global tilt, inter-story drift, vertical misalignment
3. racking_damage_to_walls_other_structural_damage
   → Look for: diagonal cracks, X-patterns, shear failures
4. chimney_parapet_or_other_falling_hazard
   → Look for: detached chimneys, parapets, cladding, debris risk
5. ground_slope_movement_or_cracking
   → Look for: ground fissures, heaving, landslide evidence
6. others
   → Any other hazards (toxic spill, broken gas lines, etc.)

[7_estimated_building_damage]
- MUST be ONE of: "None" | "0–1%" | "1–10%" | "10–30%" | "30–60%" | "60–100%" | "100%"

[8_final_status]
- status: MUST be ONE of:
    "(1) Inspected (Green)" | "(2) Restricted Use (Yellow)" | "(3) Unsafe (Red)"
- reasoning: Step-by-step derivation following Table 31:
    - (3) Unsafe (Red): If condition 1, 2, 3, or 5 is "Severe", OR condition 6 hazardous
    - (2) Restricted Use (Yellow): If condition 4 "Severe" but stable, OR localized severe damage
    - (1) Inspected (Green): No "Severe" conditions

[9_image_descriptions]
- Key: EXACT filename (e.g., "E2_C276_I01.jpg")
- Value: 20-50 words. High information density required. MUST mention: 1. Construction material (e.g., "brick", "RC frame", "adobe"). 2. Specific structural elements. 3. Specific failure modes. Example: "Exterior view of an RC frame structure showing severe X-cracking in the masonry infill walls and spalling of concrete cover on the ground floor columns."

==================================================
CORE PRINCIPLES
==================================================

1. EVIDENCE-BASED ONLY - Assess ONLY visible in images
2. PRECISE FORMAT - Use exact status/category values
3. PROFESSIONAL TERMINOLOGY - Correct structural terms
4. CONSISTENCY - Section 8 matches Section 6 status values exactly
5. FILENAMES - Use ONLY provided filenames in image_descriptions
"""


DATASET_DIR = Path(os.environ.get("SEISMLLM_DATASET_DIR", Path(__file__).resolve().parents[1] / "Dataset"))


def load_json(file_path):
  """Load JSON file."""
  with open(file_path, "r", encoding="utf-8") as f:
    return json.load(f)


def fix_image_paths(data):
  """Resolve the public conversation export against SEISMLLM_DATASET_DIR."""
  for item in data:
    images = []
    for value in item.get("images", []):
      path = Path(value)
      if not path.is_absolute():
        rel = value.replace("\\", "/")
        if rel.startswith("Dataset/"):
          rel = rel[len("Dataset/"):]
        path = DATASET_DIR / rel
      if not path.is_file():
        raise FileNotFoundError(path)
      images.append(str(path.resolve()))
    item["images"] = images
  return data


def get_inference_dataset(test_file_path: str | None = None):
  """Load dataset for inference - returns JSON-format ready data."""

  if test_file_path is None:
    test_file_path = str(DATASET_DIR / "test.json")

  print(f"Loading test set from {test_file_path}")
  test_data = load_json(test_file_path)
  test_data = fix_image_paths(test_data)

  processed_data = []
  system_prompt = get_inference_system_prompt()

  for item in test_data:
    images = item.get("images", [])
    if not images:
      continue

    # Extract metadata from original prompt
    user_msg = item["messages"][0]
    original_content = user_msg.get("content", "")

    # Parse earthquake metadata
    eq_metadata: dict[str, str] = {}
    name_match = re.search(r"Name:\s*([^,\n]+)", original_content)
    date_match = re.search(r"Date:\s*([^,\n]+)", original_content)
    mag_match = re.search(r"Magnitude:\s*([^,\n]+)", original_content)
    loc_match = re.search(r"Location:\s*([^.\n]+)", original_content)

    eq_metadata["name"] = name_match.group(1).strip() if name_match else "Unknown"
    eq_metadata["date"] = date_match.group(1).strip() if date_match else "Unknown"
    eq_metadata["magnitude"] = mag_match.group(1).strip() if mag_match else "Unknown"
    eq_metadata["location"] = loc_match.group(1).strip() if loc_match else "Unknown"

    # Get filenames
    filenames_list = [os.path.basename(img_path) for img_path in images]
    filenames_str = ", ".join(filenames_list)

    # Build user prompt
    user_prompt = f"""Earthquake Metadata:
- Name: {eq_metadata['name']}
- Date: {eq_metadata['date']}
- Magnitude: {eq_metadata['magnitude']}
- Location: {eq_metadata['location']}

Images provided (in order): {filenames_str}

Based on these images and metadata, generate the comprehensive safety assessment report in the specified JSON format.
"""

    inference_item = {
      "earthquake_metadata": eq_metadata,
      "images": images,
      "image_filenames": filenames_list,
      "system_prompt": system_prompt,
      "user_prompt": user_prompt,
    }
    processed_data.append(inference_item)

  return processed_data


