"""Shared category vocabulary and adapters for released assessment reports."""

import json
import re


PO_CATEGORIES = (
    "Dwelling", "Other residential", "Commercial", "School", "Offices",
    "Public assembly", "Industrial", "Historic", "Government", "Emergency services", "Other",
)
CT_CATEGORIES = (
    "Wood frame", "Steel frame", "Concrete frame", "Concrete shear wall",
    "Unreinforced masonry (URM)", "Reinforced masonry", "Confined masonry",
    "Bottom-frame masonry", "Tilt-up concrete", "Other", "Not visible from the provided images",
)
PO_MAPPING = {value.lower(): value for value in PO_CATEGORIES}
PO_MAPPING.update({"office": "Offices", "residential": "Dwelling", "mixed-use": "Other", "mixed use": "Other"})
CT_MAPPING = {value.lower(): value for value in CT_CATEGORIES}
CT_MAPPING.update({
    "reinforced concrete frame": "Concrete frame", "rc frame": "Concrete frame",
    "urm": "Unreinforced masonry (URM)", "unreinforced masonry": "Unreinforced masonry (URM)",
    "masonry": "Unreinforced masonry (URM)", "bottom frame masonry": "Bottom-frame masonry",
    "tilt up concrete": "Tilt-up concrete",
})
CONDITIONS = {
    "collapse": "1_collapse_partial_collapse_or_building_off_foundation",
    "building": "2_building_or_story_leaning",
    "racking": "3_racking_damage_to_walls_other_structural_damage",
    "chimney": "4_chimney_parapet_or_other_falling_hazard",
    "ground": "5_ground_slope_movement_or_cracking",
    "others": "6_others",
}
CONDITION_STATUSES = (
    "Minor/none", "Moderate", "Severe", "Not visible from the provided images",
)


def _category(value, mapping):
    return mapping.get(value.strip().lower(), value.strip())


def _category_section(value, mapping):
    match = re.fullmatch(r"Evidence:\s*(.*);\s*Category:\s*(.+)", value, re.S)
    if not match:
        raise ValueError("Expected Evidence: ...; Category: ...")
    return {"visual_evidence": match[1].strip(), "category": _category(match[2], mapping)}


def parse_assessment_output(text):
    """Accept JSON or the released tagged report; never infer missing safety labels.

    Older reports used [CDD] for the same six ratings/descriptions now in [DC].
    A narrative-only CDD is not a substitute for these six condition objects.
    """
    text = text.strip()
    if "{" in text:
        result, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        if not isinstance(result, dict):
            raise ValueError("Assessment JSON must be an object")
        return result

    fields = dict(re.findall(r"^\[([A-Z]+)\]:[ \t]*(.*?)(?=^\[[A-Z]+\]:|\Z)", text, re.M | re.S))
    for key in ("IS", "PO", "CT", "SN", "SID"):
        if key not in fields:
            raise ValueError(f"Missing report field [{key}]")
    damage = fields.get("DC", fields.get("CDD", "")).strip()
    matches = list(re.finditer(r"(?:^|;\s*)(collapse|building|racking|chimney|ground|others)\s*:\s*", damage))
    evaluations = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(damage)
        value = damage[match.end():end].strip()
        status, separator, description = value.partition(" (")
        if status not in CONDITION_STATUSES or (separator and not description.endswith(")")):
            raise ValueError(f"Invalid or incomplete condition: {match[1]}")
        key = CONDITIONS[match[1]]
        if key in evaluations:
            raise ValueError(f"Duplicate condition: {match[1]}")
        evaluations[key] = {"status": status, "description": description[:-1] if separator else ""}
    if len(evaluations) != len(CONDITIONS):
        raise ValueError("Expected all six damage conditions")

    stories = {}
    for label, key in [("Over ground", "over_ground"), ("Under ground", "under_ground")]:
        match = re.search(rf"{label}:\s*(\d+|None|null|Unknown)\b", fields["SN"], re.I)
        stories[key] = int(match[1]) if match and match[1].isdigit() else None

    descriptions = {}
    sid = fields["SID"].strip()
    caption_matches = list(re.finditer(r"^-[ \t]+([^:\n]+):[ \t]*", sid, re.M))
    if sid and not caption_matches:
        raise ValueError("Expected filename-keyed [SID] descriptions")
    for index, match in enumerate(caption_matches):
        end = caption_matches[index + 1].start() if index + 1 < len(caption_matches) else len(sid)
        descriptions[match[1].strip()] = sid[match.end():end].strip()

    return {
        "2_view_type": fields["IS"].strip(),
        "3_primary_occupation": _category_section(fields["PO"].strip(), PO_MAPPING),
        "4_type_of_construction": _category_section(fields["CT"].strip(), CT_MAPPING),
        "5_stories": stories,
        "6_evaluation": evaluations,
        "9_image_descriptions": descriptions,
    }
