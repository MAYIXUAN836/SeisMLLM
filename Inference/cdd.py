"""Resolve the released CDD text without treating DC ratings as descriptions."""


def get_case_damage_description(record):
    """Use explicit CDD, or map the existing condition descriptions to CDD."""
    value = record.get("CDD")
    if isinstance(value, str) and value.strip():
        return value.strip()
    conditions = record.get("6_evaluation", {})
    return " ".join(
        item["description"] for item in conditions.values()
        if isinstance(item, dict) and isinstance(item.get("description"), str)
        and item["description"]
    )
