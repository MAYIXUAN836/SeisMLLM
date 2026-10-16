"""Enforce training-only retrieval using the released image-file identities."""

import json
import os
from pathlib import Path


def image_names(record):
    """Ignore legacy folder/case numbering; keep complete image filenames."""
    paths = record.get("images") or record.get("dataset_images")
    if not paths:
        raise ValueError("Image references are required for reliable case identity")
    return frozenset(str(p).replace("\\", "/").rsplit("/", 1)[-1] for p in paths)


def dataset_directory():
    return Path(os.environ.get("SEISMLLM_DATASET_DIR", Path(__file__).resolve().parents[2] / "Dataset"))


def load_splits(dataset_dir=None):
    """Load explicit train/test manifests; reject overlapping image identities."""
    root = Path(dataset_dir) if dataset_dir is not None else dataset_directory()
    splits = {}
    used_images = set()
    for name in ("train", "test"):
        records = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        cases = set()
        for record in records:
            images = image_names(record)
            if images & used_images:
                raise ValueError("Repeated image identity in the train/test manifests")
            cases.add(images)
            used_images.update(images)
        if not cases:
            raise ValueError(f"The {name} manifest is empty")
        splits[name] = cases
    return splits


def select_training_records(records, splits):
    """Only cases matching train.json enter the index; test cases are excluded."""
    selected = []
    seen = set()
    for record in records:
        images = image_names(record)
        if images in splits["test"]:
            continue
        if images not in splits["train"]:
            raise ValueError("Source case does not match a complete released train/test case")
        if images in seen:
            raise ValueError("Duplicate training case in the index source")
        seen.add(images)
        selected.append(record)
    if not selected:
        raise ValueError("No training cases remain for indexing")
    return selected


def validate_index(kb, splits, evaluation_records=()):
    """Recheck loaded indices against current splits and actual evaluation inputs."""
    metadata = kb["metadata"]
    if not metadata or len(kb["embeddings"]) != len(metadata):
        raise ValueError("Empty or misaligned similar-case index")
    if kb["bm25_model"].corpus_size != len(metadata):
        raise ValueError("BM25 and case metadata are misaligned")
    evaluation_images = set()
    for record in evaluation_records:
        evaluation_images.update(image_names(record))
    seen = set()
    for meta in metadata:
        images = image_names(meta["original_data"])
        if images not in splits["train"]:
            raise ValueError("Index contains a test or unregistered case; rebuild from training cases")
        if images & evaluation_images:
            raise ValueError("Index overlaps the evaluation images")
        if images in seen:
            raise ValueError("Duplicate case in the loaded index")
        seen.add(images)
