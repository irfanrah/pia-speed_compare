import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image


ANNOTATION_ENV = "CLIP_EBC_ANNOTATION_PATH"
IMAGE_ROOT_ENV = "CLIP_EBC_IMAGE_ROOT"
MAX_MAE_ENV = "CLIP_EBC_MAX_MAE"
POINT_LABEL_ENV = "CLIP_EBC_POINT_LABEL"


def _resolve_image_path(image_path: str, annotation_path: Path) -> Path:
    image_candidate = Path(image_path)
    if image_candidate.is_absolute():
        return image_candidate

    image_root = os.getenv(IMAGE_ROOT_ENV)
    if image_root:
        return Path(image_root) / image_candidate

    return annotation_path.parent / image_candidate


def _get_first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _load_xml_annotation_records(annotation_path: Path) -> List[Tuple[Path, float]]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    point_label = os.getenv(POINT_LABEL_ENV, "people")

    records: List[Tuple[Path, float]] = []
    for image_elem in root.findall(".//image"):
        image_name = image_elem.get("name")
        if not image_name:
            raise ValueError("Each <image> entry in the XML must include a name attribute.")

        gt_count = 0
        for point_elem in image_elem.findall("points"):
            label = point_elem.get("label")
            if label == point_label:
                gt_count += 1

        records.append((_resolve_image_path(image_name, annotation_path), float(gt_count)))

    return records


def _load_annotation_records(annotation_path: Path) -> List[Tuple[Path, float]]:
    suffix = annotation_path.suffix.lower()
    records: List[Tuple[Path, float]] = []

    if suffix == ".xml":
        return _load_xml_annotation_records(annotation_path)

    if suffix == ".json":
        with annotation_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            payload = payload.get("annotations", payload.get("data", []))

        if not isinstance(payload, list):
            raise ValueError("JSON annotation file must contain a list of annotation objects.")

        for item in payload:
            image_path = _get_first_present(item, ["image_path", "image", "file_name"])
            gt_count = _get_first_present(item, ["count", "gt_count", "crowd_count"])
            if image_path is None or gt_count is None:
                raise ValueError(
                    "Each JSON annotation item must include image_path/file_name and count/gt_count."
                )
            records.append((_resolve_image_path(str(image_path), annotation_path), float(gt_count)))
        return records

    if suffix == ".csv":
        with annotation_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = _get_first_present(row, ["image_path", "image", "file_name"])
                gt_count = _get_first_present(row, ["count", "gt_count", "crowd_count"])
                if image_path is None or gt_count is None:
                    raise ValueError(
                        "CSV annotation file must include image_path/file_name and count/gt_count columns."
                    )
                records.append((_resolve_image_path(image_path, annotation_path), float(gt_count)))
        return records

    raise ValueError("Unsupported annotation format. Use .xml, .json, or .csv.")


def _load_rgb_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def _get_annotation_path() -> Path:
    raw_path = os.getenv(ANNOTATION_ENV)
    if not raw_path:
        pytest.skip(f"Set {ANNOTATION_ENV} to run the CLIP-EBC MAE evaluation test.")

    annotation_path = Path(raw_path)
    if not annotation_path.is_file():
        pytest.skip(f"Annotation file not found: {annotation_path}")

    return annotation_path


def test_clip_ebc_inference_with_fixed_images(model, image_paths):
    resolved_image_paths = [Path(image_path) for image_path in image_paths]
    missing_images = [str(image_path) for image_path in resolved_image_paths if not image_path.is_file()]
    assert not missing_images, f"Some fixed test images do not exist: {missing_images}"

    images = [_load_rgb_image(image_path) for image_path in resolved_image_paths]
    pred_counts = model(images)

    assert isinstance(pred_counts, list), "Prediction result must be a list."
    assert len(pred_counts) == len(images), "Prediction count must match image count."
    assert all(isinstance(pred, (float, np.floating)) for pred in pred_counts)

    print("---------------------------------------------------")
    print(f"image_paths: {[str(image_path) for image_path in resolved_image_paths]}")
    print(f"num_images: {len(images)}")
    print(f"pred_counts: {pred_counts}")


def test_clip_ebc_mae_with_annotation_file(model, annotation_paths):
    annotation_path = Path(annotation_paths)
    if not annotation_path.is_file():
        pytest.skip(f"Annotation file not found: {annotation_path}")
    records = _load_annotation_records(annotation_path)

    assert records, "Annotation file must contain at least one evaluation sample."

    image_paths = [image_path for image_path, _ in records]
    gt_counts = [gt_count for _, gt_count in records]

    assert len(image_paths) == len(gt_counts), "Each evaluation image must have exactly one GT count."
    assert len(image_paths) == len(records), "Annotation rows must map 1:1 to evaluation images."

    missing_images = [str(image_path) for image_path in image_paths if not image_path.is_file()]
    assert not missing_images, f"Some annotated images do not exist: {missing_images}"

    images = [_load_rgb_image(image_path) for image_path in image_paths]
    pred_counts = model(images)

    assert isinstance(pred_counts, list), "Prediction result must be a list."
    assert len(pred_counts) == len(gt_counts), "Prediction count must match annotation count."
    assert all(isinstance(pred, (float, np.floating)) for pred in pred_counts)

    mae = float(np.mean(np.abs(np.asarray(pred_counts, dtype=np.float32) - np.asarray(gt_counts))))
    print("---------------------------------------------------")
    print(f"annotation_path: {annotation_path}")
    print(f"num_images: {len(images)}")
    print(f"pred_counts: {pred_counts}")
    print(f"gt_counts: {gt_counts}")
    print(f"MAE: {mae:.6f}")

    max_mae = os.getenv(MAX_MAE_ENV)
    if max_mae is not None:
        assert mae <= float(max_mae), f"MAE {mae:.6f} exceeded threshold {float(max_mae):.6f}"
