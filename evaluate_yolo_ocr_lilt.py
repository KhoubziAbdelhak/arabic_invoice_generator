"""Batch-evaluate YOLO -> OCR -> LiLT on real invoice images.

The script creates ground truth from ``real_invoices/annotation_corrected``,
runs the local detector/OCR/KIE pipeline, and writes metrics plus error
analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ultralytics import YOLO


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TATWEEL = "\u0640"
ARABIC_PUNCTUATION_RE = re.compile(r"[،؛؟]")
WHITESPACE_RE = re.compile(r"\s+")

EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

FIELD_TO_LILT = {
    "invoice_number": "INVOICE_NUMBER",
    "invoice_ref": "INVOICE_NUMBER",
    "offer_number": "INVOICE_NUMBER",
    "issue_date": "INVOICE_DATE",
    "issue_datetime": "INVOICE_DATE",
    "hijri_date": "INVOICE_DATE",
    "issue_time": "INVOICE_TIME",
    "document_type": "INVOICE_TYPE",
    "invoice_type": "INVOICE_TYPE",
    "payment_type": "INVOICE_TYPE",
    "branch_name": "BRANCH",
    "warehouse": "BRANCH",
    "page_number": "PAGE_NUMBER",
    "customer_name": "CUSTOMER_NAME",
    "recipient_name": "CUSTOMER_NAME",
    "recipient_company": "CUSTOMER_NAME",
    "customer_vat_number": "CUSTOMER_VAT",
    "recipient_vat_number": "CUSTOMER_VAT",
    "customer_number": "CUSTOMER_NUMBER",
    "customer_phone": "CUSTOMER_PHONE",
    "recipient_phone": "CUSTOMER_PHONE",
    "street_address": "CUSTOMER_ADDRESS",
    "vendor_name_ar": "COMPANY_NAME",
    "vendor_name_en": "COMPANY_NAME",
    "vendor_activity": "COMPANY_NAME",
    "company_name": "COMPANY_NAME",
    "vendor_address_en": "SELLER_ADDRESS",
    "seller_address": "SELLER_ADDRESS",
    "vendor_vat_number": "SELLER_VAT",
    "seller_vat_number": "SELLER_VAT",
    "vendor_cr_number": "SELLER_VAT",
    "vendor_phone": "SELLER_PHONE",
    "vendor_admin_phone": "SELLER_PHONE",
    "seller_name": "SELLER_NAME",
    "subtotal": "SUBTOTAL",
    "discount": "DISCOUNT",
    "net_amount": "NET_AMOUNT",
    "vat_amount": "TAX",
    "vat_rate": "TAX",
    "tax": "TAX",
    "grand_total": "TOTAL",
    "total": "TOTAL",
    "total_amount": "TOTAL",
    "vat_taxable_qty": "TAX",
    "email": "EMAIL",
    "website": "WEBSITE",
    "instructions": "INSTRUCTIONS",
}

LILT_TO_FIELDS = defaultdict(list)
for gt_field, lilt_field in FIELD_TO_LILT.items():
    LILT_TO_FIELDS[lilt_field].append(gt_field)


@dataclass
class OcrToken:
    text: str
    bbox: list[int]
    bbox_norm: list[int]
    ocr_confidence: float
    detection_confidence: float
    detection_id: str


@dataclass
class PipelineOutput:
    image_path: Path
    detections: list[dict]
    ocr_tokens: list[dict]
    lilt_predictions: list[dict]
    entities: list[dict]
    fields: dict[str, list[dict]]
    timings: dict[str, float]
    gpu_memory_mb: float | None = None


@dataclass
class EvaluationConfig:
    images_dir: Path = Path("real_invoices/images")
    annotations_dir: Path = Path("real_invoices/annotation_corrected")
    yolo_model: Path = Path("yolo26_text-detection.pt")
    lilt_model: Path = Path("lilt_arabic_best")
    output_dir: Path = Path("evaluation_output")
    preprocess_mode: str = "enhance"
    reading_direction: str = "rtl"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 1280
    crop_padding: int = 4
    min_ocr_conf: float = 0.05
    max_seq_length: int = 512
    warmup_runs: int = 1
    strip_diacritics: bool = True
    punctuation_cleanup: bool = False
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def normalize_arabic_text(
    value: Any,
    *,
    strip_diacritics: bool = True,
    normalize_digits: bool = True,
    lowercase: bool = False,
    punctuation_cleanup: bool = False,
) -> str:
    """Normalize mixed Arabic/English text for OCR and field metrics."""
    if value is None:
        text = ""
    else:
        text = str(value)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")

    if strip_diacritics:
        text = ARABIC_DIACRITICS_RE.sub("", text)

    text = (
        text.replace("إ", "ا")
        .replace("أ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ئ", "ي")
        .replace("ؤ", "و")
    )

    if normalize_digits:
        text = text.translate(EASTERN_ARABIC_DIGITS).translate(PERSIAN_DIGITS)

    if punctuation_cleanup:
        text = ARABIC_PUNCTUATION_RE.sub(" ", text)
        text = re.sub(r"[^\w\s./:%+-]", " ", text, flags=re.UNICODE)

    if lowercase:
        text = text.lower()

    return WHITESPACE_RE.sub(" ", text).strip()


def arabic_tokenize(text: Any, *, strip_diacritics: bool = True) -> list[str]:
    normalized = normalize_arabic_text(text, strip_diacritics=strip_diacritics)
    return re.findall(r"[\u0600-\u06ff]+|[A-Za-z]+|\d+(?:[./:-]\d+)*|[^\s]", normalized)


def levenshtein_distance(seq_a: list[Any] | str, seq_b: list[Any] | str) -> int:
    a = list(seq_a)
    b = list(seq_b)
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if item_a == item_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_cer(prediction: Any, ground_truth: Any, *, strip_diacritics: bool = True) -> dict[str, Any]:
    pred = normalize_arabic_text(prediction, strip_diacritics=strip_diacritics)
    gt = normalize_arabic_text(ground_truth, strip_diacritics=strip_diacritics)
    distance = levenshtein_distance(pred, gt)
    denominator = len(gt)
    cer = distance / denominator if denominator else (0.0 if not pred else 1.0)
    return {"cer": cer, "distance": distance, "gt_chars": denominator, "pred_chars": len(pred)}


def compute_wer(prediction: Any, ground_truth: Any, *, strip_diacritics: bool = True) -> dict[str, Any]:
    pred_tokens = arabic_tokenize(prediction, strip_diacritics=strip_diacritics)
    gt_tokens = arabic_tokenize(ground_truth, strip_diacritics=strip_diacritics)
    distance = levenshtein_distance(pred_tokens, gt_tokens)
    denominator = len(gt_tokens)
    wer = distance / denominator if denominator else (0.0 if not pred_tokens else 1.0)
    return {"wer": wer, "distance": distance, "gt_words": denominator, "pred_words": len(pred_tokens)}


def compute_exact_match(
    prediction: Any,
    ground_truth: Any,
    *,
    strip_diacritics: bool = True,
    punctuation_cleanup: bool = False,
) -> int:
    pred = normalize_arabic_text(
        prediction,
        strip_diacritics=strip_diacritics,
        lowercase=True,
        punctuation_cleanup=punctuation_cleanup,
    )
    gt = normalize_arabic_text(
        ground_truth,
        strip_diacritics=strip_diacritics,
        lowercase=True,
        punctuation_cleanup=punctuation_cleanup,
    )
    return int(pred == gt)


def compute_edit_accuracy(
    prediction: Any,
    ground_truth: Any,
    *,
    strip_diacritics: bool = True,
    punctuation_cleanup: bool = False,
) -> float:
    pred = normalize_arabic_text(
        prediction,
        strip_diacritics=strip_diacritics,
        lowercase=True,
        punctuation_cleanup=punctuation_cleanup,
    )
    gt = normalize_arabic_text(
        ground_truth,
        strip_diacritics=strip_diacritics,
        lowercase=True,
        punctuation_cleanup=punctuation_cleanup,
    )
    if not gt:
        return 1.0 if not pred else 0.0
    score = 1.0 - (levenshtein_distance(pred, gt) / len(gt))
    return max(0.0, min(1.0, score))


def bbox_to_xyxy(bbox: dict[str, Any] | list[int] | tuple[Any, ...], width: int, height: int) -> list[int]:
    if isinstance(bbox, dict):
        x1 = float(bbox.get("x", 0))
        y1 = float(bbox.get("y", 0))
        x2 = x1 + float(bbox.get("width", 0))
        y2 = y1 + float(bbox.get("height", 0))
    else:
        x1, y1, x2, y2 = [float(v) for v in bbox]

    x1 = max(0, min(width - 1, round(x1)))
    y1 = max(0, min(height - 1, round(y1)))
    x2 = max(0, min(width - 1, round(x2)))
    y2 = max(0, min(height - 1, round(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(x1), int(y1), int(x2), int(y2)]


def annotation_box_to_xyxy(item: dict[str, Any], width: int, height: int) -> list[int]:
    if item.get("bbox"):
        return bbox_to_xyxy(item["bbox"], width, height)
    bounding_poly = item.get("bounding_poly") or {}
    vertices = bounding_poly.get("vertices") or []
    if vertices:
        xs = [float(vertex.get("x", 0)) for vertex in vertices]
        ys = [float(vertex.get("y", 0)) for vertex in vertices]
        return bbox_to_xyxy([min(xs), min(ys), max(xs), max(ys)], width, height)
    return [0, 0, 0, 0]


def normalize_bbox_xyxy(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(round(1000 * x1 / width)))),
        max(0, min(1000, int(round(1000 * y1 / height)))),
        max(0, min(1000, int(round(1000 * x2 / width)))),
        max(0, min(1000, int(round(1000 * y2 / height)))),
    ]


def bbox_union(boxes: list[list[int]]) -> list[int]:
    arr = np.array(boxes, dtype=np.int32)
    x1, y1 = arr[:, :2].min(axis=0)
    x2, y2 = arr[:, 2:].max(axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def center_in_box(center: tuple[float, float], box: list[int]) -> bool:
    x, y = center
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def sort_by_reading_order(items: list[dict], direction: str = "rtl", y_tolerance: int | None = None) -> list[dict]:
    if not items:
        return []

    heights = [max(1, item["bbox"][3] - item["bbox"][1]) for item in items]
    tolerance = y_tolerance if y_tolerance is not None else max(8, int(np.median(heights) * 0.60))

    def center_y(item: dict) -> float:
        return (item["bbox"][1] + item["bbox"][3]) / 2

    def center_x(item: dict) -> float:
        return (item["bbox"][0] + item["bbox"][2]) / 2

    lines: list[dict] = []
    for item in sorted(items, key=lambda value: (center_y(value), center_x(value))):
        cy = center_y(item)
        for line in lines:
            if abs(cy - line["cy"]) <= tolerance:
                line["items"].append(item)
                line["cy"] = float(np.mean([center_y(line_item) for line_item in line["items"]]))
                break
        else:
            lines.append({"cy": cy, "items": [item]})

    ordered = []
    for line in sorted(lines, key=lambda value: value["cy"]):
        ordered.extend(sorted(line["items"], key=center_x, reverse=direction.lower() == "rtl"))
    return ordered


def preprocess_invoice(image_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if mode == "enhance":
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    if mode == "threshold":
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported preprocess mode: {mode}")


def load_image_bgr(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image_bgr


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def extract_yolo_detections(result: Any, width: int, height: int) -> list[dict]:
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for idx, box in enumerate(xyxy):
            class_id = int(classes[idx])
            detections.append(
                {
                    "detection_id": f"det_{idx:04d}",
                    "bbox": bbox_to_xyxy(box.tolist(), width, height),
                    "confidence": float(confs[idx]),
                    "class_id": class_id,
                    "class_name": class_name(result.names, class_id),
                    "type": "box",
                }
            )

    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy().astype(int)
        start_idx = len(detections)
        for rel_idx, polygon in enumerate(polygons):
            polygon = polygon.reshape(4, 2)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            x1, y1 = polygon.min(axis=0)
            x2, y2 = polygon.max(axis=0)
            class_id = int(classes[rel_idx])
            detections.append(
                {
                    "detection_id": f"det_{start_idx + rel_idx:04d}",
                    "bbox": bbox_to_xyxy([x1, y1, x2, y2], width, height),
                    "polygon": polygon.round().astype(int).tolist(),
                    "confidence": float(confs[rel_idx]),
                    "class_id": class_id,
                    "class_name": class_name(result.names, class_id),
                    "type": "obb",
                }
            )
    return detections


def crop_with_padding(image_bgr: np.ndarray, bbox: list[int], padding: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    padded = bbox_to_xyxy([x1 - padding, y1 - padding, x2 + padding, y2 + padding], width, height)
    px1, py1, px2, py2 = padded
    return image_bgr[py1 : py2 + 1, px1 : px2 + 1], (px1, py1)


def prepare_crop_for_ocr(crop_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    if crop_bgr.size == 0:
        return crop_bgr, 1.0, 1.0
    height, width = crop_bgr.shape[:2]
    scale = 1
    if min(height, width) < 48:
        scale = 2
    if max(height, width) < 160:
        scale = max(scale, 2)
    if scale > 1:
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced, width / max(1, enhanced.shape[1]), height / max(1, enhanced.shape[0])


def polygon_to_xyxy(points: Any) -> list[int]:
    arr = np.array(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def split_text_into_token_boxes(text: str, bbox: list[int]) -> list[tuple[str, list[int]]]:
    pieces = [piece for piece in re.split(r"\s+", normalize_arabic_text(text, strip_diacritics=False)) if piece]
    if not pieces:
        return []
    if len(pieces) == 1:
        return [(pieces[0], bbox)]

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    weights = np.array([max(1, len(piece)) for piece in pieces], dtype=np.float32)
    weights = weights / weights.sum()
    rtl = any("\u0600" <= ch <= "\u06ff" for ch in text)

    token_boxes = []
    if rtl:
        cursor = float(x2)
        for piece, weight in zip(pieces, weights):
            token_width = float(width * weight)
            part_x1 = cursor - token_width
            token_boxes.append((piece, bbox_to_xyxy([part_x1, y1, cursor, y2], 10**9, 10**9)))
            cursor = part_x1
    else:
        cursor = float(x1)
        for piece, weight in zip(pieces, weights):
            token_width = float(width * weight)
            part_x2 = cursor + token_width
            token_boxes.append((piece, bbox_to_xyxy([cursor, y1, part_x2, y2], 10**9, 10**9)))
            cursor = part_x2
    return token_boxes


def deduplicate_ocr_tokens(tokens: list[dict], reading_direction: str, iou_threshold: float = 0.80) -> list[dict]:
    kept = []
    for token in sorted(tokens, key=lambda value: value["ocr_confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            same_text = normalize_arabic_text(token["text"]) == normalize_arabic_text(existing["text"])
            if same_text and bbox_iou(token["bbox"], existing["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return sort_by_reading_order(kept, reading_direction)


def build_ground_truths(config: EvaluationConfig) -> dict[str, dict]:
    ground_truths: dict[str, dict] = {}
    for annotation_path in sorted(config.annotations_dir.glob("*.json")):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        stem = annotation_path.stem
        image_path = config.images_dir / f"{stem}.png"
        if not image_path.exists():
            matches = sorted(config.images_dir.glob(f"{stem}.*"))
            image_path = matches[0] if matches else image_path

        width = int(data.get("image_size", {}).get("width", 1))
        height = int(data.get("image_size", {}).get("height", 1))

        gt_tokens = []
        for token in data.get("entities", {}).get("ocr_text", []):
            bbox = annotation_box_to_xyxy(token, width, height)
            gt_tokens.append(
                {
                    "id": token.get("id"),
                    "text": token.get("text", ""),
                    "bbox": bbox,
                    "bbox_norm": normalize_bbox_xyxy(bbox, width, height),
                    "direction": token.get("direction"),
                    "line_id": token.get("line_id"),
                }
            )

        gt_tokens = sort_by_reading_order(gt_tokens)
        token_id_to_index = {token["id"]: idx for idx, token in enumerate(gt_tokens)}

        fields = {}
        field_boxes = {}
        field_token_ids = {}
        gt_token_labels = ["O"] * len(gt_tokens)

        for field_item in data.get("entities", {}).get("kie_fields", []):
            label = field_item.get("label")
            if not label:
                continue
            text = field_item.get("normalized_value", field_item.get("text", ""))
            fields[label] = text
            if field_item.get("bbox"):
                field_boxes[label] = bbox_to_xyxy(field_item["bbox"], width, height)
            token_ids = field_item.get("token_ids", []) or []
            field_token_ids[label] = token_ids

            lilt_label = FIELD_TO_LILT.get(label)
            if not lilt_label:
                continue
            first = True
            for token_id in token_ids:
                idx = token_id_to_index.get(token_id)
                if idx is None:
                    continue
                prefix = "B" if first else "I"
                gt_token_labels[idx] = f"{prefix}-{lilt_label}"
                first = False

        line_items = []
        for item in data.get("entities", {}).get("line_items", []):
            fields_payload = {}
            for label, payload in item.get("fields", {}).items():
                fields_payload[label] = payload.get("text", "")
            line_items.append({"item_index": item.get("item_index"), "fields": fields_payload})

        ground_truths[stem] = {
            "doc_id": data.get("doc_id", stem),
            "image_path": str(image_path),
            "image_size": {"width": width, "height": height},
            "ocr_tokens": gt_tokens,
            "ocr_text": " ".join(token["text"] for token in gt_tokens if token.get("text")),
            "fields": fields,
            "field_boxes": field_boxes,
            "field_token_ids": field_token_ids,
            "token_labels": gt_token_labels,
            "line_items": line_items,
        }
    return ground_truths


def run_ocr_on_detections(
    image_bgr: np.ndarray,
    detections: list[dict],
    reader: easyocr.Reader,
    config: EvaluationConfig,
    crops_dir: Path,
) -> list[dict]:
    height, width = image_bgr.shape[:2]
    ocr_tokens = []
    crops_dir.mkdir(parents=True, exist_ok=True)

    for det_idx, detection in enumerate(detections):
        crop_bgr, (offset_x, offset_y) = crop_with_padding(image_bgr, detection["bbox"], config.crop_padding)
        if crop_bgr.size == 0:
            continue
        crop_path = crops_dir / f"crop_{det_idx:04d}.jpg"
        cv2.imwrite(str(crop_path), crop_bgr)

        crop_gray, scale_x, scale_y = prepare_crop_for_ocr(crop_bgr)
        crop_h, crop_w = crop_gray.shape[:2]
        recognized = reader.recognize(
            crop_gray,
            horizontal_list=[[0, crop_w, 0, crop_h]],
            free_list=[],
            detail=1,
            paragraph=False,
        )

        ocr_items = []
        if recognized:
            _, text, confidence = recognized[0]
            text = normalize_arabic_text(text, strip_diacritics=False)
            confidence = float(confidence)
            if text and confidence >= config.min_ocr_conf:
                ocr_items.append((text, confidence, detection["bbox"]))

        if not ocr_items:
            fallback_results = reader.readtext(crop_gray, detail=1, paragraph=False)
            for local_polygon, text, confidence in fallback_results:
                text = normalize_arabic_text(text, strip_diacritics=False)
                confidence = float(confidence)
                if not text or confidence < config.min_ocr_conf:
                    continue
                local_bbox = polygon_to_xyxy(local_polygon)
                local_bbox = [
                    int(round(local_bbox[0] * scale_x)),
                    int(round(local_bbox[1] * scale_y)),
                    int(round(local_bbox[2] * scale_x)),
                    int(round(local_bbox[3] * scale_y)),
                ]
                global_bbox = bbox_to_xyxy(
                    [
                        local_bbox[0] + offset_x,
                        local_bbox[1] + offset_y,
                        local_bbox[2] + offset_x,
                        local_bbox[3] + offset_y,
                    ],
                    width,
                    height,
                )
                ocr_items.append((text, confidence, global_bbox))

        for text, confidence, global_bbox in ocr_items:
            for token_text, token_bbox in split_text_into_token_boxes(text, global_bbox):
                token_bbox = bbox_to_xyxy(token_bbox, width, height)
                ocr_tokens.append(
                    {
                        "text": token_text,
                        "bbox": token_bbox,
                        "bbox_norm": normalize_bbox_xyxy(token_bbox, width, height),
                        "ocr_confidence": float(confidence),
                        "detection_confidence": float(detection["confidence"]),
                        "detection_id": detection["detection_id"],
                        "crop_path": str(crop_path),
                    }
                )

    return deduplicate_ocr_tokens(ocr_tokens, config.reading_direction)


def run_lilt_token_classification(
    ocr_tokens: list[dict],
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    config: EvaluationConfig,
) -> list[dict]:
    if not ocr_tokens:
        return []

    tokens = [item["text"] for item in ocr_tokens]
    boxes = [item["bbox_norm"] for item in ocr_tokens]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config.max_seq_length,
    )

    word_ids = encoding.word_ids(batch_index=0)
    aligned_boxes = []
    for word_id in word_ids:
        aligned_boxes.append([0, 0, 0, 0] if word_id is None else boxes[word_id])

    device = torch.device(config.device)
    model_inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
    model_inputs["bbox"] = torch.tensor([aligned_boxes], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(**model_inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()
        predicted_ids = probabilities.argmax(dim=-1).tolist()

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    seen_word_ids = set()
    predictions = []
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids or word_id >= len(ocr_tokens):
            continue
        seen_word_ids.add(word_id)
        label_id = int(predicted_ids[token_idx])
        predictions.append(
            {
                **ocr_tokens[word_id],
                "label": id2label.get(label_id, str(label_id)),
                "label_id": label_id,
                "lilt_score": float(probabilities[token_idx, label_id]),
                "token_index": int(word_id),
            }
        )
    return predictions


def bio_to_field(label: str) -> tuple[str | None, str | None]:
    if label == "O" or "-" not in label:
        return None, None
    prefix, field_name = label.split("-", 1)
    return prefix, field_name


def merge_bio_entities(predictions: list[dict], width: int, height: int) -> list[dict]:
    entities = []
    current = None

    def close_current():
        nonlocal current
        if current is None:
            return
        current["text"] = " ".join(current.pop("tokens"))
        current["bbox"] = bbox_union(current.pop("boxes"))
        current["bbox_norm"] = normalize_bbox_xyxy(current["bbox"], width, height)
        current["score"] = float(np.mean(current.pop("scores")))
        entities.append(current)
        current = None

    for prediction in predictions:
        prefix, field_name = bio_to_field(prediction["label"])
        if field_name is None:
            close_current()
            continue

        starts_new = prefix == "B" or current is None or current["field"] != field_name
        if starts_new:
            close_current()
            current = {
                "field": field_name,
                "tokens": [prediction["text"]],
                "boxes": [prediction["bbox"]],
                "scores": [prediction["lilt_score"]],
                "token_indices": [prediction["token_index"]],
            }
        else:
            current["tokens"].append(prediction["text"])
            current["boxes"].append(prediction["bbox"])
            current["scores"].append(prediction["lilt_score"])
            current["token_indices"].append(prediction["token_index"])

    close_current()
    return entities


def map_lilt_entities_to_gt_fields(entities: list[dict]) -> dict[str, list[dict]]:
    fields: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        gt_fields = LILT_TO_FIELDS.get(entity["field"], [entity["field"].lower()])
        for gt_field in gt_fields:
            fields[gt_field].append(
                {
                    "text": entity["text"],
                    "bbox": entity["bbox"],
                    "score": entity["score"],
                    "source_lilt_field": entity["field"],
                }
            )
    return dict(fields)


def choose_predicted_field_text(predicted_fields: dict[str, list[dict]], field_name: str) -> str:
    candidates = predicted_fields.get(field_name, [])
    if not candidates:
        return ""
    best = max(candidates, key=lambda item: item.get("score", 0.0))
    return best.get("text", "")


def draw_boxes(image_bgr: np.ndarray, boxes: list[dict], output_path: Path) -> None:
    canvas = image_bgr.copy()
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        label = item.get("label", "")
        color = item.get("color", (0, 0, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(canvas, label[:42], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(output_path), canvas)


def run_pipeline_on_image(
    image_path: Path,
    yolo_model: YOLO,
    ocr_reader: easyocr.Reader,
    tokenizer: AutoTokenizer,
    lilt_model: AutoModelForTokenClassification,
    config: EvaluationConfig,
    measured: bool = True,
) -> PipelineOutput:
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    image_bgr = load_image_bgr(image_path)
    height, width = image_bgr.shape[:2]

    start = time.perf_counter()
    preprocessed = preprocess_invoice(image_bgr, config.preprocess_mode)
    timings["preprocess_s"] = time.perf_counter() - start

    start = time.perf_counter()
    predict_kwargs = {
        "source": preprocessed,
        "conf": config.yolo_conf,
        "iou": config.yolo_iou,
        "imgsz": config.yolo_imgsz,
        "verbose": False,
        "device": 0 if config.device == "cuda" else "cpu",
    }
    yolo_result = yolo_model.predict(**predict_kwargs)[0]
    detections = sort_by_reading_order(extract_yolo_detections(yolo_result, width, height), config.reading_direction)
    timings["detection_s"] = time.perf_counter() - start

    stem = image_path.stem
    image_out_dir = config.output_dir / "samples" / stem
    image_out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_out_dir / "preprocessed.jpg"), preprocessed)
    draw_boxes(
        image_bgr,
        [{"bbox": det["bbox"], "label": f"{idx}:{det['confidence']:.2f}"} for idx, det in enumerate(detections, 1)],
        image_out_dir / "detections.jpg",
    )

    start = time.perf_counter()
    ocr_tokens = run_ocr_on_detections(image_bgr, detections, ocr_reader, config, image_out_dir / "crops")
    timings["ocr_s"] = time.perf_counter() - start

    start = time.perf_counter()
    lilt_predictions = run_lilt_token_classification(ocr_tokens, tokenizer, lilt_model, config)
    entities = merge_bio_entities(lilt_predictions, width, height)
    fields = map_lilt_entities_to_gt_fields(entities)
    timings["lilt_s"] = time.perf_counter() - start

    timings["total_s"] = time.perf_counter() - total_start
    gpu_memory_mb = None
    if config.device == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    draw_boxes(
        image_bgr,
        [
            {"bbox": entity["bbox"], "label": f"{entity['field']} {entity['score']:.2f}", "color": (0, 180, 0)}
            for entity in entities
        ],
        image_out_dir / "lilt_fields.jpg",
    )

    if measured:
        (image_out_dir / "predictions.json").write_text(
            json.dumps(
                {
                    "detections": detections,
                    "ocr_tokens": ocr_tokens,
                    "lilt_predictions": lilt_predictions,
                    "entities": entities,
                    "fields": fields,
                    "timings": timings,
                    "gpu_memory_mb": gpu_memory_mb,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return PipelineOutput(
        image_path=image_path,
        detections=detections,
        ocr_tokens=ocr_tokens,
        lilt_predictions=lilt_predictions,
        entities=entities,
        fields=fields,
        timings=timings,
        gpu_memory_mb=gpu_memory_mb,
    )


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = np.array([0.0] + recalls + [1.0])
    mpre = np.array([0.0] + precisions + [0.0])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_detection_ap(
    predictions_by_doc: dict[str, list[dict]],
    gt_boxes_by_doc: dict[str, list[list[int]]],
    iou_threshold: float,
) -> dict[str, Any]:
    records = []
    total_gt = sum(len(boxes) for boxes in gt_boxes_by_doc.values())
    matched_by_doc = {doc_id: set() for doc_id in gt_boxes_by_doc}

    for doc_id, predictions in predictions_by_doc.items():
        for pred in predictions:
            records.append((doc_id, float(pred.get("confidence", 0.0)), pred["bbox"]))
    records.sort(key=lambda item: item[1], reverse=True)

    tp = []
    fp = []
    ious = []
    for doc_id, _, pred_box in records:
        gt_boxes = gt_boxes_by_doc.get(doc_id, [])
        best_iou = 0.0
        best_idx = None
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_by_doc.setdefault(doc_id, set()):
                continue
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        ious.append(best_iou)
        if best_iou >= iou_threshold and best_idx is not None:
            matched_by_doc[doc_id].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not records or total_gt == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "mean_iou": 0.0}

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = (tp_cumsum / total_gt).tolist()
    precisions = (tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)).tolist()
    return {
        "ap": average_precision(recalls, precisions),
        "precision": float(precisions[-1]) if precisions else 0.0,
        "recall": float(recalls[-1]) if recalls else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def compute_detection_metrics(outputs: dict[str, PipelineOutput], ground_truths: dict[str, dict]) -> dict[str, Any]:
    predictions_by_doc = {doc_id: output.detections for doc_id, output in outputs.items()}
    gt_boxes_by_doc = {
        doc_id: [token["bbox"] for token in gt.get("ocr_tokens", [])]
        for doc_id, gt in ground_truths.items()
        if doc_id in outputs
    }

    ap50 = compute_detection_ap(predictions_by_doc, gt_boxes_by_doc, 0.5)
    thresholds = [round(threshold, 2) for threshold in np.arange(0.50, 1.00, 0.05)]
    ap_by_threshold = {
        f"AP@{threshold:.2f}": compute_detection_ap(predictions_by_doc, gt_boxes_by_doc, threshold)["ap"]
        for threshold in thresholds
    }

    return {
        "box_level_mean_iou": ap50["mean_iou"],
        "per_class_ap": {"text": ap50["ap"]},
        "mAP@0.5": ap50["ap"],
        "mAP@0.5:0.95": float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
        "ap_by_threshold": ap_by_threshold,
        "precision@0.5": ap50["precision"],
        "recall@0.5": ap50["recall"],
    }


def field_ocr_text(pred_tokens: list[dict], field_box: list[int] | None, reading_direction: str) -> str:
    if not field_box:
        return ""
    selected = []
    for token in pred_tokens:
        x1, y1, x2, y2 = token["bbox"]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if center_in_box(center, field_box) or bbox_iou(token["bbox"], field_box) > 0.15:
            selected.append(token)
    selected = sort_by_reading_order(selected, reading_direction)
    return " ".join(item["text"] for item in selected)


def compute_ocr_metrics(
    outputs: dict[str, PipelineOutput],
    ground_truths: dict[str, dict],
    config: EvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    global_pred_text = []
    global_gt_text = []
    field_accumulators = defaultdict(lambda: {"pred": [], "gt": []})

    for doc_id, gt in ground_truths.items():
        output = outputs.get(doc_id)
        if output is None:
            continue
        pred_text = " ".join(token["text"] for token in output.ocr_tokens)
        gt_text = gt.get("ocr_text", "")
        global_pred_text.append(pred_text)
        global_gt_text.append(gt_text)
        cer = compute_cer(pred_text, gt_text, strip_diacritics=config.strip_diacritics)
        wer = compute_wer(pred_text, gt_text, strip_diacritics=config.strip_diacritics)
        rows.append(
            {
                "doc_id": doc_id,
                "metric_scope": "document",
                "field": "",
                "gt": gt_text,
                "prediction": pred_text,
                "cer": cer["cer"],
                "wer": wer["wer"],
                "edit_distance_chars": cer["distance"],
                "edit_distance_words": wer["distance"],
            }
        )

        for field_name, gt_value in gt.get("fields", {}).items():
            pred_value = field_ocr_text(output.ocr_tokens, gt.get("field_boxes", {}).get(field_name), config.reading_direction)
            field_accumulators[field_name]["pred"].append(pred_value)
            field_accumulators[field_name]["gt"].append(gt_value)
            field_cer = compute_cer(pred_value, gt_value, strip_diacritics=config.strip_diacritics)
            field_wer = compute_wer(pred_value, gt_value, strip_diacritics=config.strip_diacritics)
            rows.append(
                {
                    "doc_id": doc_id,
                    "metric_scope": "ocr_field",
                    "field": field_name,
                    "gt": gt_value,
                    "prediction": pred_value,
                    "cer": field_cer["cer"],
                    "wer": field_wer["wer"],
                    "edit_distance_chars": field_cer["distance"],
                    "edit_distance_words": field_wer["distance"],
                }
            )

    global_cer = compute_cer(" ".join(global_pred_text), " ".join(global_gt_text), strip_diacritics=config.strip_diacritics)
    global_wer = compute_wer(" ".join(global_pred_text), " ".join(global_gt_text), strip_diacritics=config.strip_diacritics)
    per_field = {}
    for field_name, payload in field_accumulators.items():
        per_field[field_name] = {
            "cer": compute_cer(" ".join(payload["pred"]), " ".join(payload["gt"]), strip_diacritics=config.strip_diacritics)["cer"],
            "wer": compute_wer(" ".join(payload["pred"]), " ".join(payload["gt"]), strip_diacritics=config.strip_diacritics)["wer"],
        }

    return (
        {
            "global_cer": global_cer["cer"],
            "global_wer": global_wer["wer"],
            "per_document": {
                row["doc_id"]: {"cer": row["cer"], "wer": row["wer"]}
                for row in rows
                if row["metric_scope"] == "document"
            },
            "per_field": per_field,
        },
        rows,
    )


def compute_field_accuracy(
    outputs: dict[str, PipelineOutput],
    ground_truths: dict[str, dict],
    config: EvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    by_field = defaultdict(list)

    for doc_id, gt in ground_truths.items():
        output = outputs.get(doc_id)
        if output is None:
            continue
        for field_name, gt_value in gt.get("fields", {}).items():
            pred_value = choose_predicted_field_text(output.fields, field_name)
            exact = compute_exact_match(
                pred_value,
                gt_value,
                strip_diacritics=config.strip_diacritics,
                punctuation_cleanup=config.punctuation_cleanup,
            )
            edit_acc = compute_edit_accuracy(
                pred_value,
                gt_value,
                strip_diacritics=config.strip_diacritics,
                punctuation_cleanup=config.punctuation_cleanup,
            )
            row = {
                "doc_id": doc_id,
                "field": field_name,
                "gt": gt_value,
                "prediction": pred_value,
                "exact_match": exact,
                "edit_accuracy": edit_acc,
            }
            rows.append(row)
            by_field[field_name].append(row)

    per_field = {}
    for field_name, field_rows in by_field.items():
        per_field[field_name] = {
            "exact_match": float(np.mean([row["exact_match"] for row in field_rows])),
            "edit_accuracy": float(np.mean([row["edit_accuracy"] for row in field_rows])),
            "support": len(field_rows),
        }

    if rows:
        macro_exact = float(np.mean([value["exact_match"] for value in per_field.values()]))
        macro_edit = float(np.mean([value["edit_accuracy"] for value in per_field.values()]))
        micro_exact = float(np.mean([row["exact_match"] for row in rows]))
        micro_edit = float(np.mean([row["edit_accuracy"] for row in rows]))
    else:
        macro_exact = macro_edit = micro_exact = micro_edit = 0.0

    return (
        {
            "per_field": per_field,
            "macro_exact_match": macro_exact,
            "micro_exact_match": micro_exact,
            "macro_edit_accuracy": macro_edit,
            "micro_edit_accuracy": micro_edit,
        },
        rows,
    )


def align_predictions_to_gt_tokens(output: PipelineOutput, gt: dict) -> tuple[list[str], list[str]]:
    gt_tokens = gt.get("ocr_tokens", [])
    gt_labels = gt.get("token_labels", [])
    if not gt_tokens or not output.lilt_predictions:
        return [], []

    y_true = []
    y_pred = []
    used_pred_indices = set()
    for gt_idx, gt_token in enumerate(gt_tokens):
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred in enumerate(output.lilt_predictions):
            if pred_idx in used_pred_indices:
                continue
            iou = bbox_iou(pred["bbox"], gt_token["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx

        if best_idx is not None and best_iou >= 0.10:
            used_pred_indices.add(best_idx)
            y_true.append(gt_labels[gt_idx] if gt_idx < len(gt_labels) else "O")
            y_pred.append(output.lilt_predictions[best_idx]["label"])

    return y_true, y_pred


def compute_classification_metrics(
    outputs: dict[str, PipelineOutput],
    ground_truths: dict[str, dict],
    output_dir: Path,
) -> dict[str, Any]:
    y_true_all = []
    y_pred_all = []
    for doc_id, gt in ground_truths.items():
        output = outputs.get(doc_id)
        if output is None:
            continue
        y_true, y_pred = align_predictions_to_gt_tokens(output, gt)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    if not y_true_all:
        return {"skipped": True, "reason": "No matched GT/predicted token boxes for classification metrics."}

    labels = sorted(set(y_true_all) | set(y_pred_all))
    report = classification_report(y_true_all, y_pred_all, labels=labels, zero_division=0, output_dict=True)
    averages = {}
    for average in ["micro", "macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_all,
            y_pred_all,
            labels=labels,
            average=average,
            zero_division=0,
        )
        averages[average] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    matrix = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    plt.figure(figsize=(max(10, len(labels) * 0.35), max(8, len(labels) * 0.35)))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("LiLT Token Classification Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=7)
    plt.yticks(tick_marks, labels, fontsize=7)
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=180)
    plt.close()

    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")

    return {
        "skipped": False,
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "averages": averages,
        "classification_report": report,
        "support": len(y_true_all),
        "labels": labels,
        "confusion_matrix_png": str(confusion_path),
    }


def compute_runtime_metrics(outputs: dict[str, PipelineOutput]) -> dict[str, Any]:
    latencies = [output.timings["total_s"] for output in outputs.values()]
    if not latencies:
        return {}
    latency_arr = np.array(latencies, dtype=np.float64)
    total_time = float(latency_arr.sum())
    stage_metrics = {}
    for stage in ["preprocess_s", "detection_s", "ocr_s", "lilt_s", "total_s"]:
        values = [output.timings.get(stage, 0.0) for output in outputs.values()]
        stage_metrics[stage] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    gpu_values = [output.gpu_memory_mb for output in outputs.values() if output.gpu_memory_mb is not None]
    return {
        "per_page_latency_s": {doc_id: output.timings["total_s"] for doc_id, output in outputs.items()},
        "mean_latency_s": float(latency_arr.mean()),
        "std_latency_s": float(latency_arr.std()),
        "min_latency_s": float(latency_arr.min()),
        "max_latency_s": float(latency_arr.max()),
        "throughput_pages_per_sec": len(latencies) / total_time if total_time else 0.0,
        "stage_metrics": stage_metrics,
        "gpu_memory_mb": {
            "max": float(np.max(gpu_values)) if gpu_values else None,
            "mean": float(np.mean(gpu_values)) if gpu_values else None,
        },
    }


def save_error_analysis(
    output_dir: Path,
    ocr_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
) -> None:
    ocr_df = pd.DataFrame(ocr_rows)
    field_df = pd.DataFrame(field_rows)

    if not ocr_df.empty:
        ocr_df.sort_values("cer", ascending=False).head(20).to_csv(output_dir / "worst_cer_examples.csv", index=False)
        ocr_df.sort_values("wer", ascending=False).head(20).to_csv(output_dir / "worst_wer_examples.csv", index=False)

        plt.figure(figsize=(8, 5))
        ocr_df["cer"].dropna().clip(0, 3).hist(bins=30)
        plt.title("CER Distribution")
        plt.xlabel("CER")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "cer_histogram.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        ocr_df["wer"].dropna().clip(0, 3).hist(bins=30)
        plt.title("WER Distribution")
        plt.xlabel("WER")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "wer_histogram.png", dpi=160)
        plt.close()

    if not field_df.empty:
        failures = field_df[field_df["exact_match"] == 0].copy()
        failures.to_csv(output_dir / "field_extraction_failures.csv", index=False)

        plt.figure(figsize=(8, 5))
        field_df["edit_accuracy"].dropna().hist(bins=20)
        plt.title("Field Edit-Accuracy Distribution")
        plt.xlabel("Normalized Edit-Based Accuracy")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "field_edit_accuracy_histogram.png", dpi=160)
        plt.close()


def benchmark_inference(
    image_paths: list[Path],
    yolo_model: YOLO,
    ocr_reader: easyocr.Reader,
    tokenizer: AutoTokenizer,
    lilt_model: AutoModelForTokenClassification,
    config: EvaluationConfig,
) -> dict[str, PipelineOutput]:
    if config.warmup_runs > 0 and image_paths:
        print(f"Warmup runs: {config.warmup_runs}")
        for warmup_idx in range(config.warmup_runs):
            run_pipeline_on_image(
                image_paths[warmup_idx % len(image_paths)],
                yolo_model,
                ocr_reader,
                tokenizer,
                lilt_model,
                config,
                measured=False,
            )
        print("Warmup complete.")

    outputs = {}
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing {image_path.name}")
        outputs[image_path.stem] = run_pipeline_on_image(
            image_path,
            yolo_model,
            ocr_reader,
            tokenizer,
            lilt_model,
            config,
            measured=True,
        )
        print(f"  total={outputs[image_path.stem].timings['total_s']:.2f}s")
    return outputs


def write_summary(evaluation: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "Evaluation Summary",
        "==================",
        f"CER                 : {evaluation['ocr']['global_cer']:.4f}",
        f"WER                 : {evaluation['ocr']['global_wer']:.4f}",
        f"Exact Match micro   : {evaluation['field_extraction']['micro_exact_match']:.4f}",
        f"Exact Match macro   : {evaluation['field_extraction']['macro_exact_match']:.4f}",
        f"Edit Accuracy micro : {evaluation['field_extraction']['micro_edit_accuracy']:.4f}",
        f"Edit Accuracy macro : {evaluation['field_extraction']['macro_edit_accuracy']:.4f}",
    ]

    classification = evaluation.get("classification", {})
    if not classification.get("skipped"):
        lines.append(f"Token F1 micro      : {classification['averages']['micro']['f1']:.4f}")
        lines.append(f"Token F1 macro      : {classification['averages']['macro']['f1']:.4f}")
        lines.append(f"Token F1 weighted   : {classification['averages']['weighted']['f1']:.4f}")
    else:
        lines.append("Token F1            : skipped")

    detection = evaluation.get("detection", {})
    lines.extend(
        [
            f"mAP@0.5             : {detection.get('mAP@0.5', 0.0):.4f}",
            f"mAP@0.5:0.95        : {detection.get('mAP@0.5:0.95', 0.0):.4f}",
            f"Mean IoU            : {detection.get('box_level_mean_iou', 0.0):.4f}",
            f"Avg Latency         : {evaluation['runtime'].get('mean_latency_s', 0.0):.4f}s",
            f"Throughput          : {evaluation['runtime'].get('throughput_pages_per_sec', 0.0):.4f} pages/sec",
            "",
            "Saved Files",
            "===========",
            str(output_dir / "ground_truths.json"),
            str(output_dir / "evaluation_results.json"),
            str(output_dir / "per_sample_results.csv"),
            str(output_dir / "error_analysis.csv"),
            str(output_dir / "field_extraction_failures.csv"),
            str(output_dir / "confusion_matrix.png"),
        ]
    )
    summary = "\n".join(lines)
    print("\n" + summary)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")


def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    if not config.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {config.images_dir}")
    if not config.annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory does not exist: {config.annotations_dir}")

    annotation_paths = sorted(config.annotations_dir.glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No JSON annotations found in: {config.annotations_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "samples").mkdir(parents=True, exist_ok=True)

    ground_truths = build_ground_truths(config)
    (config.output_dir / "ground_truths.json").write_text(
        json.dumps(ground_truths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created ground truths for {len(ground_truths)} documents.")

    all_image_paths = sorted(path for path in config.images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not all_image_paths:
        raise FileNotFoundError(f"No supported image files found in: {config.images_dir}")

    image_paths = [path for path in all_image_paths if path.stem in ground_truths]
    if not image_paths:
        image_stems = [path.stem for path in all_image_paths[:10]]
        gt_stems = list(ground_truths)[:10]
        raise FileNotFoundError(
            "No images with matching ground truths found. "
            f"images_dir={config.images_dir} annotations_dir={config.annotations_dir} "
            f"sample_image_stems={image_stems} sample_annotation_stems={gt_stems}"
        )

    print(f"Loading models on {config.device}...")
    yolo_model = YOLO(str(config.yolo_model))
    ocr_reader = easyocr.Reader(["ar", "en"], gpu=(config.device == "cuda"))
    tokenizer = AutoTokenizer.from_pretrained(str(config.lilt_model), use_fast=True, local_files_only=True)
    lilt_model = AutoModelForTokenClassification.from_pretrained(str(config.lilt_model), local_files_only=True)
    lilt_model.to(torch.device(config.device))
    lilt_model.eval()

    outputs = benchmark_inference(image_paths, yolo_model, ocr_reader, tokenizer, lilt_model, config)

    ocr_metrics, ocr_rows = compute_ocr_metrics(outputs, ground_truths, config)
    field_metrics, field_rows = compute_field_accuracy(outputs, ground_truths, config)
    classification_metrics = compute_classification_metrics(outputs, ground_truths, config.output_dir)
    detection_metrics = compute_detection_metrics(outputs, ground_truths)
    runtime_metrics = compute_runtime_metrics(outputs)

    per_sample_rows = []
    for doc_id, output in outputs.items():
        doc_ocr = ocr_metrics["per_document"].get(doc_id, {})
        per_sample_rows.append(
            {
                "doc_id": doc_id,
                "image_path": str(output.image_path),
                "num_detections": len(output.detections),
                "num_ocr_tokens": len(output.ocr_tokens),
                "num_entities": len(output.entities),
                "cer": doc_ocr.get("cer"),
                "wer": doc_ocr.get("wer"),
                "latency_s": output.timings["total_s"],
                "detection_s": output.timings["detection_s"],
                "ocr_s": output.timings["ocr_s"],
                "lilt_s": output.timings["lilt_s"],
                "gpu_memory_mb": output.gpu_memory_mb,
            }
        )

    pd.DataFrame(per_sample_rows).to_csv(config.output_dir / "per_sample_results.csv", index=False)
    pd.DataFrame(ocr_rows).to_csv(config.output_dir / "ocr_error_analysis.csv", index=False)
    pd.DataFrame(field_rows).to_csv(config.output_dir / "error_analysis.csv", index=False)
    save_error_analysis(config.output_dir, ocr_rows, field_rows)

    evaluation = {
        "config": {
            "images_dir": str(config.images_dir),
            "annotations_dir": str(config.annotations_dir),
            "yolo_model": str(config.yolo_model),
            "lilt_model": str(config.lilt_model),
            "preprocess_mode": config.preprocess_mode,
            "reading_direction": config.reading_direction,
            "strip_diacritics": config.strip_diacritics,
            "punctuation_cleanup": config.punctuation_cleanup,
            "device": config.device,
            "warmup_runs": config.warmup_runs,
        },
        "ocr": ocr_metrics,
        "field_extraction": field_metrics,
        "classification": classification_metrics,
        "detection": detection_metrics,
        "runtime": runtime_metrics,
    }

    (config.output_dir / "evaluation_results.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(evaluation, config.output_dir)
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO OCR LiLT invoice extraction.")
    parser.add_argument("--images-dir", type=Path, default=Path("real_invoices/images"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("real_invoices/annotation_corrected"))
    parser.add_argument("--yolo-model", type=Path, default=Path("yolo26_text-detection.pt"))
    parser.add_argument("--lilt-model", type=Path, default=Path("lilt_arabic_best"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_output"))
    parser.add_argument("--preprocess", choices=["none", "enhance", "threshold"], default="enhance")
    parser.add_argument("--reading-direction", choices=["rtl", "ltr"], default="rtl")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--punctuation-cleanup", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        yolo_model=args.yolo_model,
        lilt_model=args.lilt_model,
        output_dir=args.output_dir,
        preprocess_mode=args.preprocess,
        reading_direction=args.reading_direction,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
        yolo_imgsz=args.imgsz,
        warmup_runs=args.warmup_runs,
        strip_diacritics=not args.keep_diacritics,
        punctuation_cleanup=args.punctuation_cleanup,
        device=args.device,
    )
    evaluate(config)


if __name__ == "__main__":
    main()
