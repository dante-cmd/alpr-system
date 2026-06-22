"""Métricas de detección, OCR y end-to-end."""

from typing import Dict, List, Tuple


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """IoU entre dos bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def levenshtein(a: str, b: str) -> int:
    """Distancia de Levenshtein."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def character_accuracy_rate(predictions: List[str], ground_truth: List[str]) -> float:
    """CAR: accuracy por carácter promedio."""
    correct = 0
    total = 0
    for pred, gt in zip(predictions, ground_truth):
        total += len(gt)
        correct += sum(1 for pc, gc in zip(pred, gt) if pc == gc)
    return correct / total if total else 0.0


def word_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """WA: exactitud a nivel palabra/placa."""
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def exact_plate_match(predictions: List[str], ground_truth: List[str]) -> float:
    """Alias de word accuracy para nomenclatura ALPR."""
    return word_accuracy(predictions, ground_truth)


def average_levenshtein(predictions: List[str], ground_truth: List[str]) -> float:
    """Distancia de Levenshtein promedio."""
    if not ground_truth:
        return 0.0
    return sum(levenshtein(p, g) for p, g in zip(predictions, ground_truth)) / len(ground_truth)


def detection_metrics(
    pred_boxes: List[List[Tuple[int, int, int, int]]],
    gt_boxes: List[List[Tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcula precision, recall y F1 por imagen."""
    tp_total = fp_total = fn_total = 0
    for preds, gts in zip(pred_boxes, gt_boxes):
        matched_gt = set()
        tp = 0
        for pred in preds:
            best_iou = 0.0
            best_gt = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt:
                    continue
                score = iou(pred, gt)
                if score > best_iou:
                    best_iou = score
                    best_gt = idx
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp_total += 1
        tp_total += tp
        fn_total += len(gts) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_batch(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluación end-to-end con listas de dicts con 'text' y 'bbox'."""
    pred_texts = [p["text"] for p in predictions]
    gt_texts = [g["text"] for g in ground_truth]
    pred_boxes = [[tuple(p["bbox"].values())] for p in predictions]
    gt_boxes = [[tuple(g["bbox"].values())] for g in ground_truth]

    det = detection_metrics(pred_boxes, gt_boxes, iou_threshold)
    return {
        "exact_match": exact_plate_match(pred_texts, gt_texts),
        "character_accuracy_rate": character_accuracy_rate(pred_texts, gt_texts),
        "word_accuracy": word_accuracy(pred_texts, gt_texts),
        "avg_levenshtein": average_levenshtein(pred_texts, gt_texts),
        **det,
    }
