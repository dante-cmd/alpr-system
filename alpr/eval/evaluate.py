"""Script CLI para evaluar el pipeline sobre un directorio de imágenes con ground truth."""

import argparse
import json
import time
from pathlib import Path

import cv2

from alpr.core.pipeline import ALPRPipeline
from alpr.eval.metrics import evaluate_batch


def load_ground_truth(annotations_path: Path) -> dict:
    """Carga anotaciones en formato {filename: {'text': 'ABC1D23', 'bbox': [...]}}."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación end-to-end ALPR")
    parser.add_argument("--images", required=True, type=Path, help="Directorio de imágenes")
    parser.add_argument("--annotations", required=True, type=Path, help="JSON de ground truth")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_results.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = ALPRPipeline(device=args.device)
    gt = load_ground_truth(args.annotations)

    predictions = []
    ground_truth_list = []
    latencies = []

    for filename, ann in gt.items():
        img_path = args.images / filename
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        start = time.perf_counter()
        result = pipeline.run(image)
        latencies.append((time.perf_counter() - start) * 1000)

        # Tomar la placa de mayor confianza
        best = max(result["plates"], key=lambda p: p["confidence"]) if result["plates"] else None
        predictions.append(
            {
                "text": best["plate"] if best else "",
                "bbox": best["bbox"] if best else {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
            }
        )
        ground_truth_list.append(
            {
                "text": ann["text"],
                "bbox": ann.get("bbox", {"x1": 0, "y1": 0, "x2": 0, "y2": 0}),
            }
        )

    metrics = evaluate_batch(predictions, ground_truth_list)
    metrics["latency_ms_mean"] = sum(latencies) / len(latencies) if latencies else 0.0
    metrics["latency_ms_p95"] = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
    metrics["samples"] = len(latencies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
