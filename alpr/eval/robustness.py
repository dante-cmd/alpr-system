"""Evaluación de robustez bajo condiciones adversas."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from alpr.core.pipeline import ALPRPipeline


def apply_condition(image: np.ndarray, condition: str) -> np.ndarray:
    """Aplica una transformación que simula condición adversa."""
    if condition == "original":
        return image
    if condition == "night":
        # Reducir brillo y aumentar contraste
        darker = cv2.convertScaleAbs(image, alpha=0.4, beta=-30)
        return darker
    if condition == "rain":
        # Ruido gaussiano ligero + desenfoque
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(noisy, (3, 3), 0)
    if condition == "motion_blur":
        kernel = np.zeros((5, 5))
        kernel[2, :] = np.ones(5) / 5
        return cv2.filter2D(image, -1, kernel)
    if condition == "low_resolution":
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if condition == "occlusion":
        # Cubrir 15 % inferior con rectángulo negro
        h, w = image.shape[:2]
        occluded = image.copy()
        occluded[int(h * 0.85):, :] = 0
        return occluded
    if condition == "angle":
        # Rotación leve simulando inclinación
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    if condition == "dirt":
        # Manchas aleatorias
        dirty = image.copy()
        for _ in range(20):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            cv2.circle(dirty, (x, y), np.random.randint(2, 8), (0, 0, 0), -1)
        return dirty
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de robustez ALPR")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness_results.json"))
    parser.add_argument("--conditions", nargs="+", default=[
        "original", "night", "rain", "motion_blur", "low_resolution",
        "occlusion", "angle", "dirt",
    ])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = ALPRPipeline(device=args.device)
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    results = {}
    for condition in args.conditions:
        correct = 0
        total = 0
        latencies = []
        for filename, ann in annotations.items():
            img_path = args.images / filename
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            transformed = apply_condition(image, condition)
            start = time.perf_counter()
            res = pipeline.run(transformed)
            latencies.append((time.perf_counter() - start) * 1000)
            best = max(res["plates"], key=lambda p: p["confidence"]) if res["plates"] else None
            pred = best["plate"] if best else ""
            total += 1
            if pred == ann["text"]:
                correct += 1
        accuracy = correct / total if total else 0.0
        results[condition] = {
            "accuracy": accuracy,
            "samples": total,
            "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
