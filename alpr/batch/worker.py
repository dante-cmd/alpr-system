"""Worker batch para procesar un directorio de imágenes."""

import argparse
import json
import time
from pathlib import Path

import cv2

from alpr.core.pipeline import ALPRPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ALPR")
    parser.add_argument("--input", required=True, type=Path, help="Directorio de imágenes")
    parser.add_argument("--output", required=True, type=Path, help="Archivo JSON de salida")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = ALPRPipeline(device=args.device)
    results = []

    for img_path in sorted(args.input.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        start = time.perf_counter()
        res = pipeline.run(image)
        res["filename"] = img_path.name
        res["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        results.append(res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Procesadas {len(results)} imágenes. Resultados en {args.output}")


if __name__ == "__main__":
    main()
