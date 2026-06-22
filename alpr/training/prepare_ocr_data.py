"""Prepara crops de placas para entrenar PaddleOCR.

Lee un dataset en formato YOLO (imagen + .txt con bbox y texto) y genera
una estructura compatible con PaddleOCR:
  rec_gt_train.txt:  <ruta_imagen>\t<texto>
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path, help="JSON {filename: {text, bbox}}")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    out_dir = args.output / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    gt_lines = []
    for filename, ann in annotations.items():
        img_path = args.images / filename
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        bbox = ann["bbox"]
        crop = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        if crop.size == 0:
            continue
        out_name = f"{Path(filename).stem}.jpg"
        crop_path = out_dir / out_name
        cv2.imwrite(str(crop_path), crop)
        gt_lines.append(f"{crop_path}\t{ann['text']}\n")

    with open(args.output / f"rec_gt_{args.split}.txt", "w", encoding="utf-8") as f:
        f.writelines(gt_lines)

    print(f"Generados {len(gt_lines)} crops en {out_dir}")


if __name__ == "__main__":
    main()
