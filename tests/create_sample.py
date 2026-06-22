"""Crea una imagen sintética de placa Mercosur para pruebas."""

from pathlib import Path

import cv2
import numpy as np


def create_plate_image(text: str, output: Path) -> None:
    width, height = 300, 80
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Borde
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), (0, 0, 0), 2)
    # Texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"Created {output}")


if __name__ == "__main__":
    create_plate_image("ABC1D23", Path("data/samples/plate_abc1d23.jpg"))
