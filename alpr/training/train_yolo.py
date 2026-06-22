"""Fine-tuning de YOLOv8 para detección de placas."""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 para placas")
    parser.add_argument("--data", required=True, help="Ruta a data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Modelo base")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'cpu' o índice GPU")
    parser.add_argument("--project", default="outputs/yolo_training")
    parser.add_argument("--name", default="plate_detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=10,
        save=True,
    )
    # Exportar a ONNX para serving optimizado
    best = YOLO(f"{args.project}/{args.name}/weights/best.pt")
    best.export(format="onnx", imgsz=args.imgsz, dynamic=True)


if __name__ == "__main__":
    main()
