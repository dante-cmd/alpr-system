# Fase 3: Diseño del pipeline

```text
Imagen / Frame
    ↓
[Preprocesamiento] resize, normalización
    ↓
[Detección de vehículo] YOLOv8n COCO → filtro car/motorcycle/truck/bus
    ↓
[Detección de placa] YOLOv8n/s fine-tuned → bbox
    ↓
[Corrección de perspectiva] recorte + CLAHE (homografía con 4 esquinas si están disponibles)
    ↓
[OCR] PaddleOCR PP-OCRv4 → texto + score
    ↓
[Postprocesamiento] regex Mercosur, filtro confianza, NMS temporal
    ↓
[Salida] JSON con placa, bbox, confianza, metadata
```

## Etapas

| Etapa | Algoritmo | Alternativas | Costo computacional |
|-------|-----------|--------------|---------------------|
| Preprocesamiento | Resize a 1280px manteniendo aspecto | Letterbox, padding | Bajo |
| Detección vehículo | YOLOv8n COCO | YOLOv11, RT-DETR | ~30 ms CPU |
| Detección placa | YOLOv8n/s fine-tuned | RT-DETR, PaddleDetection | ~30-50 ms CPU |
| Corrección | Recorte + CLAHE | Homografía, SR | ~5 ms CPU |
| OCR | PaddleOCR PP-OCRv4 | EasyOCR, TrOCR | ~50-100 ms CPU |
| Postprocesado | Regex + NMS | CRF, diccionario | <1 ms CPU |

## Selección de modelos

| Modelo | Aplicación | mAP50 | FPS CPU | Producción |
|--------|-----------|-------|---------|------------|
| YOLOv8n | Vehículo/placa | 0.918 | ~30 | Alta |
| YOLOv8s | Placa | 0.933 | ~20 | Alta |
| PaddleOCR v4 | OCR | 70.1 % en | ~10-20 | Alta |

## Costo total estimado

- CPU (YOLOv8n + PaddleOCR): ~100-200 ms por imagen 720p.
- Memoria: ~2 GB con modelos cargados.
