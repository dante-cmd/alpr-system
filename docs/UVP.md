# Utility Validation Protocol (UVP)

Ponderaciones:

```text
Utility Score = 0.30 * Evidencia + 0.20 * Rendimiento + 0.20 * Robustez + 0.15 * Mantenibilidad + 0.15 * Escalabilidad
```

Reglas:

- Score ≥ 4.0 → aprobado
- 3.0–4.0 → requiere experimento
- < 3.0 → rechazado

## Componentes aprobados

### YOLOv8 (detección de vehículos y placas)

```yaml
componente: YOLOv8n/s
evidencia: 5
mantenibilidad: 5
rendimiento: 4
robustez: 4
escalabilidad: 5
utility_score: 4.55
justificacion:
  - Papers revisados por pares (Laroca IJCNN 2018, IET ITS 2021; Satya ETASR 2025)
  - Benchmarks públicos COCO y ALPR
  - Ultralytics >40k estrellas, mantenimiento activo
  - Uso industrial documentado
```

### PaddleOCR PP-OCRv4 (OCR)

```yaml
componente: PaddleOCR PP-OCRv4
evidencia: 5
mantenibilidad: 5
rendimiento: 4
robustez: 4
escalabilidad: 5
utility_score: 4.55
justificacion:
  - >70k estrellas GitHub (PaddleOCR 3.0 Technical Report, arXiv 2025)
  - Usado en producción por MinerU, RAGFlow, UmiOCR
  - Benchmarks en inglés/latino y chino
  - OpenVINO optimizado para CPU
```

### FastAPI + ONNX Runtime / OpenVINO (serving)

```yaml
componente: FastAPI + ONNX Runtime OpenVINO EP
evidencia: 4
mantenibilidad: 5
rendimiento: 4
robustez: 4
escalabilidad: 4
utility_score: 4.15
justificacion:
  - FastAPI es estándar en APIs ML
  - ONNX Runtime con OpenVINO EP benchmarkeado en CPU
  - Documentación extensa y comunidad activa
```

## Resumen

Todos los componentes principales superan el umbral de 4.0 y tienen evidencia previa documentada.
